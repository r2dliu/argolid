#include "pyramid_compositor.h"

#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cmath>

#include "tensorstore/context.h"
#include "tensorstore/array.h"
#include "tensorstore/driver/zarr/dtype.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/open.h"

#include <nlohmann/json.hpp>

using ::tensorstore::internal_zarr::ChooseBaseDType;
using json = nlohmann::json;

namespace fs = std::filesystem;

namespace argolid {
PyramidCompositor::PyramidCompositor(const std::string& input_pyramids_loc, const std::string& out_dir, const std::string& output_pyramid_name): 
    _input_pyramids_loc(input_pyramids_loc),
    _output_pyramid_name(std::filesystem::path(out_dir) / output_pyramid_name),
    _ome_metadata_file(out_dir + "/" + output_pyramid_name + "/METADATA.ome.xml"),
    _pyramid_levels(-1),
    _num_channels(-1) {}


void PyramidCompositor::create_xml() {
    CreateXML(
        this->_output_pyramid_name.u8string(), 
        this->_ome_metadata_file,
        this->_plate_image_shapes[0],
        this->_image_dtype
    );
}

void PyramidCompositor::create_zattr_file() {
    WriteTSZattrFilePlateImage(
        this->_output_pyramid_name.u8string(),
        this->_output_pyramid_name.u8string() + "/data.zarr/0",
        this->_plate_image_shapes
    );
}

void PyramidCompositor::create_zgroup_file() {
    WriteVivZgroupFiles(this->_output_pyramid_name.u8string());
}

void PyramidCompositor::create_auxiliary_files() {
    create_xml();
    create_zattr_file();
    create_zgroup_file();
}

void PyramidCompositor::write_zarr_chunk(int level, int channel, int y_index, int x_index) {

    std::tuple<int, int, int, int> chunk = std::make_tuple(level, channel, y_index, x_index);

    if (_chunk_cache.find(chunk) != _chunk_cache.end()) {
        return;
    }

    if (_composition_map.empty()) {
        std::cerr << "No composition map is set. Unable to generate pyramid" << std::endl;
        return;
    }

    if (_unit_image_shapes.find(level) == _unit_image_shapes.end()) {
        std::cerr << "Requested level (" + std::to_string(level) + ") does not exist" << std::endl;
        return;
    }

    if (channel >= _num_channels) {
        std::cerr << "Requested channel (" + std::to_string(channel) + ") does not exist" << std::endl;
        return;
    }

    auto plate_shape_it = _plate_image_shapes.find(level);
    if (plate_shape_it == _plate_image_shapes.end()) {
        std::cerr << "No plate image shapes found for level (" + std::to_string(level) + ")" << std::endl;
        return;
    }

    const auto& plate_shape = plate_shape_it->second;
    if (plate_shape.size() < 4 || y_index > (plate_shape[3] / CHUNK_SIZE)) {
        std::cerr << "Requested y index (" + std::to_string(y_index) + ") does not exist" << std::endl;
        return;
    }

    if (plate_shape.size() < 5 || x_index > (plate_shape[4] / CHUNK_SIZE)) {
        std::cerr << "Requested x index (" + std::to_string(x_index) + ") does not exist" << std::endl;
        return;
    }

    // get datatype by opening
    std::string input_file_name = _composition_map[{x_index, y_index, channel}];
    std::filesystem::path zarrArrayLoc = _output_pyramid_name / input_file_name / std::filesystem::path("data.zarr/0/") / std::to_string(level);

    auto read_spec = GetZarrSpecToRead(zarrArrayLoc.u8string());

    tensorstore::TensorStore<void, -1, tensorstore::ReadWriteMode::dynamic> source;

    TENSORSTORE_CHECK_OK_AND_ASSIGN(source, tensorstore::Open(
                read_spec,
                tensorstore::OpenMode::open,
                tensorstore::ReadWriteMode::read).result());
    
    auto data_type = GetDataTypeCode(source.dtype().name());

    switch(data_type){
        case 1:
            _write_zarr_chunk<uint8_t>(level, channel, y_index, x_index);
            break;
        case 2:
            _write_zarr_chunk<uint16_t>(level, channel, y_index, x_index);
            break;
        case 4:
            _write_zarr_chunk<uint32_t>(level, channel, y_index, x_index);
            break;
        case 8:
            _write_zarr_chunk<uint64_t>(level, channel, y_index, x_index);
            break;
        case 16:
            _write_zarr_chunk<int8_t>(level, channel, y_index, x_index);
            break;
        case 32:
            _write_zarr_chunk<int16_t>(level, channel, y_index, x_index);
            break;
        case 64:
            _write_zarr_chunk<int32_t>(level, channel, y_index, x_index);
            break;
        case 128:
            _write_zarr_chunk<int64_t>(level, channel, y_index, x_index);
            break;
        case 256:
            _write_zarr_chunk<float>(level, channel, y_index, x_index);
            break;
        case 512:
            _write_zarr_chunk<double>(level, channel, y_index, x_index);
            break;
        default:
            break;
        }
    
    _chunk_cache.insert(chunk);
}

template <typename T>
void PyramidCompositor::_write_zarr_chunk(int level, int channel, int y_index, int x_index) {

    // Compute ranges in global coordinates
    auto image_shape = _zarr_arrays[level].domain().shape();
    int y_start = y_index * CHUNK_SIZE;
    int y_end = std::min((y_index + 1) * CHUNK_SIZE, (int)image_shape[3]);
    int x_start = x_index * CHUNK_SIZE;
    int x_end = std::min((x_index + 1) * CHUNK_SIZE, (int)image_shape[4]);

    int assembled_width = x_end - x_start;
    int assembled_height = y_end - y_start;
    
    std::vector<T> assembled_image_vec(assembled_width*assembled_height);

    // Determine required input images
    int unit_image_height = _unit_image_shapes[level].first;
    int unit_image_width = _unit_image_shapes[level].second;

    int row_start_pos = y_start;

    //int row, local_y_start, tile_y_start, tile_y_dim, tile_y_end, col_start_pos;
    //int col, local_x_start, tile_x_start, tile_x_dim, tile_x_end;
    while (row_start_pos < y_end) {

        int row = row_start_pos / unit_image_height;
        int local_y_start = row_start_pos - y_start;
        int tile_y_start = row_start_pos - row * unit_image_height;
        int tile_y_dim = std::min((row + 1) * unit_image_height - row_start_pos, y_end - row_start_pos);
        int tile_y_end = tile_y_start + tile_y_dim;

        int col_start_pos = x_start;

        while (col_start_pos < x_end) {

            int col = col_start_pos / unit_image_width;
            int local_x_start = col_start_pos - x_start;
            int tile_x_start = col_start_pos - col * unit_image_width;
            int tile_x_dim = std::min((col + 1) * unit_image_width - col_start_pos, x_end - col_start_pos);
            int tile_x_end = tile_x_start + tile_x_dim;

            _th_pool.detach_task([ 
                row,
                col,
                channel,
                level,
                x_start,
                x_end,
                local_x_start, 
                tile_x_start, 
                tile_x_dim, 
                tile_x_end,
                local_y_start, 
                tile_y_start, 
                tile_y_dim, 
                tile_y_end,
                y_start,
                y_end,
                &assembled_image_vec, 
                assembled_width,
                this
            ]() {
                // Fetch input file path from composition map
                std::string input_file_name = _composition_map[{col, row, channel}];
                std::filesystem::path zarrArrayLoc = std::filesystem::path(input_file_name) / "data.zarr/0/" / std::to_string(level);

                auto read_spec = GetZarrSpecToRead(zarrArrayLoc.u8string());

                tensorstore::TensorStore<void, -1, tensorstore::ReadWriteMode::dynamic> source;

                if (_zarr_readers.find(zarrArrayLoc.u8string()) != _zarr_readers.end()) {
                    source = _zarr_readers[zarrArrayLoc.u8string()];
                } else {
                    TENSORSTORE_CHECK_OK_AND_ASSIGN(source, tensorstore::Open(
                                read_spec,
                                tensorstore::OpenMode::open,
                                tensorstore::ReadWriteMode::read).result());

                    _zarr_readers[zarrArrayLoc.u8string()] = source;
                }

                std::vector<T> read_buffer((tile_x_end-tile_x_start)*(tile_y_end-tile_y_start));
                auto array = tensorstore::Array(read_buffer.data(), {tile_y_end-tile_y_start, tile_x_end-tile_x_start}, tensorstore::c_order);

                tensorstore::IndexTransform<> read_transform = tensorstore::IdentityTransform(source.domain());

                Seq rows = Seq(y_start, y_end);
                Seq cols = Seq(x_start, x_end); 
                Seq tsteps = Seq(0, 0);
                Seq channels = Seq(channel-1, channel); 
                Seq layers = Seq(0, 0);
            
                read_transform = (std::move(read_transform) | tensorstore::Dims(_y_index).ClosedInterval(rows.Start(), rows.Stop()-1) |
                                                            tensorstore::Dims(_x_index).ClosedInterval(cols.Start(), cols.Stop()-1)).value();

                tensorstore::Read(source | read_transform, tensorstore::UnownedToShared(array)).value();

                for (int i = 0; i < tile_y_end - tile_y_start; ++i) {
                    std::memcpy(
                        &assembled_image_vec[(local_y_start + i) * assembled_width + local_x_start], // Destination
                        &read_buffer[i * (tile_x_end - tile_x_start)], // Source
                        (tile_x_end - tile_x_start) * sizeof(assembled_image_vec[0]) // Number of bytes
                    );
                }
            });

            col_start_pos += tile_x_end - tile_x_start;
        }
        row_start_pos += tile_y_end - tile_y_start;
    }

    // wait for threads to finish assembling vector before writing
    _th_pool.wait();

    WriteImageData(
        _zarr_arrays[level],
        assembled_image_vec,  // Access the data from the std::shared_ptr
        Seq(y_start, y_end), 
        Seq(x_start, x_end), 
        Seq(0, 0), 
        Seq(channel, channel), 
        Seq(0, 0)
    );
}

void PyramidCompositor::set_composition(const std::unordered_map<std::tuple<int, int, int>, std::string, TupleHash>& comp_map) {
    _composition_map = comp_map;

    for (const auto& coord : comp_map) {
        std::string file_path = coord.second;
        std::filesystem::path attr_file_loc = std::filesystem::path(file_path) / "data.zarr/0/.zattrs";

        if (std::filesystem::exists(attr_file_loc)) {
            std::ifstream attr_file(attr_file_loc);
            json attrs;
            attr_file >> attrs;

            const auto& multiscale_metadata = attrs["multiscales"][0]["datasets"];
            _pyramid_levels = multiscale_metadata.size();
            for (const auto& dic : multiscale_metadata) {

                _th_pool.detach_task([
                    dic,
                    file_path,
                    this
                ](){
                    std::string res_key = dic["path"];

                    std::filesystem::path zarr_array_loc = std::filesystem::path(file_path) / "data.zarr/0/" / res_key;

                    auto read_spec = GetZarrSpecToRead(zarr_array_loc.u8string());

                    tensorstore::TensorStore<void, -1, tensorstore::ReadWriteMode::dynamic> source;

                    TENSORSTORE_CHECK_OK_AND_ASSIGN(source, tensorstore::Open(
                                read_spec,
                                tensorstore::OpenMode::open,
                                tensorstore::ReadWriteMode::read).result());

                    // store reader to use later
                    _zarr_readers[zarr_array_loc.u8string()] = source;
                    
                    auto image_shape = source.domain().shape();
                    _image_ts_dtype = source.dtype();
                    _image_dtype = source.dtype().name();
                    _image_dtype_code = GetDataTypeCode(_image_dtype);

                    _unit_image_shapes[std::stoi(res_key)] = {
                        image_shape[image_shape.size()-2],
                        image_shape[image_shape.size()-1]
                    };
                });
            }
            break;
        }
    }

    _th_pool.wait();

    int num_rows = 0, num_cols = 0;
    _num_channels = 0;
    for (const auto& coord : comp_map) {
        num_rows = std::max(num_rows, std::get<1>(coord.first));
        num_cols = std::max(num_cols, std::get<0>(coord.first));
        _num_channels = std::max(_num_channels, std::get<2>(coord.first));
    }

    num_cols += 1;
    num_rows += 1;
    _num_channels += 1;

    _plate_image_shapes.clear();
    _zarr_arrays.clear();
    _chunk_cache.clear();

    for (auto& [level, shape]: _unit_image_shapes) {
        
        _plate_image_shapes[level] = std::vector<std::int64_t> {
            1,
            _num_channels,
            1,
            num_rows * shape.first,
            num_cols * shape.second
        };
    }

    for (const auto& [level, shape] : _unit_image_shapes) {
        std::string path = _output_pyramid_name.u8string() + "/data.zarr/0/" + std::to_string(level);

        _th_pool.detach_task([
                    path,
                    level=level,
                    this
        ](){
            tensorstore::TensorStore<void, -1, tensorstore::ReadWriteMode::dynamic> source;

            TENSORSTORE_CHECK_OK_AND_ASSIGN(source, tensorstore::Open(
                        GetZarrSpecToWrite(
                            path,
                            _plate_image_shapes[level],
                            {1,1,1, CHUNK_SIZE, CHUNK_SIZE},
                            ChooseBaseDType(_image_ts_dtype).value().encoded_dtype
                        ),
                        tensorstore::OpenMode::create |
                        tensorstore::OpenMode::delete_existing,
                        tensorstore::ReadWriteMode::write).result());
            
            _zarr_arrays[level] = source;
        });
    }

    _th_pool.wait();

    create_auxiliary_files();
}

void PyramidCompositor::reset_composition() {
    fs::remove_all(_output_pyramid_name);
    _composition_map.clear();
    _plate_image_shapes.clear();
    _chunk_cache.clear();
    _plate_image_shapes.clear();
    _zarr_arrays.clear();
}

template <typename T>
void PyramidCompositor::WriteImageData(
    tensorstore::TensorStore<void, -1, tensorstore::ReadWriteMode::dynamic>& source,
    std::vector<T>& image,
    const Seq& rows,
    const Seq& cols,
    const std::optional<Seq>& layers,
    const std::optional<Seq>& channels,
    const std::optional<Seq>& tsteps) {

    std::vector<std::int64_t> shape;

    auto result_array = tensorstore::Array(image.data(), {rows.Stop()-rows.Start(), cols.Stop()-cols.Start()}, tensorstore::c_order);

    auto output_transform = tensorstore::IdentityTransform(source.domain());

    output_transform = (std::move(output_transform) | tensorstore::Dims(_c_index).ClosedInterval(channels.value().Start(), channels.value().Stop())).value();

    output_transform = (std::move(output_transform) | tensorstore::Dims(_y_index).ClosedInterval(rows.Start(), rows.Stop()-1) |
                                                      tensorstore::Dims(_x_index).ClosedInterval(cols.Start(), cols.Stop()-1)).value();

    shape.emplace_back(rows.Stop() - rows.Start()+1);
    shape.emplace_back(cols.Stop() - cols.Start()+1);

    // Write data array to TensorStore
    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(result_array), source | output_transform).result();

    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }
}
} // ns argolid 