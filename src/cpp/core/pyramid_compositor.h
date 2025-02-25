#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <set>
#include <filesystem>
#include <tuple>
#include "BS_thread_pool.hpp"

#include "tensorstore/tensorstore.h"
#include "tensorstore/spec.h"

#include "../utilities/utilities.h"

namespace fs = std::filesystem;
namespace argolid {
static constexpr int CHUNK_SIZE = 1024;

class Seq
{
    private:
        long start_index_, stop_index_, step_;
    public:
        inline Seq(const long start, const long  stop, const long  step=1):start_index_(start), stop_index_(stop), step_(step){} 
        inline long Start()  const  {return start_index_;}
        inline long Stop()  const {return stop_index_;}
        inline long Step()  const {return step_;}
};

class PyramidCompositor {
public:

    PyramidCompositor(const std::string& well_pyramid_loc, const std::string& out_dir, const std::string& pyramid_file_name);

    void reset_composition();

    void create_xml();
    void create_zattr_file();
    void create_zgroup_file();
    void create_auxiliary_files();

    void write_zarr_chunk(int level, int channel, int y_index, int x_index);

    void set_composition(const std::unordered_map<std::tuple<int, int, int>, std::string, TupleHash>& comp_map);

private:

    template <typename T>
    void _write_zarr_chunk(int level, int channel, int y_index, int x_index);
    
    template <typename T>
    void WriteImageData(
        tensorstore::TensorStore<void, -1, tensorstore::ReadWriteMode::dynamic>& source,
        std::vector<T>& image,
        const Seq& rows,
        const Seq& cols,
        const std::optional<Seq>& layers,
        const std::optional<Seq>& channels,
        const std::optional<Seq>& tsteps);

    std::string _input_pyramids_loc;
    std::filesystem::path _output_pyramid_name;
    std::string _ome_metadata_file;

    std::unordered_map<int, std::vector<std::int64_t>> _plate_image_shapes;

    std::unordered_map<int, tensorstore::TensorStore<void, -1, tensorstore::ReadWriteMode::dynamic>> _zarr_arrays;

    std::unordered_map<std::string, tensorstore::TensorStore<void, -1, tensorstore::ReadWriteMode::dynamic>> _zarr_readers;

    std::unordered_map<int, std::pair<int, int>> _unit_image_shapes;

    std::unordered_map<std::tuple<int, int, int>, std::string, TupleHash> _composition_map;

    std::set<std::tuple<int, int, int, int>> _chunk_cache;
    int _pyramid_levels;
    int _num_channels;

    tensorstore::DataType _image_ts_dtype;
    std::string _image_dtype;
    std::uint16_t _image_dtype_code;

    int _x_index = 4;
    int _y_index = 3;
    int _c_index = 1;

    BS::thread_pool<BS::tp::none> _th_pool;
};
} // ns argolid
