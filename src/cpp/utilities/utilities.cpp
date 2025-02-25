#include <iomanip>
#include <ctime>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <plog/Log.h>
#include <tiffio.h>
#include "pugixml.hpp"
#include <nlohmann/json.hpp>
#include "utilities.h"
#include <thread>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace argolid {
tensorstore::Spec GetOmeTiffSpecToRead(const std::string& filename){
    return tensorstore::Spec::FromJson({{"driver", "ometiff"},

                            {"kvstore", {{"driver", "tiled_tiff"},
                                         {"path", filename}}
                            },
                            {"context", {
                              {"cache_pool", {{"total_bytes_limit", 1000000000}}},
                              {"data_copy_concurrency", {{"limit", std::thread::hardware_concurrency()}}},
                              {"file_io_concurrency", {{"limit", std::thread::hardware_concurrency()}}},
                            }},
                            }).value();
}

tensorstore::Spec GetZarrSpecToWrite(   const std::string& filename, 
                                        const std::vector<std::int64_t>& image_shape, 
                                        const std::vector<std::int64_t>& chunk_shape,
                                        const std::string& dtype){
    return tensorstore::Spec::FromJson({{"driver", "zarr"},
                            {"kvstore", {{"driver", "file"},
                                         {"path", filename}}
                            },
                            {"context", {
                              {"cache_pool", {{"total_bytes_limit", 1000000000}}},
                              {"data_copy_concurrency", {{"limit", std::thread::hardware_concurrency()}}},
                              {"file_io_concurrency", {{"limit", std::thread::hardware_concurrency()}}},
                            }},
                            {"metadata", {
                                          {"zarr_format", 2},
                                          {"shape", image_shape},
                                          {"chunks", chunk_shape},
                                          {"dtype", dtype},
                                          },
                            }}).value();
}

tensorstore::Spec GetZarrSpecToRead(const std::string& filename){
    return tensorstore::Spec::FromJson({{"driver", "zarr"},
                            {"kvstore", {{"driver", "file"},
                                         {"path", filename}}
                            }
                            }).value();
}

tensorstore::Spec GetNPCSpecToRead(const std::string& filename, const std::string& scale_key){
    return tensorstore::Spec::FromJson({{"driver", "neuroglancer_precomputed"},
                            {"kvstore", {{"driver", "file"},
                                         {"path", filename}}
                            },
                            {"scale_metadata", {
                                            {"key", scale_key},
                                            },
                            }}).value();

}

tensorstore::Spec GetNPCSpecToWrite(const std::string& filename, 
                                    const std::string& scale_key,
                                    const std::vector<std::int64_t>& image_shape, 
                                    const std::vector<std::int64_t>& chunk_shape,
                                    std::int64_t resolution,
                                    std::int64_t num_channels,
                                    std::string_view dtype, bool base_level){
    if (base_level){
      return tensorstore::Spec::FromJson({{"driver", "neuroglancer_precomputed"},
                              {"kvstore", {{"driver", "file"},
                                          {"path", filename}}
                              },
                              {"context", {
                                {"cache_pool", {{"total_bytes_limit", 1000000000}}},
                                {"data_copy_concurrency", {{"limit", std::thread::hardware_concurrency()}}},
                                {"file_io_concurrency", {{"limit", std::thread::hardware_concurrency()}}},
                              }},
                              {"multiscale_metadata", {
                                            {"data_type", dtype},
                                            {"num_channels", num_channels},
                                            {"type", "image"},
                              }},
                              {"scale_metadata", {
                                            {"encoding", "raw"},
                                            {"key", scale_key},
                                            {"size", image_shape},
                                            {"chunk_size", chunk_shape},
                                            {"resolution", {resolution, resolution, 1}}
                                            },
                              }}).value();
    } else {
      return tensorstore::Spec::FromJson({{"driver", "neuroglancer_precomputed"},
                        {"kvstore", {{"driver", "file"},
                                      {"path", filename}}
                        },
                        {"context", {
                          {"cache_pool", {{"total_bytes_limit", 1000000000}}},
                          {"data_copy_concurrency", {{"limit", std::thread::hardware_concurrency()}}},
                          {"file_io_concurrency", {{"limit", std::thread::hardware_concurrency()}}},
                        }},
                        {"scale_metadata", {
                                      {"encoding", "raw"},
                                      {"key", scale_key},
                                      {"size", image_shape},
                                      {"chunk_size", chunk_shape},
                                      {"resolution", {resolution, resolution, 1}}
                                      },
                        }}).value();
    }

}

uint16_t GetDataTypeCode (std::string_view type_name){

  if (type_name == std::string_view{"uint8"}) {return 1;}
  else if (type_name == std::string_view{"uint16"}) {return 2;}
  else if (type_name == std::string_view{"uint32"}) {return 4;}
  else if (type_name == std::string_view{"uint64"}) {return 8;}
  else if (type_name == std::string_view{"int8"}) {return 16;}
  else if (type_name == std::string_view{"int16"}) {return 32;}
  else if (type_name == std::string_view{"int32"}) {return 64;}
  else if (type_name == std::string_view{"int64"}) {return 128;}
  else if (type_name == std::string_view{"float32"}) {return 256;}
  else if (type_name == std::string_view{"float64"}) {return 512;}
  else {return 2;}
}

std::string GetUTCString() {
    // Get the current UTC time
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);

    // Convert UTC time to a string
    const int bufferSize = 16; // Sufficient size for most date/time formats
    char buffer[bufferSize];
    std::tm timeInfo;

#if defined(_WIN32)
    // Use gmtime_s instead of gmtime to get the UTC time on Windows
    gmtime_s(&timeInfo, &time);
#else
    // On other platforms, use the standard gmtime function
    gmtime_r(&time, &timeInfo);
#endif
    // Format the time string (You can modify the format as per your requirements)
    std::strftime(buffer, bufferSize, "%Y%m%d%H%M%S", &timeInfo);

    return std::string(buffer);
}

void WriteTSZattrFile(const std::string& tiff_file_name, const std::string& zarr_root_dir, int min_level, int max_level){

    json zarr_multiscale_axes;
    zarr_multiscale_axes = json::parse(R"([
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"}
                ])");
    
    
    float level = 1.0;
    json scale_metadata_list = json::array();
    for(int i=min_level; i<=max_level; ++i){
        json scale_metadata;
        scale_metadata["path"] = std::to_string(i);
        scale_metadata["coordinateTransformations"] = {{{"type", "scale"}, {"scale", {1.0, 1.0, level, level}}}};
        scale_metadata_list.push_back(scale_metadata);
        level = level*2;
    }

    json combined_metadata;
    combined_metadata["datasets"] = scale_metadata_list;
    combined_metadata["version"] = "0.4";
    combined_metadata["axes"] = zarr_multiscale_axes;
    combined_metadata["name"] = tiff_file_name;
    combined_metadata["metadata"] = {{"method", "mean"}};
    json final_formated_metadata;
#if defined(__clang__) || defined(_MSC_VER)
// more details here: https://github.com/nlohmann/json/issues/2311
    final_formated_metadata["multiscales"][0] = {combined_metadata};
#else
    final_formated_metadata["multiscales"] = {combined_metadata};
#endif
    std::ofstream f(zarr_root_dir + "/.zattrs",std::ios_base::trunc |std::ios_base::out);
    if (f.is_open()){
        f << final_formated_metadata;
    } else {
        PLOG_INFO <<"Unable to write .zattr file at "<< zarr_root_dir << ".";
    }
}

void WriteTSZattrFilePlateImage(
    const std::string& tiff_file_name, 
    const std::string& zarr_root_dir, 
    const std::unordered_map<int, std::vector<std::int64_t>>& plate_image_shape){

    json zarr_multiscale_axes;
    zarr_multiscale_axes = json::parse(R"([
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"}
                ])");
    
    
    float level = 1.0;
    json scale_metadata_list = json::array();
    for(const auto& key_val: plate_image_shape){
        json scale_metadata;
        scale_metadata["path"] = std::to_string(std::get<0>(key_val));
        scale_metadata["coordinateTransformations"] = {{{"type", "scale"}, {"scale", {1.0, 1.0, level, level}}}};
        scale_metadata_list.push_back(scale_metadata);
        level = level*2;
    }

    json combined_metadata;
    combined_metadata["datasets"] = scale_metadata_list;
    combined_metadata["version"] = "0.4";
    combined_metadata["axes"] = zarr_multiscale_axes;
    combined_metadata["name"] = tiff_file_name;
    combined_metadata["metadata"] = {{"method", "mean"}};
    json final_formated_metadata;
#if defined(__clang__) || defined(_MSC_VER)
// more details here: https://github.com/nlohmann/json/issues/2311
    final_formated_metadata["multiscales"][0] = {combined_metadata};
#else
    final_formated_metadata["multiscales"] = {combined_metadata};
#endif
    std::ofstream f(zarr_root_dir + "/.zattrs",std::ios_base::trunc |std::ios_base::out);
    if (f.is_open()){
        f << final_formated_metadata;
    } else {
        PLOG_INFO <<"Unable to write .zattr file at "<< zarr_root_dir << ".";
    }
}

void WriteVivZattrFile(const std::string& tiff_file_name, const std::string& zattr_file_loc, int min_level, int max_level){
    
    json scale_metadata_list = json::array();
    for(int i=min_level; i<=max_level; ++i){
        json scale_metadata;
        scale_metadata["path"] = std::to_string(i);
        scale_metadata_list.push_back(scale_metadata);
    }

    json combined_metadata;
    combined_metadata["datasets"] = scale_metadata_list;
    combined_metadata["version"] = "0.1";
    combined_metadata["name"] = tiff_file_name;
    combined_metadata["metadata"] = {{"method", "mean"}};
    json final_formated_metadata;
#if defined(__clang__) || defined(_MSC_VER)
// more details here: https://github.com/nlohmann/json/issues/2311
    final_formated_metadata["multiscales"][0] = {combined_metadata};
#else
    final_formated_metadata["multiscales"] = {combined_metadata};
#endif    
    std::ofstream f(zattr_file_loc + "/.zattrs",std::ios_base::trunc |std::ios_base::out);
    if (f.is_open()){   
        f << final_formated_metadata;
    } else {
        PLOG_INFO << "Unable to write .zattr file at " << zattr_file_loc << ".";
    }
}

void WriteVivZgroupFiles(const std::string& output_loc){
    std::string zgroup_text = "{\"zarr_format\": 2}";
    std::ofstream zgroup_file_1(output_loc+"/data.zarr/.zgroup", std::ios_base::out );
    if(zgroup_file_1.is_open()){
        zgroup_file_1 << zgroup_text << std::endl;
    }
    std::ofstream zgroup_file_2( output_loc + "/data.zarr/0/.zgroup", std::ios_base::out );
    if(zgroup_file_2.is_open()){
        zgroup_file_2 << zgroup_text << std::endl;
    }
}

void ExtractAndWriteXML(const std::string& input_file, const std::string& xml_loc){
    TIFF *tiff_ = TIFFOpen(input_file.c_str(), "r");
    if (tiff_ != nullptr) {
        char* infobuf;
        TIFFGetField(tiff_, TIFFTAG_IMAGEDESCRIPTION , &infobuf);
        char* new_pos = strstr(infobuf, "<OME");
        std::ofstream metadata_file( xml_loc+"/METADATA.ome.xml", std::ios_base::out );
        if(metadata_file.is_open()){
            metadata_file << new_pos << std::endl;
        }
        if(!metadata_file){
            PLOG_INFO << "Unable to write metadata file";
        }
        TIFFClose(tiff_);
    }
}

void WriteMultiscaleMetadataForImageCollection(const std::string& image_file_name , const std::string& output_dir, 
                                                                        int min_level, int max_level, VisType v, ImageInfo& whole_image)
{
    std::string chunked_file_dir = output_dir + "/" + image_file_name + ".zarr";
    if(v == VisType::NG_Zarr){
        WriteTSZattrFile(image_file_name, chunked_file_dir, min_level, max_level);
    } else if (v == VisType::Viv){
        GenerateOmeXML(image_file_name, chunked_file_dir+"/METADATA.ome.xml", whole_image);                   
        WriteVivZattrFile(image_file_name, chunked_file_dir+"/data.zarr/0/", min_level, max_level);
        WriteVivZgroupFiles(chunked_file_dir);
    }
}

void GenerateOmeXML(const std::string& image_name, const std::string& output_file, ImageInfo& whole_image){

    pugi::xml_document doc;

    // Create the root element <OME>
    pugi::xml_node omeNode = doc.append_child("OME");
    
    // Add the namespaces and attributes to the root element
    omeNode.append_attribute("xmlns") = "http://www.openmicroscopy.org/Schemas/OME/2016-06";
    omeNode.append_attribute("xmlns:xsi") = "http://www.w3.org/2001/XMLSchema-instance";
    auto creator = std::string{"Argolid "} + std::string{VERSION_INFO};
    omeNode.append_attribute("Creator") = creator.c_str();
    omeNode.append_attribute("UUID") = "urn:uuid:ce3367ae-0512-4e87-a045-20d87db14001";
    omeNode.append_attribute("xsi:schemaLocation") = "http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd";

    // Create the <Image> element
    pugi::xml_node imageNode = omeNode.append_child("Image");
    imageNode.append_attribute("ID") = "Image:0";
    imageNode.append_attribute("Name") =image_name.c_str();

    // Create the <Pixels> element
    pugi::xml_node pixelsNode = imageNode.append_child("Pixels");
    pixelsNode.append_attribute("BigEndian") = "false";
    pixelsNode.append_attribute("DimensionOrder") = "XYZCT";
    pixelsNode.append_attribute("ID") = "Pixels:0";
    pixelsNode.append_attribute("Interleaved") = "false";
    pixelsNode.append_attribute("SizeC") = std::to_string(whole_image._num_channels).c_str();;
    pixelsNode.append_attribute("SizeT") = "1";
    pixelsNode.append_attribute("SizeX") = std::to_string(whole_image._full_image_width).c_str();
    pixelsNode.append_attribute("SizeY") = std::to_string(whole_image._full_image_height).c_str();
    pixelsNode.append_attribute("SizeZ") = "1";
    pixelsNode.append_attribute("Type") = whole_image._data_type.c_str();

    // Create the <Channel> elements
    for(std::int64_t i=0; i<whole_image._num_channels; ++i){
      pugi::xml_node channelNode = pixelsNode.append_child("Channel");
      channelNode.append_attribute("ID") = ("Channel:0:" + std::to_string(i)).c_str();
      channelNode.append_attribute("SamplesPerPixel") = "1";
      // Create the <LightPath> elements
      channelNode.append_child("LightPath");
    }
  
    doc.save_file(output_file.c_str());
}

void CreateXML(
    const std::string& image_name, 
    const std::string& output_file, 
    const std::vector<std::int64_t>& image_shape,
    const std::string& image_dtype) {


    if (image_shape.size() != 5) {
        throw std::invalid_argument("image_shape must have length 5.");
    }

    pugi::xml_document doc;

    // Create the root element <OME>
    pugi::xml_node omeNode = doc.append_child("OME");
    
    // Add the namespaces and attributes to the root element
    omeNode.append_attribute("xmlns") = "http://www.openmicroscopy.org/Schemas/OME/2016-06";
    omeNode.append_attribute("xmlns:xsi") = "http://www.w3.org/2001/XMLSchema-instance";
    auto creator = std::string{"Argolid "} + std::string{VERSION_INFO};
    omeNode.append_attribute("Creator") = creator.c_str();
    omeNode.append_attribute("UUID") = "urn:uuid:ce3367ae-0512-4e87-a045-20d87db14001";
    omeNode.append_attribute("xsi:schemaLocation") = "http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd";

    // Create the <Image> element
    pugi::xml_node imageNode = omeNode.append_child("Image");
    imageNode.append_attribute("ID") = "Image:0";
    imageNode.append_attribute("Name") = image_name.c_str();

    // Create the <Pixels> element
    pugi::xml_node pixelsNode = imageNode.append_child("Pixels");
    pixelsNode.append_attribute("BigEndian") = "false";
    pixelsNode.append_attribute("DimensionOrder") = "XYZCT";
    pixelsNode.append_attribute("ID") = "Pixels:0";
    pixelsNode.append_attribute("Interleaved") = "false";
    pixelsNode.append_attribute("SizeC") = std::to_string(image_shape[1]).c_str();
    pixelsNode.append_attribute("SizeZ") = std::to_string(image_shape[2]).c_str();
    pixelsNode.append_attribute("SizeT") = std::to_string(image_shape[0]).c_str();
    pixelsNode.append_attribute("SizeX") = std::to_string(image_shape[4]).c_str();
    pixelsNode.append_attribute("SizeY") = std::to_string(image_shape[3]).c_str();
    pixelsNode.append_attribute("Type") = image_dtype.c_str();

    // Create the <Channel> elements
    for(std::int64_t i=0; i<image_shape[1]; ++i){
      pugi::xml_node channelNode = pixelsNode.append_child("Channel");
      channelNode.append_attribute("ID") = ("Channel:0:" + std::to_string(i)).c_str();
      channelNode.append_attribute("SamplesPerPixel") = "1";
      // Create the <LightPath> elements
      channelNode.append_child("LightPath");
    }
  
    doc.save_file(output_file.c_str());
}

void WriteMultiscaleMetadataForSingleFile( const std::string& input_file , const std::string& output_dir, 
                                                                    int min_level, int max_level, VisType v)
{
    std::string tiff_file_name = fs::path(input_file).stem().string();
    std::string chunked_file_dir = output_dir + "/" + tiff_file_name + ".zarr";
    if(v == VisType::NG_Zarr){

        WriteTSZattrFile(tiff_file_name, chunked_file_dir, min_level, max_level);
    } else if (v == VisType::Viv){
        ExtractAndWriteXML(input_file, chunked_file_dir);
        WriteVivZattrFile(tiff_file_name, chunked_file_dir+"/data.zarr/0/", min_level, max_level);
        WriteVivZgroupFiles(chunked_file_dir);
    }
}

std::optional<std::tuple<std::uint32_t, std::uint32_t>> GetTiffDims (const std::string filename){
    TIFF *tiff_ = TIFFOpen(filename.c_str(), "r");
    if (tiff_ != nullptr) {
        uint32_t image_width = 0, image_height = 0;
        TIFFGetField(tiff_, TIFFTAG_IMAGEWIDTH, &image_width);
        TIFFGetField(tiff_, TIFFTAG_IMAGELENGTH, &image_height);
        TIFFClose(tiff_);  
        return {{image_height, image_width}};
    } else {
        return std::nullopt;
    }

}

std::tuple<std::optional<int>, std::optional<int>, std::optional<int>>ParseMultiscaleMetadata(const std::string& axes_list, int len){
    
    std::optional<int> t_index{std::nullopt}, c_index{std::nullopt}, z_index{std::nullopt};

    assert(axes_list.length() <= 5);

    if (axes_list.length() == len){
        // no speculation
        for (int i=0; i<axes_list.length(); ++i){
            if(axes_list[i] == char{'T'}) t_index.emplace(i);
            if(axes_list[i] == char{'C'}) c_index.emplace(i);
            if(axes_list[i] == char{'Z'}) z_index.emplace(i);
        }
    } else // speculate
    {
        if (len == 3) {
            z_index.emplace(0);
        } else if (len == 4) {
            z_index.emplace(1);
            c_index.emplace(0);
        }
    }
    return {t_index, c_index, z_index};
}

} // ns argolid