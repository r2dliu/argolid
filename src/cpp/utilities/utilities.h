#pragma once
#include <string>
#include <memory>
#include <vector>
#include <cmath>
#include <tuple>

#include "tensorstore/tensorstore.h"
#include "tensorstore/spec.h"

namespace argolid {
enum VisType {Viv, NG_Zarr, PCNG};

enum class DSType {Mean, Mode_Max, Mode_Min};

struct ImageInfo
{
  std::int64_t _full_image_height, _full_image_width, _chunk_size_x, _chunk_size_y, _num_channels;
  std::string _data_type;
};

tensorstore::Spec GetOmeTiffSpecToRead(const std::string& filename);
//tensorstore::Spec GetZarrSpecToRead(const std::string& filename, const std::string& scale_key);
tensorstore::Spec GetZarrSpecToRead(const std::string& filename);
tensorstore::Spec GetZarrSpecToWrite(   const std::string& filename, 
                                        const std::vector<std::int64_t>& image_shape, 
                                        const std::vector<std::int64_t>& chunk_shape,
                                        const std::string& dtype);
tensorstore::Spec GetNPCSpecToRead(const std::string& filename, const std::string& scale_key);
tensorstore::Spec GetNPCSpecToWrite(const std::string& filename, 
                                    const std::string& scale_key,
                                    const std::vector<std::int64_t>& image_shape, 
                                    const std::vector<std::int64_t>& chunk_shape,
                                    std::int64_t resolution,
                                    std::int64_t num_channels,
                                    std::string_view dtype,
                                    bool base_level);


uint16_t GetDataTypeCode (std::string_view type_name);
std::string GetUTCString();
void WriteTSZattrFile(const std::string& tiff_file_name, const std::string& zattr_file_loc, int min_level, int max_level);
void WriteVivZattrFile(const std::string& tiff_file_name, const std::string& zattr_file_loc, int min_level, int max_level);
void WriteVivZgroupFiles(const std::string& output_loc);
void ExtractAndWriteXML(const std::string& input_file, const std::string& xml_loc);
void WriteMultiscaleMetadataForImageCollection(const std::string& image_file_name , const std::string& output_dir, 
                                                int min_level, int max_level, VisType v, ImageInfo& whole_image);
void GenerateOmeXML(const std::string& image_name, const std::string& output_file, ImageInfo& whole_image);
void WriteMultiscaleMetadataForSingleFile( const std::string& input_file , const std::string& output_dir, 
                                                                    int min_level, int max_level, VisType v);
inline std::tuple<int,int,int,int> GetZarrParams(VisType v){
  // returns {x_dim_index, y_dim_index, c_dim_index, num_dims}
  if (v == VisType::Viv){ //5D file
    return {4,3,1,5};
  } else if (v == VisType::NG_Zarr ){ // 3D file
    return {3,2,0,4};
  } else if (v == VisType::PCNG ){ // 3D file
    return {0,1,3,3};
  }
}

std::optional<std::tuple<std::uint32_t, std::uint32_t>> GetTiffDims (const std::string filename);

void CreateXML(
    const std::string& image_name, 
    const std::string& output_file, 
    const std::vector<std::int64_t>& image_shape,
    const std::string& image_dtype);

void WriteTSZattrFilePlateImage(
    const std::string& tiff_file_name, 
    const std::string& zarr_root_dir, 
    const std::unordered_map<int, std::vector<std::int64_t>>& plate_image_shape);

std::tuple<std::optional<int>, std::optional<int>, std::optional<int>>ParseMultiscaleMetadata(const std::string& axes_list, int len);

// custom hashing for using std::tuple as key
struct TupleHash {
    template <typename... Args>
    std::size_t operator()(const std::tuple<Args...>& t) const {
        return std::apply([](const auto&... args) {
            return (std::hash<std::decay_t<decltype(args)>>{}(args) ^ ...);
        }, t);
    }
};
} // ns argolid