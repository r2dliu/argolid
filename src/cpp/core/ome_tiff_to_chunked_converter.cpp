#include "ome_tiff_to_chunked_converter.h"
#include "../utilities/utilities.h"
#include <string>
#include <stdint.h>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <chrono>
#include <future>
#include <cmath>

#include "tensorstore/tensorstore.h"
#include "tensorstore/context.h"
#include "tensorstore/array.h"
#include "tensorstore/driver/zarr/dtype.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/open.h"

using ::tensorstore::Context;
using ::tensorstore::internal_zarr::ChooseBaseDType;

namespace argolid {
void OmeTiffToChunkedConverter::Convert( const std::string& input_file, const std::string& output_file, 
                                      const std::string& scale_key, const VisType v, BS::thread_pool<BS::tp::none>& th_pool){
  
  const auto [x_dim, y_dim, c_dim, num_dims] = GetZarrParams(v);

  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store1, tensorstore::Open(GetOmeTiffSpecToRead(input_file),
                            tensorstore::OpenMode::open,
                            tensorstore::ReadWriteMode::read).result());

  auto shape = store1.domain().shape();
  auto data_type = GetDataTypeCode(store1.dtype().name()); 

  std::int64_t image_length = shape[3]; // as per tiled_tiff spec
  std::int64_t image_width = shape[4];
  std::vector<std::int64_t> new_image_shape(num_dims,1);
  std::vector<std::int64_t> chunk_shape(num_dims,1);
  new_image_shape[y_dim] = image_length;
  new_image_shape[x_dim] = image_width;
  auto chunk_layout = store1.chunk_layout().value();
  chunk_shape[y_dim] = static_cast<std::int64_t>(chunk_layout.read_chunk_shape()[3]);
  chunk_shape[x_dim] = static_cast<std::int64_t>(chunk_layout.read_chunk_shape()[4]);

  auto num_rows = static_cast<std::int64_t>(ceil(1.0*image_length/chunk_shape[y_dim]));
  auto num_cols = static_cast<std::int64_t>(ceil(1.0*image_width/chunk_shape[x_dim]));

  auto output_spec = [&](){
    if (v == VisType::NG_Zarr | v == VisType::Viv){
      return GetZarrSpecToWrite(output_file + "/" + scale_key, new_image_shape, chunk_shape, ChooseBaseDType(store1.dtype()).value().encoded_dtype);
    } else if (v == VisType::PCNG){
      return GetNPCSpecToWrite(output_file, scale_key, new_image_shape, chunk_shape, 1, 1, store1.dtype().name(), true);
    } else {
      return tensorstore::Spec();
    }
  }(); 

  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store2, tensorstore::Open(
                            output_spec,
                            tensorstore::OpenMode::create |
                            tensorstore::OpenMode::delete_existing,
                            tensorstore::ReadWriteMode::write).result());

  for(std::int64_t i=0; i<num_rows; ++i){
    std::int64_t y_start = i*chunk_shape[y_dim];
    std::int64_t y_end = std::min({(i+1)*chunk_shape[y_dim], image_length});
    for(std::int64_t j=0; j<num_cols; ++j){
      std::int64_t x_start = j*chunk_shape[x_dim];
      std::int64_t x_end = std::min({(j+1)*chunk_shape[x_dim], image_width});
      th_pool.detach_task([&store1, &store2, x_start, x_end, y_start, y_end, x_dim=x_dim, y_dim=y_dim, v](){  

        auto array = tensorstore::AllocateArray({y_end-y_start, x_end-x_start},tensorstore::c_order,
                                tensorstore::value_init, store1.dtype());
        // initiate a read
        tensorstore::Read(store1 | 
                          tensorstore::Dims(3).ClosedInterval(y_start,y_end-1) |
                          tensorstore::Dims(4).ClosedInterval(x_start,x_end-1) ,
                          array).value();
        
        tensorstore::IndexTransform<> transform = tensorstore::IdentityTransform(store2.domain());
        if(v == VisType::PCNG){
          transform = (std::move(transform) | tensorstore::Dims(2, 3).IndexSlice({0,0}) 
                                            | tensorstore::Dims(y_dim).ClosedInterval(y_start,y_end-1) 
                                            | tensorstore::Dims(x_dim).ClosedInterval(x_start,x_end-1)
                                            | tensorstore::Dims(x_dim, y_dim).Transpose({y_dim, x_dim})).value();
        }else if (v == VisType::NG_Zarr || v == VisType::Viv){
          transform = (std::move(transform) | tensorstore::Dims(y_dim).ClosedInterval(y_start,y_end-1) 
                                            | tensorstore::Dims(x_dim).ClosedInterval(x_start,x_end-1)).value();
        }
        tensorstore::Write(array, store2 | transform).value();
      });       

    }
  }
  th_pool.wait();
}
} // ns argolid

