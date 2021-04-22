// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/dynload/nvjpeg.h"

namespace paddle {
namespace operators {

static nvjpegHandle_t nvjpeg_handle = nullptr;

void init_nvjpegImage(nvjpegImage_t* img) {
  for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
    img->channel[c] = nullptr;
    img->pitch[c] = 0;
  }
}

template <typename T>
class GPUDecodeJpegKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // Create nvJPEG handle
    auto* x = ctx.Input<framework::Tensor>("X");

    auto* x_data = x->data<T>();

    std::cout << x->place() << std::endl;

    if (nvjpeg_handle == nullptr) {
      nvjpegStatus_t create_status =
          platform::dynload::nvjpegCreateSimple(&nvjpeg_handle);

      if (create_status != NVJPEG_STATUS_SUCCESS) {
        std::cout << "nvjpegCreateSimple failed: " << create_status
                  << std::endl;
      }
      // (
      //     create_status == NVJPEG_STATUS_SUCCESS,
      //     "nvjpegCreateSimple failed: ",
      //     create_status);
    }

    nvjpegJpegState_t nvjpeg_state;
    nvjpegStatus_t state_status =
        platform::dynload::nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state);

    if (state_status != NVJPEG_STATUS_SUCCESS) {
      std::cout << "nvjpegJpegStateCreate failed: " << state_status
                << std::endl;
    }

    int components;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];

    nvjpegStatus_t info_status = platform::dynload::nvjpegGetImageInfo(
        nvjpeg_handle, x_data, (size_t)x->numel(), &components, &subsampling,
        widths, heights);

    if (info_status != NVJPEG_STATUS_SUCCESS) {
      std::cout << "nvjpegGetImageInfo failed: " << std::endl;
      // nvjpegJpegStateDestroy(nvjpeg_state);
      // TORCH_CHECK(false, "nvjpegGetImageInfo failed: ", info_status);
    }

    int width = widths[0];
    int height = heights[0];

    nvjpegOutputFormat_t outputFormat;
    int outputComponents;

    if (components == 1) {
      outputFormat = NVJPEG_OUTPUT_Y;
      outputComponents = 1;
    } else if (components == 3) {
      outputFormat = NVJPEG_OUTPUT_RGB;
      outputComponents = 3;
    } else {
      std::cout << "The provided mode is not supported for JPEG files on GPU"
                << std::endl;
      // nvjpegJpegStateDestroy(nvjpeg_state);
      // TORCH_CHECK(
      //     false, "The provided mode is not supported for JPEG files on GPU");
    }

    nvjpegImage_t outImage;
    init_nvjpegImage(&outImage);

    auto stream = ctx.cuda_device_context().stream();

    int sz = widths[0] * heights[0];

    auto* out = ctx.Output<framework::LoDTensor>("Out");
    std::vector<int64_t> out_shape = {outputComponents, height, width};
    out->Resize(framework::make_ddim(out_shape));

    T* data = out->mutable_data<T>(ctx.GetPlace());

    for (int c = 0; c < outputComponents; c++) {
      outImage.channel[c] = data + c * sz;
      outImage.pitch[c] = width;
    }

    nvjpegStatus_t decode_status = platform::dynload::nvjpegDecode(
        nvjpeg_handle, nvjpeg_state, x_data, x->numel(), outputFormat,
        &outImage, stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(decode_jpeg, ops::GPUDecodeJpegKernel<uint8_t>)
