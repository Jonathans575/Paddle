// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <fstream>
#include <string>
#include <vector>

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

using LoDTensorArray = framework::LoDTensorArray;

enum BufferStatus {
  kBufferStatusSuccess = 0,
  kBufferStatusErrorClosed,
  kBufferStatusEmpty
};

template <typename T>
class Buffer final {
 public:
  explicit Buffer(size_t max_len = 2) : max_len_(max_len), is_closed_(false) {}
  ~Buffer() = default;

  BufferStatus Push(const T& item);
  BufferStatus Pull(T* item);
  BufferStatus TryReceive(T* item);
  void Close();

 private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  size_t max_len_;
  bool is_closed_;
  std::condition_variable cond_;
};

template <typename T>
BufferStatus Buffer<T>::Push(const T& item) {
  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [this]() { return queue_.size() < max_len_ || is_closed_; });
  if (is_closed_) {
    return kBufferStatusErrorClosed;
  }

  queue_.push(item);
  cond_.notify_one();
  return kBufferStatusSuccess;
}

template <typename T>
BufferStatus Buffer<T>::Pull(T* item) {
  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [this]() { return (!queue_.empty()) || is_closed_; });
  if (queue_.empty()) {
    return kBufferStatusErrorClosed;
  }
  *item = queue_.front();
  queue_.pop();
  if (queue_.size() < max_len_) {
    cond_.notify_all();
  }
  return kBufferStatusSuccess;
}

template <typename T>
void Buffer<T>::Close() {
  std::unique_lock<std::mutex> lock(mutex_);
  is_closed_ = true;
  cond_.notify_all();
}

class FileDataReader {
 public:
  explicit FileDataReader(const framework::ExecutionContext& ctx) {
    std::vector<std::string> files =
        ctx.Attr<std::vector<std::string>>("files");
    std::vector<int> labels = ctx.Attr<std::vector<int>>("labels");
    rank_ = ctx.Attr<int>("rank");
    world_size_ = ctx.Attr<int>("world_size");
    // std::cout << "files and labels size: " << files.size() << " "
    //           << labels.size() << std::endl;
    batch_size_ = ctx.Attr<int>("batch_size");
    current_epoch_ = 0;
    current_iter_ = 0;
    iters_per_epoch_ = labels.size() / (batch_size_ * world_size_);
    is_closed_ = false;
    for (int i = 0, n = files.size(); i < n; i++)
      image_label_pairs_.emplace_back(std::move(files[i]), labels[i]);
    StartLoadThread();
  }

  int GetStartIndex() {
    int start_idx =
        batch_size_ * world_size_ * (current_iter_ % iters_per_epoch_) +
        rank_ * batch_size_;
    current_iter_++;
    return start_idx;
  }

  framework::LoDTensor ReadSample(const std::string filename) {
    std::ifstream input(filename.c_str(),
                        std::ios::in | std::ios::binary | std::ios::ate);
    std::streamsize file_size = input.tellg();

    input.seekg(0, std::ios::beg);

    // auto* out = ctx.Output<framework::LoDTensor>("Out");
    framework::LoDTensor out;
    std::vector<int64_t> out_shape = {file_size};
    out.Resize(framework::make_ddim(out_shape));

    uint8_t* data = out.mutable_data<uint8_t>(platform::CPUPlace());

    input.read(reinterpret_cast<char*>(data), file_size);
    return out;
  }

  void StartLoadThread() {
    if (load_thrd_.joinable()) {
      return;
    }

    load_thrd_ = std::thread([this] {
      while (!is_closed_.load() && LoadBatch()) {
      }
    });
  }

  std::pair<LoDTensorArray, std::vector<int>> Read() {
    LoDTensorArray ret;
    std::vector<int> label;
    ret.reserve(batch_size_);
    int start_index = GetStartIndex();
    for (int32_t i = start_index; i < start_index + batch_size_; ++i) {
      // FIXME
      i %= image_label_pairs_.size();
      framework::LoDTensor tmp = ReadSample(image_label_pairs_[i].first);
      ret.push_back(std::move(tmp));
      label.push_back(image_label_pairs_[i].second);
    }
    return std::make_pair(ret, label);
  }

  std::pair<LoDTensorArray, std::vector<int>> Next() {
    std::pair<LoDTensorArray, std::vector<int>> batch_data;
    batch_buffer_.Pull(&batch_data);
    return batch_data;
  }

  bool LoadBatch() {
    // std::cout << "start LoadBatch 0.01" << std::endl;
    std::pair<LoDTensorArray, std::vector<int>> batch_data = std::move(Read());
    return batch_buffer_.Push(batch_data) == BufferStatus::kBufferStatusSuccess;
  }

 private:
  int batch_size_;
  std::string file_root_, file_list_;
  std::vector<std::pair<std::string, int>> image_label_pairs_;
  int current_epoch_;
  int current_iter_;
  int rank_;
  int world_size_;
  int iters_per_epoch_;
  std::atomic<bool> is_closed_;

  Buffer<std::pair<LoDTensorArray, std::vector<int>>> batch_buffer_;
  // Buffer<LoDTensorArray> batch_buffer_;
  std::thread load_thrd_;
};

class FileDataReaderWrapper {
 public:
  void SetUp(const framework::ExecutionContext& ctx) {
    reader.reset(new FileDataReader(ctx));
  }

  std::shared_ptr<FileDataReader> reader = nullptr;
};

FileDataReaderWrapper reader_wrapper;

template <typename T>
class CPUFileLabelKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

class FileLabelReaderOp : public framework::OperatorBase {
 public:
  // using framework::OperatorWithKernel::OperatorWithKernel;
  FileLabelReaderOp(const std::string& type,
                    const framework::VariableNameMap& inputs,
                    const framework::VariableNameMap& outputs,
                    const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext* ctx) const {
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of ReadFileOp is null."));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    return framework::OpKernelType(framework::proto::VarType::UINT8,
                                   platform::CPUPlace());
  }

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    LOG(ERROR) << "FileLabelReaderOp RunImpl start";
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& dev_ctx = *pool.Get(dev_place);
    framework::RuntimeContext run_ctx(Inputs(), Outputs(), scope);
    framework::ExecutionContext ctx(*this, scope, dev_ctx, run_ctx);
    if (reader_wrapper.reader == nullptr) {
      // create reader
      reader_wrapper.SetUp(ctx);
    }
    LOG(ERROR) << "FileLabelReaderOp RunImpl start 0.0";
    std::pair<LoDTensorArray, std::vector<int>> samples =
        reader_wrapper.reader->Next();
    LOG(ERROR) << "FileLabelReaderOp RunImpl start 0.1";
    auto* out = scope.FindVar(Output("Out"));
    auto& out_array = *out->GetMutable<framework::LoDTensorArray>();
    auto* label = scope.FindVar(Output("Label"));
    LOG(ERROR) << "FileLabelReaderOp RunImpl start 0.2";
    auto& label_tensor = *label->GetMutable<framework::LoDTensor>();
    LOG(ERROR) << "FileLabelReaderOp RunImpl start 0.3, "
               << static_cast<int64_t>(samples.first.size());
    label_tensor.Resize(
        framework::make_ddim({static_cast<int64_t>(samples.first.size())}));
    LOG(ERROR) << "FileLabelReaderOp RunImpl start 0.4";
    platform::CPUPlace cpu;
    auto* label_data = label_tensor.mutable_data<int>(cpu);
    LOG(ERROR) << "FileLabelReaderOp RunImpl start 0.5";
    out_array.resize(samples.first.size());
    LOG(ERROR) << "FileLabelReaderOp RunImpl start 0.6";
    for (size_t i = 0; i < samples.first.size(); ++i) {
      copy_tensor(samples.first[i], &out_array[i]);
      LOG(ERROR) << "FileLabelReaderOp RunImpl start 0.7";
      label_data[i] = samples.second[i];
    }
    LOG(ERROR) << "FileLabelReaderOp RunImpl finish";
  }

  void copy_tensor(const framework::LoDTensor& lod_tensor,
                   framework::LoDTensor* out) const {
    if (lod_tensor.numel() == 0) return;
    auto& out_tensor = *out;
    TensorCopy(lod_tensor, lod_tensor.place(), &out_tensor);
    out_tensor.set_lod(lod_tensor.lod());
  }

  // std::shared_ptr<FileDataReader> reader=nullptr;
};

class FileLabelReaderOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "The output tensor of ReadFile op");
    AddOutput("Label", "The output tensor of ReadFile op");
    AddComment(R"DOC(
This operator read a file.
)DOC");
    AddAttr<std::string>("root_dir", "Path of the file to be readed.")
        .SetDefault("");
    AddAttr<int>("batch_size", "Path of the file to be readed.").SetDefault(1);
    AddAttr<int>("rank", "Path of the file to be readed.").SetDefault(0);
    AddAttr<int>("world_size", "Path of the file to be readed.").SetDefault(1);
    AddAttr<std::vector<std::string>>("files", "Path of the file to be readed.")
        .SetDefault({});
    AddAttr<std::vector<int>>("labels", "Path of the file to be readed.")
        .SetDefault({});
  }
};

class FileLabelReaderInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* context) const override {
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out",
                   "FileLabelReader");
  }
};

class FileLabelReaderInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    ctx->SetOutputType("Out", framework::proto::VarType::LOD_TENSOR_ARRAY,
                       framework::ALL_ELEMENTS);
    ctx->SetOutputType("Label", framework::proto::VarType::LOD_TENSOR);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    file_label_reader, ops::FileLabelReaderOp, ops::FileLabelReaderOpMaker,
    ops::FileLabelReaderInferShape, ops::FileLabelReaderInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>)

REGISTER_OP_CPU_KERNEL(file_label_reader, ops::CPUFileLabelKernel<uint8_t>)
