// SPDX-License-Identifier: MIT
#define REALM_ORT_BRIDGE_BUILD
#include "onnx_cmajor_bridge.h"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
struct Model {
    Ort::Session session{nullptr};
    std::string input_name, output_name;
    ONNXTensorElementDataType input_type{}, output_type{};
};
Ort::Env& env() { static Ort::Env value(ORT_LOGGING_LEVEL_WARNING, "realm-quant"); return value; }
std::mutex models_mutex;
std::unordered_map<int64_t, std::unique_ptr<Model>> models;
int64_t next_handle = 1;
thread_local std::string last_error;
size_t checked_count(const int64_t* dims, int32_t rank) {
    if (!dims || rank < 1 || rank > 16) throw std::runtime_error("rank must be 1..16");
    size_t count = 1;
    for (int32_t i = 0; i < rank; ++i) {
        if (dims[i] <= 0) throw std::runtime_error("runtime dimensions must be positive");
        const auto d = static_cast<size_t>(dims[i]);
        if (count > std::numeric_limits<size_t>::max() / d) throw std::runtime_error("tensor size overflow");
        count *= d;
    }
    return count;
}
bool supported(ONNXTensorElementDataType t) {
    return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
           t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
           t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
           t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}
}

extern "C" int64_t rq_ort_open(const char* path, int32_t threads) {
    try {
        if (!path || !*path) throw std::runtime_error("model path is empty");
        if (threads < 1 || threads > 256) throw std::runtime_error("threads must be in [1,256]");
        auto m = std::make_unique<Model>();
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(threads);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        m->session = Ort::Session(env(), path, options);
        if (m->session.GetInputCount() != 1 || m->session.GetOutputCount() != 1)
            throw std::runtime_error("bridge requires exactly one ONNX input and one output");
        Ort::AllocatorWithDefaultOptions allocator;
        auto in = m->session.GetInputNameAllocated(0, allocator);
        auto out = m->session.GetOutputNameAllocated(0, allocator);
        m->input_name = in.get(); m->output_name = out.get();
        m->input_type = m->session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
        m->output_type = m->session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
        if (!supported(m->input_type) || !supported(m->output_type))
            throw std::runtime_error("supported tensor types: float16, float32, int32, int64");
        std::lock_guard<std::mutex> lock(models_mutex);
        const int64_t id = next_handle++;
        models.emplace(id, std::move(m));
        last_error.clear();
        return id;
    } catch (const std::exception& e) { last_error = e.what(); return 0; }
}

extern "C" int64_t rq_ort_run_f32(int64_t handle, const float* input,
    const int64_t* input_dims, int32_t input_rank, float* output,
    int64_t output_capacity, int64_t* output_dims, int32_t max_output_rank,
    int32_t* output_rank) {
    try {
        if (!input || !output || !output_dims || !output_rank ||
            output_capacity < 0 || max_output_rank < 0 || max_output_rank > 16)
            throw std::runtime_error("null buffer or invalid output capacity/rank");
        const size_t n = checked_count(input_dims, input_rank);
        std::lock_guard<std::mutex> lock(models_mutex);
        auto it = models.find(handle);
        if (it == models.end()) throw std::runtime_error("invalid model handle");
        Model& m = *it->second;
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<float> f32(input, input + n);
        std::vector<int32_t> i32;
        std::vector<int64_t> i64;
        std::vector<Ort::Float16_t> f16;
        void* data = f32.data();
        size_t element_bytes = sizeof(float);
        if (m.input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            i32.reserve(n);
            for (float x : f32) {
                if (!std::isfinite(x) || static_cast<double>(x) < std::numeric_limits<int32_t>::min() ||
                    static_cast<double>(x) >= static_cast<double>(std::numeric_limits<int32_t>::max()) + 1.0)
                    throw std::runtime_error("input value is outside int32 range");
                i32.push_back(static_cast<int32_t>(x));
            }
            data = i32.data(); element_bytes = sizeof(int32_t);
        } else if (m.input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            i64.reserve(n);
            const double int64_limit = std::ldexp(1.0, 63);
            for (float x : f32) {
                if (!std::isfinite(x) || static_cast<double>(x) < -int64_limit ||
                    static_cast<double>(x) >= int64_limit)
                    throw std::runtime_error("input value is outside int64 range");
                i64.push_back(static_cast<int64_t>(x));
            }
            data = i64.data(); element_bytes = sizeof(int64_t);
        } else if (m.input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            f16.reserve(n); for (float x : f32) f16.emplace_back(x);
            data = f16.data(); element_bytes = sizeof(Ort::Float16_t);
        }
        std::vector<int64_t> dims(input_dims, input_dims + input_rank);
        auto tensor = Ort::Value::CreateTensor(memory, data, n * element_bytes,
            dims.data(), dims.size(), m.input_type);
        const char* ins[] = {m.input_name.c_str()};
        const char* outs[] = {m.output_name.c_str()};
        auto values = m.session.Run(Ort::RunOptions{nullptr}, ins, &tensor, 1, outs, 1);
        auto info = values.at(0).GetTensorTypeAndShapeInfo();
        const auto shape = info.GetShape();
        if (shape.size() > static_cast<size_t>(max_output_rank))
            throw std::runtime_error("output shape buffer too small");
        size_t count = 1;
        for (auto d : shape) {
            if (d < 0) throw std::runtime_error("unresolved output dimension");
            if (d == 0) { count = 0; break; }
            if (count > std::numeric_limits<size_t>::max() / static_cast<size_t>(d))
                throw std::runtime_error("output size overflow");
            count *= static_cast<size_t>(d);
        }
        if (count > static_cast<size_t>(output_capacity))
            throw std::runtime_error("output value buffer too small");
        const void* raw = values[0].GetTensorRawData();
        switch (info.GetElementType()) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                std::copy_n(static_cast<const float*>(raw), count, output); break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                auto p = static_cast<const int32_t*>(raw);
                for (size_t i = 0; i < count; ++i) { output[i] = static_cast<float>(p[i]); }
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                auto p = static_cast<const int64_t*>(raw);
                for (size_t i = 0; i < count; ++i) { output[i] = static_cast<float>(p[i]); }
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
                auto p = static_cast<const Ort::Float16_t*>(raw);
                for (size_t i = 0; i < count; ++i) { output[i] = static_cast<float>(p[i]); }
                break;
            }
            default: throw std::runtime_error("unsupported output element type");
        }
        std::copy(shape.begin(), shape.end(), output_dims);
        *output_rank = static_cast<int32_t>(shape.size());
        last_error.clear();
        return static_cast<int64_t>(count);
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
extern "C" int32_t rq_ort_last_error(char* dst, int32_t capacity) {
    if (!dst || capacity <= 0) return static_cast<int32_t>(last_error.size());
    const auto n = std::min(last_error.size(), static_cast<size_t>(capacity - 1));
    std::memcpy(dst, last_error.data(), n); dst[n] = '\0';
    return static_cast<int32_t>(last_error.size());
}
extern "C" void rq_ort_close(int64_t handle) {
    std::lock_guard<std::mutex> lock(models_mutex);
    models.erase(handle);
}

extern "C" void* rq_ort_resolve(const char* symbol) {
    if (!symbol) return nullptr;
    const char* name = std::strrchr(symbol, ':');
    name = name ? name + 1 : symbol;
    if (std::strcmp(name, "rq_ort_open") == 0)
        return reinterpret_cast<void*>(&rq_ort_open);
    if (std::strcmp(name, "rq_ort_run_f32") == 0)
        return reinterpret_cast<void*>(&rq_ort_run_f32);
    if (std::strcmp(name, "rq_ort_last_error") == 0)
        return reinterpret_cast<void*>(&rq_ort_last_error);
    if (std::strcmp(name, "rq_ort_close") == 0)
        return reinterpret_cast<void*>(&rq_ort_close);
    return nullptr;
}
