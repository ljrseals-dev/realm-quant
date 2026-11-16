// SPDX-License-Identifier: MIT
#pragma once
#include <stdint.h>
#ifdef _WIN32
  #ifdef REALM_ORT_BRIDGE_BUILD
    #define RQ_BRIDGE_API __declspec(dllexport)
  #else
    #define RQ_BRIDGE_API __declspec(dllimport)
  #endif
#else
  #define RQ_BRIDGE_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
RQ_BRIDGE_API int64_t rq_ort_open(const char* model_path, int32_t intra_op_threads);
RQ_BRIDGE_API int64_t rq_ort_run_f32(int64_t handle, const float* input,
    const int64_t* input_dims, int32_t input_rank, float* output,
    int64_t output_capacity, int64_t* output_dims, int32_t max_output_rank,
    int32_t* output_rank);
RQ_BRIDGE_API int32_t rq_ort_last_error(char* dst, int32_t capacity);
// Host-side Engine::link resolver. Accepts a bare or namespace-qualified name.
RQ_BRIDGE_API void* rq_ort_resolve(const char* symbol);
RQ_BRIDGE_API void rq_ort_close(int64_t handle);
#ifdef __cplusplus
}
#endif
