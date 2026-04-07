#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

void unicore_core_gemm_bf16(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C,
                                   int M, int N, int K, int device_id);
void unicore_core_gemm_fp16(half* A, half* B, half* C,
                                   int M, int N, int K, int device_id);

void unicore_core_gemm_fp8(uint8_t* A, uint8_t* B, __nv_bfloat16* C,
                                   int M, int N, int K, int comp_method, int device_id);

void unicore_core_gemm_fp8_grouped(uint8_t* A, uint8_t* B, __nv_bfloat16* C,
                                __nv_bfloat16* SA, __nv_bfloat16* SB,
                                int M, int N, int K, int group_size, int comp_method, int device_id);

void unicore_core_gemm_fp4(uint8_t* A, uint8_t* B, __nv_bfloat16* C,
                                // __nv_bfloat16* SA, __nv_bfloat16* SB,
                                int M, int N, int K, int device_id);
void unicore_core_gemm_fp4_comptable(uint8_t* A, uint8_t* B, __nv_bfloat16* C,
                                int M, int N, int K, int device_id);

void torch_unicore_core_group_gemm_bf16(
    const torch::Tensor &A,
    const torch::Tensor &B,
    torch::Tensor &C,
    // torch::Tensor &S,
    int M, int N, int K) {

    int device_id = A.device().index();

    TORCH_CHECK(A.device() == B.device(), "Tensors A and B must be on the same device");
    if (A.dtype() != torch::kBFloat16 || B.dtype() != torch::kBFloat16 || C.dtype() != torch::kBFloat16) {
        throw std::invalid_argument("Tensors must be of type bfloat16.");
    }

    unicore_core_gemm_bf16(
        (__nv_bfloat16*) A.data_ptr(),
        (__nv_bfloat16*) B.data_ptr(),
        (__nv_bfloat16*) C.data_ptr(),
        // (half*) S.data_ptr(),
        M, N, K,
        device_id
    );
}

void torch_unicore_core_group_gemm_fp16(
    const torch::Tensor &A,
    const torch::Tensor &B,
    torch::Tensor &C,
    int M, int N, int K) {

    int device_id = A.device().index();

    TORCH_CHECK(A.device() == B.device(), "Tensors A and B must be on the same device");
    if (A.dtype() != torch::kFloat16 || B.dtype() != torch::kFloat16 || C.dtype() != torch::kFloat16) {
        throw std::invalid_argument("Tensors must be of type float16.");
    }

    unicore_core_gemm_fp16(
        (half*) A.data_ptr(),
        (half*) B.data_ptr(),
        (half*) C.data_ptr(),
        M, N, K,
        device_id
    );
}

void torch_unicore_core_group_gemm_fp8(
    const torch::Tensor &A,
    const torch::Tensor &B,
    torch::Tensor &C,
    // torch::Tensor &S,
    int M, int N, int K, int comp_method) {

    int device_id = A.device().index();

    TORCH_CHECK(A.device() == B.device(), "Tensors A and B must be on the same device");
    if (!((A.dtype() == torch::kUInt8 || A.dtype() == torch::kInt8) &&
          (B.dtype() == torch::kUInt8 || B.dtype() == torch::kInt8) &&
          (C.dtype() == torch::kBFloat16))) {
        throw std::invalid_argument("Tensors A and B must be integer type (int8/uint8), and C must be bfloat16.");
    }

    unicore_core_gemm_fp8(
        (uint8_t*) A.data_ptr(),
        (uint8_t*) B.data_ptr(),
        (__nv_bfloat16*) C.data_ptr(),
        // (half*) S.data_ptr(),
        M, N, K,
        comp_method,
        device_id
    );
}

void torch_unicore_core_group_gemm_fp8_grouped(
    const torch::Tensor &A,
    const torch::Tensor &B,
    torch::Tensor &C,
    const torch::Tensor &SA,
    const torch::Tensor &SB,
    int M, int N, int K, int group_size, int comp_method) {

    int device_id = A.device().index();

    TORCH_CHECK(A.device() == B.device(), "Tensors A and B must be on the same device");
    TORCH_CHECK(SA.device() == A.device(), "Scale tensors must be on the same device as data tensors");
    TORCH_CHECK(SB.device() == A.device(), "Scale tensors must be on the same device as data tensors");

    if (!((A.dtype() == torch::kUInt8 || A.dtype() == torch::kInt8) &&
          (B.dtype() == torch::kUInt8 || B.dtype() == torch::kInt8) &&
          (C.dtype() == torch::kBFloat16) &&
          (SA.dtype() == torch::kBFloat16) &&
          (SB.dtype() == torch::kBFloat16))) {
        throw std::invalid_argument("Tensors A and B must be integer type (int8/uint8), C/SA/SB must be bfloat16.");
    }

    unicore_core_gemm_fp8_grouped(
        (uint8_t*) A.data_ptr(),
        (uint8_t*) B.data_ptr(),
        (__nv_bfloat16*) C.data_ptr(),
        (__nv_bfloat16*) SA.data_ptr(),
        (__nv_bfloat16*) SB.data_ptr(),
        M, N, K, group_size, comp_method,
        device_id
    );
}


void torch_unicore_core_group_gemm_fp4(
    const torch::Tensor &A,
    const torch::Tensor &B,
    torch::Tensor &C,
    // torch::Tensor &S,
    int M, int N, int K) {

    int device_id = A.device().index();

    TORCH_CHECK(A.device() == B.device(), "Tensors A and B must be on the same device");
    if (!((A.dtype() == torch::kUInt8 || A.dtype() == torch::kInt8) &&
          (B.dtype() == torch::kUInt8 || B.dtype() == torch::kInt8) &&
          (C.dtype() == torch::kBFloat16))) {
        throw std::invalid_argument("Tensors A and B must be integer type (int8/uint8), and C must be bfloat16.");
    }

    unicore_core_gemm_fp4(
        (uint8_t*) A.data_ptr(),
        (uint8_t*) B.data_ptr(),
        (__nv_bfloat16*) C.data_ptr(),
        // (half*) S.data_ptr(),
        M, N, K,
        device_id
    );
}

void torch_unicore_core_group_gemm_fp4_comptable(
    const torch::Tensor &A,
    const torch::Tensor &B,
    torch::Tensor &C,
    int M, int N, int K) {

    int device_id = A.device().index();

    TORCH_CHECK(A.device() == B.device(), "Tensors A and B must be on the same device");
    if (!((A.dtype() == torch::kUInt8 || A.dtype() == torch::kInt8) &&
          (B.dtype() == torch::kUInt8 || B.dtype() == torch::kInt8) &&
          (C.dtype() == torch::kBFloat16))) {
        throw std::invalid_argument("Tensors A and B must be integer type (int8/uint8), and C must be bfloat16.");
    }

    unicore_core_gemm_fp4_comptable(
        (uint8_t*) A.data_ptr(),
        (uint8_t*) B.data_ptr(),
        (__nv_bfloat16*) C.data_ptr(),
        M, N, K,
        device_id
    );
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unicore_GEMM_bf16", &torch_unicore_core_group_gemm_bf16, "UniCore GEMM kernel for bfloat16");
    m.def("unicore_GEMM_fp16", &torch_unicore_core_group_gemm_fp16, "UniCore GEMM kernel for float16");
    m.def(
        "unicore_GEMM_fp8",
        &torch_unicore_core_group_gemm_fp8,
        "UniCore GEMM kernel for fp8",
        py::arg("A"),
        py::arg("B"),
        py::arg("C"),
        py::arg("M"),
        py::arg("N"),
        py::arg("K"),
        py::arg("comp_method") = 1
    );
    m.def(
        "unicore_GEMM_fp8_grouped",
        &torch_unicore_core_group_gemm_fp8_grouped,
        "UniCore GEMM kernel for fp8 with group quantization",
        py::arg("A"),
        py::arg("B"),
        py::arg("C"),
        py::arg("SA"),
        py::arg("SB"),
        py::arg("M"),
        py::arg("N"),
        py::arg("K"),
        py::arg("group_size"),
        py::arg("comp_method") = 1
    );
    m.def("unicore_GEMM_fp4", &torch_unicore_core_group_gemm_fp4, "UniCore GEMM kernel for fp4");
    m.def("unicore_GEMM_fp4_comptable", &torch_unicore_core_group_gemm_fp4_comptable, "UniCore GEMM kernel for fp4 with comp table");
}
