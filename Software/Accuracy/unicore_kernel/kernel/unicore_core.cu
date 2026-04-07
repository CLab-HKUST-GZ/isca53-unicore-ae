#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cmath> // for isfinite

#define COMP_TABLE_SIZE 8
constexpr int FP8_COMPENSATION_FPMG_CG = 0;
constexpr int FP8_COMPENSATION_UNICORE = 1;

__constant__ uint16_t comp_table_const_bf16_8x8[8][8] = {
    {0,  1,  2,  3,  4,  5,  6, 3},
    {1,  4,  7, 10, 13, 14, 10, 3},
    {2,  7, 12, 17, 19, 14,  8, 3},
    {3, 10, 17, 20, 16, 11,  7, 2},
    {4, 13, 19, 16, 12,  9,  5, 2},
    {5, 14, 14, 11,  9,  6,  4, 1},
    {6, 10,  8,  7,  5,  4,  2, 1},
    {3,  3,  3,  2,  2,  1,  1, 0},
};

__constant__ uint16_t comp_table_const_bf16_4x4[4][4] = {
    {2,  6,  9,  6},
    {6, 17, 15,  5},
    {9, 15,  9,  3},
    {6,  5,  3,  1},
};

__constant__ uint16_t comp_table_const_fp16_8x8[8][8] = {
    { 4,  12,  20,  28,  36,  44, 48, 24},
    {12,  36,  60,  84, 108, 115, 78, 26},
    {20,  60, 100, 139, 148, 110, 66, 22},
    {28,  84, 139, 158, 126,  90, 54, 18},
    {36, 108, 148, 126,  98,  70, 42, 14},
    {44, 115, 110,  90,  70,  50, 30, 10},
    {48,  78,  66,  54,  42,  30, 18,  6},
    {24,  26,  22,  18,  14,  10,  6,  2},
};


// __constant__ uint16_t comp_table_const_fp8_8x8[8][8] = {
//     {0, 0, 0, 0, 0, 0, 0, 0},
//     {0, 0, 0, 0, 0, 0, 0, 0},
//     {0, 0, 0, 0, 0, 0, 0, 0},
//     {0, 0, 0, 0, 0, 0, 0, 0},
//     {0, 0, 0, 0, 0, 0, 0, 0},
//     {0, 0, 0, 0, 0, 0, 0, 0},
//     {0, 0, 0, 0, 0, 0, 0, 0},
//     {0, 0, 0, 0, 0, 0, 0, 0},
// };

__constant__ uint16_t comp_table_const_fp8_8x8[8][8] = {
    {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 1, 1, 0, 0},
    {0, 0, 0, 1, 1, 1, 1, 0},
    {0, 0, 1, 1, 1, 1, 1, 0},
    {0, 1, 1, 1, 1, 1, 0, 0},
    {0, 1, 1, 1, 1, 1, 0, 0},
    {0, 0, 1, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0}
};

__constant__ uint16_t comp_table_const_fp8_8x8_us[8][8] = {
    {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 1, 1, 1, 1},
    {0, 0, 1, 2, 2, 2, 1, 1},
    {0, 1, 2, 2, 3, 2, 1, 1},
    {0, 1, 2, 3, 2, 2, 1, 0},
    {0, 1, 2, 2, 2, 1, 1, 0},
    {0, 1, 1, 1, 1, 1, 0, 0},
    {0, 1, 1, 1, 0, 0, 0, 0}
};

__constant__ uint16_t comp_table_const_fp4_8x8[8][8] = {
    {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 1, 0, 1},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 1, 0, 1},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 1, 0, 1}
};

// __constant__ uint16_t comp_table_const_fp8_8x8_us[8][8] = {
//     {0, 0, 0, 0, 0, 0, 0, 0},
//     {0, 0, 0, 0, 1, 1, 1, 0},
//     {0, 0, 1, 1, 2, 2, 1, 0},
//     {0, 0, 1, 2, 3, 1, 1, 0},
//     {0, 1, 2, 3, 2, 1, 1, 0},
//     {0, 1, 2, 1, 1, 1, 0, 0},
//     {0, 1, 1, 1, 1, 0, 0, 0},
//     {0, 0, 0, 0, 0, 0, 0, 0}
// };

// __constant__ uint16_t comp_table_const_fp8_8x8_us[8][8] = {
//     {0, 0, 0, 0, 0, 0, 0, 0},
//     {0, 0, 0, 0, 0, 0, 0, 0},
//     {0, 0, 0, 0, 0, 0, 0, 0},
//     {0, 0, 0, 0, 0, 0, 0, 0},
//     {0, 0, 0, 0, 0, 0, 0, 0},
//     {0, 0, 0, 0, 0, 0, 0, 0},
//     {0, 0, 0, 0, 0, 0, 0, 0},
//     {0, 0, 0, 0, 0, 0, 0, 0},
// };


typedef union {
    __nv_bfloat16 bf16;
    uint16_t u16;
} bf16_union;

typedef union {
    half fp16;
    uint16_t u16;
} fp16_union;

__device__ uint16_t bfloat16_to_uint16(__nv_bfloat16 a) {
    bf16_union u;
    u.bf16 = a;
    return u.u16;
}

__device__ __nv_bfloat16 uint16_to_bfloat16(uint16_t a) {
    bf16_union u;
    u.u16 = a;
    return u.bf16;
}

__device__ uint16_t fp16_to_uint16(half a) {
    fp16_union u;
    u.fp16 = a;
    return u.u16;
}

__device__ half uint16_to_fp16(uint16_t a) {
    fp16_union u;
    u.u16 = a;
    return u.fp16;
}

__device__ __nv_bfloat16 FPMA_CC_bf16(__nv_bfloat16 a, __nv_bfloat16 b) {
    uint16_t a_bits = bfloat16_to_uint16(a);
    uint16_t b_bits = bfloat16_to_uint16(b);

    uint16_t is_zero_a = (a_bits & 0x7FFF) == 0;
    uint16_t is_zero_b = (b_bits & 0x7FFF) == 0;
    uint16_t is_zero = is_zero_a | is_zero_b;

    uint16_t s1 = a_bits & 0x8000;
    uint16_t s2 = b_bits & 0x8000;
    uint16_t sign = s1 ^ s2;

    uint16_t mantissa_sum = (a_bits & 0x7FFF) + (b_bits & 0x7FFF);

    // BF16 Baseline: 0x3F80
    // uint16_t threshold = 0x3F80;
    uint16_t threshold = 0x3F78;
    // uint16_t threshold = 0x3F80 - 8;
    uint16_t adjusted_mantissa;
    uint16_t diff = mantissa_sum - threshold;
    uint16_t is_negative_mask = -(mantissa_sum < threshold);
    adjusted_mantissa = diff & (~is_negative_mask);
    // Combine the sign and adjusted mantissa
    uint16_t result_bits = (adjusted_mantissa & 0x7FFF) | sign;
    // uint16_t exponent_fp16 = (result_bits >> 10) & 0x001F;

    // if (exponent_fp16 == 0x001F) {
    //     result_bits = sign | (0x001E << 10) | 0x03FF;
    // }

    result_bits = is_zero ? 0 : result_bits;

    return uint16_to_bfloat16(result_bits);
}

__device__ __nv_bfloat16 FPMA_TC_bf16(__nv_bfloat16 a, __nv_bfloat16 b, uint16_t comp_table[COMP_TABLE_SIZE][COMP_TABLE_SIZE]) {
    uint16_t a_bits = bfloat16_to_uint16(a);
    uint16_t b_bits = bfloat16_to_uint16(b);

    uint16_t is_zero_a = (a_bits & 0x7FFF) == 0;
    uint16_t is_zero_b = (b_bits & 0x7FFF) == 0;
    uint16_t is_zero = is_zero_a | is_zero_b;

    uint8_t comp_table_index_a = (a_bits >> 4) & 0x07;
    uint8_t comp_table_index_b = (b_bits >> 4) & 0x07;
    // uint8_t comp_table_index_a = (a_bits >> 5) & 0x03;
    // uint8_t comp_table_index_b = (b_bits >> 5) & 0x03;
    uint16_t comp = comp_table[comp_table_index_a][comp_table_index_b];

    uint16_t s1 = a_bits & 0x8000;
    uint16_t s2 = b_bits & 0x8000;
    uint16_t sign = s1 ^ s2;

    uint16_t mantissa_sum = (a_bits & 0x7FFF) + (b_bits & 0x7FFF);

    // BF16 Baseline: 0x3F80
    uint16_t threshold = 0x3F80 - comp;
    uint16_t adjusted_mantissa;
    uint16_t diff = mantissa_sum - threshold;
    uint16_t is_negative_mask = -(mantissa_sum < threshold);
    adjusted_mantissa = diff & (~is_negative_mask);
    // Combine the sign and adjusted mantissa
    uint16_t result_bits = (adjusted_mantissa & 0x7FFF) | sign;
    // uint16_t exponent_fp16 = (result_bits >> 10) & 0x001F;

    // if (exponent_fp16 == 0x001F) {
    //     result_bits = sign | (0x001E << 10) | 0x03FF;
    // }

    result_bits = is_zero ? 0 : result_bits;

    return uint16_to_bfloat16(result_bits);
}

__device__ half FPMA_CC_fp16(half a, half b) {
    uint16_t a_bits = fp16_to_uint16(a);
    uint16_t b_bits = fp16_to_uint16(b);

    uint16_t is_zero_a = (a_bits & 0x7FFF) == 0;
    uint16_t is_zero_b = (b_bits & 0x7FFF) == 0;
    uint16_t is_zero = is_zero_a | is_zero_b;

    uint16_t s1 = a_bits & 0x8000;
    uint16_t s2 = b_bits & 0x8000;
    uint16_t sign = s1 ^ s2;

    uint16_t mantissa_sum = (a_bits & 0x7FFF) + (b_bits & 0x7FFF);

    uint16_t threshold = 0x3C00 - 58;
    uint16_t adjusted_mantissa;
    uint16_t diff = mantissa_sum - threshold;
    uint16_t is_negative_mask = -(mantissa_sum < threshold);
    adjusted_mantissa = diff & (~is_negative_mask);
    // Combine the sign and adjusted mantissa
    uint16_t result_bits = (adjusted_mantissa & 0x7FFF) | sign;
    // uint16_t exponent_fp16 = (result_bits >> 10) & 0x001F;

    // if (exponent_fp16 == 0x001F) {
    //     result_bits = sign | (0x001E << 10) | 0x03FF;
    // }

    result_bits = is_zero ? 0 : result_bits;

    return uint16_to_fp16(result_bits);
}

__device__ half FPMA_TC_fp16(half a, half b, uint16_t comp_table[COMP_TABLE_SIZE][COMP_TABLE_SIZE]) {
    uint16_t a_bits = fp16_to_uint16(a);
    uint16_t b_bits = fp16_to_uint16(b);

    uint16_t is_zero_a = (a_bits & 0x7FFF) == 0;
    uint16_t is_zero_b = (b_bits & 0x7FFF) == 0;
    uint16_t is_zero = is_zero_a | is_zero_b;

    uint8_t comp_table_index_a = (a_bits >> 7) & 0x07;
    uint8_t comp_table_index_b = (b_bits >> 7) & 0x07;
    // uint8_t comp_table_index_a = (a_bits >> 5) & 0x03;
    // uint8_t comp_table_index_b = (b_bits >> 5) & 0x03;
    uint16_t comp = comp_table[comp_table_index_a][comp_table_index_b];

    uint16_t s1 = a_bits & 0x8000;
    uint16_t s2 = b_bits & 0x8000;
    uint16_t sign = s1 ^ s2;

    uint16_t mantissa_sum = (a_bits & 0x7FFF) + (b_bits & 0x7FFF);

    // BF16 Baseline: 0x3F80
    uint16_t threshold = 0x3C00 - comp;
    uint16_t adjusted_mantissa;
    uint16_t diff = mantissa_sum - threshold;
    uint16_t is_negative_mask = -(mantissa_sum < threshold);
    adjusted_mantissa = diff & (~is_negative_mask);
    // Combine the sign and adjusted mantissa
    uint16_t result_bits = (adjusted_mantissa & 0x7FFF) | sign;
    // uint16_t exponent_fp16 = (result_bits >> 10) & 0x001F;

    // if (exponent_fp16 == 0x001F) {
    //     result_bits = sign | (0x001E << 10) | 0x03FF;
    // }

    result_bits = is_zero ? 0 : result_bits;

    return uint16_to_fp16(result_bits);
}


__device__ __nv_bfloat16 FPMA_TC_fp8(uint8_t a, uint8_t b, uint16_t comp_table[COMP_TABLE_SIZE][COMP_TABLE_SIZE]) {
    uint16_t a_bits = a;  // a_bits: 0000_0000_XXXX_XXXX
    uint16_t b_bits = b;  // b_bits: 0000_0000_XXXX_XXXX

    uint16_t is_zero_a = (a_bits & 0x007F) == 0;
    uint16_t is_zero_b = (b_bits & 0x007F) == 0;
    uint16_t is_zero = is_zero_a | is_zero_b;

    uint8_t comp_table_index_a = a_bits & 0x07; // index_a: 0000_0000_XXXX_X[XXX]
    uint8_t comp_table_index_b = b_bits & 0x07; // index_b: 0000_0000_XXXX_X[XXX]
    uint16_t comp = comp_table[comp_table_index_a][comp_table_index_b];

    uint16_t s1 = a_bits & 0x0080;
    uint16_t s2 = b_bits & 0x0080;
    uint16_t sign = (s1 ^ s2) << 8;

    uint16_t mantissa_sum = (a_bits & 0x007F) + (b_bits & 0x007F);
    mantissa_sum = mantissa_sum;
    mantissa_sum = mantissa_sum + comp;
    // X.XXX[XXXXX.XXX]XXXX
    mantissa_sum = mantissa_sum << 4; // align the E4M3 zero point to E8M7, shift 7 - 3 bits
    // BF16 Baseline: 0x3F80
    uint16_t bias = 0x3F80 - 0x0700;  // output bias is bf16, input bias is two fp8 E4M3
    uint16_t adjusted_mantissa = mantissa_sum + bias;

    // Combine the sign and adjusted mantissa
    uint16_t result_bits = (adjusted_mantissa & 0x7FFF) | sign;
    // uint16_t exponent_fp16 = (result_bits >> 10) & 0x001F;

    // if (exponent_fp16 == 0x001F) {
    //     result_bits = sign | (0x001E << 10) | 0x03FF;
    // }

    result_bits = is_zero ? 0 : result_bits;

    return uint16_to_bfloat16(result_bits);
}

__device__ __nv_bfloat16 FPMA_TC_fp8_us(uint8_t a, uint8_t b, uint16_t comp_table[COMP_TABLE_SIZE][COMP_TABLE_SIZE]) {
    uint16_t a_bits = a;  // a_bits: 0000_0000_XXXX_XXXX
    uint16_t b_bits = b;  // b_bits: 0000_0000_XXXX_XXXX

    uint16_t is_zero_a = (a_bits & 0x007F) == 0;
    uint16_t is_zero_b = (b_bits & 0x007F) == 0;
    uint16_t is_zero = is_zero_a | is_zero_b;

    uint8_t comp_table_index_a = a_bits & 0x07; // index_a: 0000_0000_XXXX_X[XXX]
    uint8_t comp_table_index_b = b_bits & 0x07; // index_b: 0000_0000_XXXX_X[XXX]
    uint16_t comp = comp_table[comp_table_index_a][comp_table_index_b];

    uint16_t s1 = a_bits & 0x0080;
    uint16_t s2 = b_bits & 0x0080;
    uint16_t sign = (s1 ^ s2) << 8;

    uint16_t mantissa_sum = (a_bits & 0x007F) + (b_bits & 0x007F);
    mantissa_sum = mantissa_sum << 1;
    mantissa_sum = mantissa_sum + comp;
    // X.XXX[XXXXX.XXX]XXXX
    mantissa_sum = mantissa_sum << 3; // align the E4M3 zero point to E8M7, shift 7 - 3 bits
    // BF16 Baseline: 0x3F80
    uint16_t bias = 0x3F80 - 0x0700;  // output bias is bf16, input bias is two fp8 E4M3
    uint16_t adjusted_mantissa = mantissa_sum + bias;

    // Combine the sign and adjusted mantissa
    uint16_t result_bits = (adjusted_mantissa & 0x7FFF) | sign;
    // uint16_t exponent_fp16 = (result_bits >> 10) & 0x001F;

    // if (exponent_fp16 == 0x001F) {
    //     result_bits = sign | (0x001E << 10) | 0x03FF;
    // }

    result_bits = is_zero ? 0 : result_bits;

    return uint16_to_bfloat16(result_bits);
}


__device__ __nv_bfloat16 FPMA_TC_fp4(uint8_t a, uint8_t b) {
    uint16_t a_bits = a;  // a_bits: 0000_0000_XXXX_X.XXX
    uint16_t b_bits = b;  // b_bits: 0000_0000_XXXX_X.XXX

    uint16_t is_zero_a = (a_bits & 0x007F) == 0;
    uint16_t is_zero_b = (b_bits & 0x007F) == 0;
    uint16_t is_zero = is_zero_a | is_zero_b;

    uint16_t s1 = a_bits & 0x0080;
    uint16_t s2 = b_bits & 0x0080;
    uint16_t sign = (s1 ^ s2) << 8;

    uint16_t a_mag = a_bits & 0x007F;
    uint16_t b_mag = b_bits & 0x007F;
    uint16_t is_subnormal_a = ((a_mag & 0x0078) == 0) & ((a_mag & 0x0007) != 0);
    uint16_t is_subnormal_b = ((b_mag & 0x0078) == 0) & ((b_mag & 0x0007) != 0);

    uint16_t mantissa_sum = a_mag + b_mag;
    mantissa_sum = mantissa_sum;
    // X.XXX[XXXXX.XXX]XXXX
    mantissa_sum = mantissa_sum << 4; // align the E4M3 zero point to E8M7, shift 7 - 3 bits
    // BF16 Baseline: 0x3F80
    uint16_t threshold = 0x0100 + ((is_subnormal_a + is_subnormal_b) << 6);
    uint16_t bias = 0x3F80 - threshold;  // output bias is bf16, input bias is two fp8 E4M3
    uint16_t adjusted_mantissa = mantissa_sum + bias;

    // Combine the sign and adjusted mantissa
    uint16_t result_bits = (adjusted_mantissa & 0x7FFF) | sign;
    // uint16_t exponent_fp16 = (result_bits >> 10) & 0x001F;

    // if (exponent_fp16 == 0x001F) {
    //     result_bits = sign | (0x001E << 10) | 0x03FF;
    // }

    result_bits = is_zero ? 0 : result_bits;

    return uint16_to_bfloat16(result_bits);
}

__device__ __nv_bfloat16 FPMA_TC_fp4_comptable(uint8_t a, uint8_t b, uint16_t comp_table[COMP_TABLE_SIZE][COMP_TABLE_SIZE]) {
    uint16_t a_bits = a;
    uint16_t b_bits = b;

    uint16_t is_zero_a = (a_bits & 0x007F) == 0;
    uint16_t is_zero_b = (b_bits & 0x007F) == 0;
    uint16_t is_zero = is_zero_a | is_zero_b;

    uint16_t s1 = a_bits & 0x0080;
    uint16_t s2 = b_bits & 0x0080;
    uint16_t sign = (s1 ^ s2) << 8;

    uint16_t a_mag = a_bits & 0x007F;
    uint16_t b_mag = b_bits & 0x007F;
    uint16_t is_subnormal_a = ((a_mag & 0x0078) == 0) & ((a_mag & 0x0007) != 0);
    uint16_t is_subnormal_b = ((b_mag & 0x0078) == 0) & ((b_mag & 0x0007) != 0);
    uint8_t comp_table_index_a = (a_mag >> 2) & 0x07;
    uint8_t comp_table_index_b = (b_mag >> 2) & 0x07;
    uint16_t comp = comp_table[comp_table_index_a][comp_table_index_b];

    uint16_t mantissa_sum = a_mag + b_mag + comp;
    mantissa_sum = mantissa_sum;
    mantissa_sum = mantissa_sum << 4;
    uint16_t threshold = 0x0100 + ((is_subnormal_a + is_subnormal_b) << 6);
    uint16_t bias = 0x3F80 - threshold;
    uint16_t adjusted_mantissa = mantissa_sum + bias;

    uint16_t result_bits = (adjusted_mantissa & 0x7FFF) | sign;
    result_bits = is_zero ? 0 : result_bits;

    return uint16_to_bfloat16(result_bits);
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void Flex_FPMA_GEMM(__nv_bfloat16 *a, __nv_bfloat16 *b, __nv_bfloat16 *c, int M, int N, int K) {
    // A [M, K], B [K, N], C [M, N]
    __shared__ __nv_bfloat16 Sa[BM * (BK + 1)];
    __shared__ __nv_bfloat16 Sb[BK * BN];

    int indA = TM * blockIdx.x * blockDim.x;
    int indB = TN * blockIdx.y * blockDim.y;
    int width = (K + BK - 1) / BK;
    float tmp_sum[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int tpb = blockDim.x * blockDim.y; // threads per block

    for (int ph = 0; ph < width; ph++) {
        // Cooperative load of A tile [BM x BK] into shared memory (row-major with (BK+1) stride)
        for (int i = tid; i < BM * BK; i += tpb) {
            int m = i % BM;           // 0..BM-1
            int k = i / BM;           // 0..BK-1
            int g_m = indA + m;
            int g_k = k + ph * BK;
            if (g_m < M && g_k < K) {
                Sa[m * (BK + 1) + k] = a[g_m * K + g_k];
            } else {
                Sa[m * (BK + 1) + k] = __float2bfloat16(0.0f);
            }
        }
        // Cooperative load of B tile [BK x BN] into shared memory (row-major with BN stride)
        for (int i = tid; i < BK * BN; i += tpb) {
            int k = i / BN;           // 0..BK-1
            int n = i % BN;           // 0..BN-1
            int g_k = k + ph * BK;
            int g_n = n + indB;
            if (g_k < K && g_n < N) {
                Sb[k * BN + n] = b[g_k * N + g_n];
            } else {
                Sb[k * BN + n] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int index_k = 0; index_k < BK; index_k++) {
            for (int index_m = 0; index_m < TM; index_m++) {
                for (int index_n = 0; index_n < TN; index_n++) {
                    int reg_c_m = threadIdx.x * TM + index_m;
                    int reg_c_n = threadIdx.y * TN + index_n;
                    // tmp_sum[index_m * TN + index_n] += __bfloat162float(__hmul(Sa[reg_c_m * (BK + 1) + index_k], Sb[index_k * BN + reg_c_n]));
                    tmp_sum[index_m * TN + index_n] += __bfloat162float(FPMA_CC_bf16(Sa[reg_c_m * (BK + 1) + index_k], Sb[index_k * BN + reg_c_n]));
                }
            }
        }
        __syncthreads();
    }
    for (int index_m = 0; index_m < TM; index_m++) {
        for (int index_n = 0; index_n < TN; index_n++) {
            int reg_c_m = threadIdx.x * TM + index_m;
            int reg_c_n = threadIdx.y * TN + index_n;
            int global_m = indA + reg_c_m;
            int global_n = indB + reg_c_n;
            if (global_m < M && global_n < N) {
                c[global_m * N + global_n] = __float2bfloat16(tmp_sum[index_m * TN + index_n]);
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void Flex_FPMA_GEMM_fp16(half *a, half *b, half *c, int M, int N, int K) {
    // A [M, K], B [K, N], C [M, N]
    __shared__ half Sa[BM * (BK + 1)];
    __shared__ half Sb[BK * BN];

    int indA = TM * blockIdx.x * blockDim.x;
    int indB = TN * blockIdx.y * blockDim.y;
    int width = (K + BK - 1) / BK;
    float tmp_sum[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int tpb = blockDim.x * blockDim.y; // threads per block

    for (int ph = 0; ph < width; ph++) {
        // Cooperative load of A tile [BM x BK] into shared memory (row-major with (BK+1) stride)
        for (int i = tid; i < BM * BK; i += tpb) {
            int m = i % BM;           // 0..BM-1
            int k = i / BM;           // 0..BK-1
            int g_m = indA + m;
            int g_k = k + ph * BK;
            if (g_m < M && g_k < K) {
                Sa[m * (BK + 1) + k] = a[g_m * K + g_k];
            } else {
                Sa[m * (BK + 1) + k] = __float2half(0.0f);
            }
        }
        // Cooperative load of B tile [BK x BN] into shared memory (row-major with BN stride)
        for (int i = tid; i < BK * BN; i += tpb) {
            int k = i / BN;           // 0..BK-1
            int n = i % BN;           // 0..BN-1
            int g_k = k + ph * BK;
            int g_n = n + indB;
            if (g_k < K && g_n < N) {
                Sb[k * BN + n] = b[g_k * N + g_n];
            } else {
                Sb[k * BN + n] = __float2half(0.0f);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int index_k = 0; index_k < BK; index_k++) {
            for (int index_m = 0; index_m < TM; index_m++) {
                for (int index_n = 0; index_n < TN; index_n++) {
                    int reg_c_m = threadIdx.x * TM + index_m;
                    int reg_c_n = threadIdx.y * TN + index_n;
                    // tmp_sum[index_m * TN + index_n] += __bfloat162float(__hmul(Sa[reg_c_m * (BK + 1) + index_k], Sb[index_k * BN + reg_c_n]));
                    tmp_sum[index_m * TN + index_n] += __half2float(FPMA_CC_fp16(Sa[reg_c_m * (BK + 1) + index_k], Sb[index_k * BN + reg_c_n]));
                }
            }
        }
        __syncthreads();
    }
    for (int index_m = 0; index_m < TM; index_m++) {
        for (int index_n = 0; index_n < TN; index_n++) {
            int reg_c_m = threadIdx.x * TM + index_m;
            int reg_c_n = threadIdx.y * TN + index_n;
            int global_m = indA + reg_c_m;
            int global_n = indB + reg_c_n;
            if (global_m < M && global_n < N) {
                c[global_m * N + global_n] = __float2half(tmp_sum[index_m * TN + index_n]);
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void Flex_FPMA_GEMM_comptable(__nv_bfloat16 *a, __nv_bfloat16 *b, __nv_bfloat16 *c, int M, int N, int K) {

    __shared__ uint16_t comp_table_shared[COMP_TABLE_SIZE][COMP_TABLE_SIZE];
    if (threadIdx.x < COMP_TABLE_SIZE && threadIdx.y < COMP_TABLE_SIZE) {
        comp_table_shared[threadIdx.x][threadIdx.y] = comp_table_const_bf16_8x8[threadIdx.x][threadIdx.y];
    }
    __syncthreads();
    // A [M, K], B [K, N], C [M, N]
    __shared__ __nv_bfloat16 Sa[BM * (BK + 1)];
    __shared__ __nv_bfloat16 Sb[BK * BN];

    int indA = TM * blockIdx.x * blockDim.x;
    int indB = TN * blockIdx.y * blockDim.y;
    int width = (K + BK - 1) / BK;
    float tmp_sum[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int tpb = blockDim.x * blockDim.y; // threads per block

    for (int ph = 0; ph < width; ph++) {
        // Cooperative load of A tile [BM x BK] into shared memory (row-major with (BK+1) stride)
        for (int i = tid; i < BM * BK; i += tpb) {
            int m = i % BM;           // 0..BM-1
            int k = i / BM;           // 0..BK-1
            int g_m = indA + m;
            int g_k = k + ph * BK;
            if (g_m < M && g_k < K) {
                Sa[m * (BK + 1) + k] = a[g_m * K + g_k];
            } else {
                Sa[m * (BK + 1) + k] = __float2bfloat16(0.0f);
            }
        }
        // Cooperative load of B tile [BK x BN] into shared memory (row-major with BN stride)
        for (int i = tid; i < BK * BN; i += tpb) {
            int k = i / BN;           // 0..BK-1
            int n = i % BN;           // 0..BN-1
            int g_k = k + ph * BK;
            int g_n = n + indB;
            if (g_k < K && g_n < N) {
                Sb[k * BN + n] = b[g_k * N + g_n];
            } else {
                Sb[k * BN + n] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int index_k = 0; index_k < BK; index_k++) {
            for (int index_m = 0; index_m < TM; index_m++) {
                for (int index_n = 0; index_n < TN; index_n++) {
                    int reg_c_m = threadIdx.x * TM + index_m;
                    int reg_c_n = threadIdx.y * TN + index_n;
                    // tmp_sum[index_m * TN + index_n] += __bfloat162float(__hmul(Sa[reg_c_m * (BK + 1) + index_k], Sb[index_k * BN + reg_c_n]));
                    tmp_sum[index_m * TN + index_n] += __bfloat162float(FPMA_TC_bf16(Sa[reg_c_m * (BK + 1) + index_k], Sb[index_k * BN + reg_c_n], comp_table_shared));
                }
            }
        }
        __syncthreads();
    }
    for (int index_m = 0; index_m < TM; index_m++) {
        for (int index_n = 0; index_n < TN; index_n++) {
            int reg_c_m = threadIdx.x * TM + index_m;
            int reg_c_n = threadIdx.y * TN + index_n;
            int global_m = indA + reg_c_m;
            int global_n = indB + reg_c_n;
            if (global_m < M && global_n < N) {
                c[global_m * N + global_n] = __float2bfloat16(tmp_sum[index_m * TN + index_n]);
            }
        }
    }

}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void Flex_FPMA_GEMM_comptable_fp16(half *a, half *b, half *c, int M, int N, int K) {

    __shared__ uint16_t comp_table_shared[COMP_TABLE_SIZE][COMP_TABLE_SIZE];
    if (threadIdx.x < COMP_TABLE_SIZE && threadIdx.y < COMP_TABLE_SIZE) {
        comp_table_shared[threadIdx.x][threadIdx.y] = comp_table_const_fp16_8x8[threadIdx.x][threadIdx.y];
    }
    __syncthreads();
    // A [M, K], B [K, N], C [M, N]
    __shared__ half Sa[BM * (BK + 1)];
    __shared__ half Sb[BK * BN];

    int indA = TM * blockIdx.x * blockDim.x;
    int indB = TN * blockIdx.y * blockDim.y;
    int width = (K + BK - 1) / BK;
    float tmp_sum[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int tpb = blockDim.x * blockDim.y; // threads per block

    for (int ph = 0; ph < width; ph++) {
        // Cooperative load of A tile [BM x BK] into shared memory (row-major with (BK+1) stride)
        for (int i = tid; i < BM * BK; i += tpb) {
            int m = i % BM;           // 0..BM-1
            int k = i / BM;           // 0..BK-1
            int g_m = indA + m;
            int g_k = k + ph * BK;
            if (g_m < M && g_k < K) {
                Sa[m * (BK + 1) + k] = a[g_m * K + g_k];
            } else {
                Sa[m * (BK + 1) + k] = __float2half(0.0f);
            }
        }
        // Cooperative load of B tile [BK x BN] into shared memory (row-major with BN stride)
        for (int i = tid; i < BK * BN; i += tpb) {
            int k = i / BN;           // 0..BK-1
            int n = i % BN;           // 0..BN-1
            int g_k = k + ph * BK;
            int g_n = n + indB;
            if (g_k < K && g_n < N) {
                Sb[k * BN + n] = b[g_k * N + g_n];
            } else {
                Sb[k * BN + n] = __float2half(0.0f);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int index_k = 0; index_k < BK; index_k++) {
            for (int index_m = 0; index_m < TM; index_m++) {
                for (int index_n = 0; index_n < TN; index_n++) {
                    int reg_c_m = threadIdx.x * TM + index_m;
                    int reg_c_n = threadIdx.y * TN + index_n;
                    // tmp_sum[index_m * TN + index_n] += __bfloat162float(__hmul(Sa[reg_c_m * (BK + 1) + index_k], Sb[index_k * BN + reg_c_n]));
                    tmp_sum[index_m * TN + index_n] += __half2float(FPMA_TC_fp16(Sa[reg_c_m * (BK + 1) + index_k], Sb[index_k * BN + reg_c_n], comp_table_shared));
                }
            }
        }
        __syncthreads();
    }
    for (int index_m = 0; index_m < TM; index_m++) {
        for (int index_n = 0; index_n < TN; index_n++) {
            int reg_c_m = threadIdx.x * TM + index_m;
            int reg_c_n = threadIdx.y * TN + index_n;
            int global_m = indA + reg_c_m;
            int global_n = indB + reg_c_n;
            if (global_m < M && global_n < N) {
                c[global_m * N + global_n] = __float2half(tmp_sum[index_m * TN + index_n]);
            }
        }
    }

}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void Flex_FPMA_GEMM_comptable_fp8(uint8_t *a, uint8_t *b, __nv_bfloat16 *c, int M, int N, int K) {

    __shared__ uint16_t comp_table_shared[COMP_TABLE_SIZE][COMP_TABLE_SIZE];
    if (threadIdx.x < COMP_TABLE_SIZE && threadIdx.y < COMP_TABLE_SIZE) {
        comp_table_shared[threadIdx.x][threadIdx.y] = comp_table_const_fp8_8x8[threadIdx.x][threadIdx.y];
    }
    __syncthreads();
    // A [M, K], B [K, N], C [M, N]
    __shared__ uint8_t Sa[BM * (BK + 1)];
    __shared__ uint8_t Sb[BK * BN];

    int indA = TM * blockIdx.x * blockDim.x;
    int indB = TN * blockIdx.y * blockDim.y;
    int width = (K + BK - 1) / BK;
    float tmp_sum[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int tpb = blockDim.x * blockDim.y; // threads per block

    for (int ph = 0; ph < width; ph++) {
        // Cooperative load of A tile [BM x BK] into shared memory (row-major with (BK+1) stride)
        for (int i = tid; i < BM * BK; i += tpb) {
            int m = i % BM;           // 0..BM-1
            int k = i / BM;           // 0..BK-1
            int g_m = indA + m;
            int g_k = k + ph * BK;
            if (g_m < M && g_k < K) {
                Sa[m * (BK + 1) + k] = a[g_m * K + g_k];
            } else {
                Sa[m * (BK + 1) + k] = 0;
            }
        }
        // Cooperative load of B tile [BK x BN] into shared memory (row-major with BN stride)
        for (int i = tid; i < BK * BN; i += tpb) {
            int k = i / BN;           // 0..BK-1
            int n = i % BN;           // 0..BN-1
            int g_k = k + ph * BK;
            int g_n = n + indB;
            if (g_k < K && g_n < N) {
                Sb[k * BN + n] = b[g_k * N + g_n];
            } else {
                Sb[k * BN + n] = 0;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int index_k = 0; index_k < BK; index_k++) {
            for (int index_m = 0; index_m < TM; index_m++) {
                for (int index_n = 0; index_n < TN; index_n++) {
                    int reg_c_m = threadIdx.x * TM + index_m;
                    int reg_c_n = threadIdx.y * TN + index_n;
                    // tmp_sum[index_m * TN + index_n] += __bfloat162float(__hmul(Sa[reg_c_m * (BK + 1) + index_k], Sb[index_k * BN + reg_c_n]));
                    tmp_sum[index_m * TN + index_n] += __bfloat162float(FPMA_TC_fp8(Sa[reg_c_m * (BK + 1) + index_k], Sb[index_k * BN + reg_c_n], comp_table_shared));
                }
            }
        }
        __syncthreads();
    }
    for (int index_m = 0; index_m < TM; index_m++) {
        for (int index_n = 0; index_n < TN; index_n++) {
            int reg_c_m = threadIdx.x * TM + index_m;
            int reg_c_n = threadIdx.y * TN + index_n;
            int global_m = indA + reg_c_m;
            int global_n = indB + reg_c_n;
            if (global_m < M && global_n < N) {
                c[global_m * N + global_n] = __float2bfloat16(tmp_sum[index_m * TN + index_n]);
            }
        }
    }
}


template <int BM, int BN, int BK, int TM, int TN>
__global__ void Flex_FPMA_GEMM_comptable_fp8_us(uint8_t *a, uint8_t *b, __nv_bfloat16 *c, int M, int N, int K) {

    __shared__ uint16_t comp_table_shared[COMP_TABLE_SIZE][COMP_TABLE_SIZE];
    if (threadIdx.x < COMP_TABLE_SIZE && threadIdx.y < COMP_TABLE_SIZE) {
        comp_table_shared[threadIdx.x][threadIdx.y] = comp_table_const_fp8_8x8_us[threadIdx.x][threadIdx.y];
    }
    __syncthreads();
    // A [M, K], B [K, N], C [M, N]
    __shared__ uint8_t Sa[BM * (BK + 1)];
    __shared__ uint8_t Sb[BK * BN];

    int indA = TM * blockIdx.x * blockDim.x;
    int indB = TN * blockIdx.y * blockDim.y;
    int width = (K + BK - 1) / BK;
    float tmp_sum[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int tpb = blockDim.x * blockDim.y; // threads per block

    for (int ph = 0; ph < width; ph++) {
        // Cooperative load of A tile [BM x BK] into shared memory (row-major with (BK+1) stride)
        for (int i = tid; i < BM * BK; i += tpb) {
            int m = i % BM;           // 0..BM-1
            int k = i / BM;           // 0..BK-1
            int g_m = indA + m;
            int g_k = k + ph * BK;
            if (g_m < M && g_k < K) {
                Sa[m * (BK + 1) + k] = a[g_m * K + g_k];
            } else {
                Sa[m * (BK + 1) + k] = 0;
            }
        }
        // Cooperative load of B tile [BK x BN] into shared memory (row-major with BN stride)
        for (int i = tid; i < BK * BN; i += tpb) {
            int k = i / BN;           // 0..BK-1
            int n = i % BN;           // 0..BN-1
            int g_k = k + ph * BK;
            int g_n = n + indB;
            if (g_k < K && g_n < N) {
                Sb[k * BN + n] = b[g_k * N + g_n];
            } else {
                Sb[k * BN + n] = 0;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int index_k = 0; index_k < BK; index_k++) {
            for (int index_m = 0; index_m < TM; index_m++) {
                for (int index_n = 0; index_n < TN; index_n++) {
                    int reg_c_m = threadIdx.x * TM + index_m;
                    int reg_c_n = threadIdx.y * TN + index_n;
                    // tmp_sum[index_m * TN + index_n] += __bfloat162float(__hmul(Sa[reg_c_m * (BK + 1) + index_k], Sb[index_k * BN + reg_c_n]));
                    // tmp_sum[index_m * TN + index_n] += __bfloat162float(FPMA_TC_fp8(Sa[reg_c_m * (BK + 1) + index_k], Sb[index_k * BN + reg_c_n], comp_table_shared));
                    tmp_sum[index_m * TN + index_n] += __bfloat162float(FPMA_TC_fp8_us(Sa[reg_c_m * (BK + 1) + index_k], Sb[index_k * BN + reg_c_n], comp_table_shared));
                }
            }
        }
        __syncthreads();
    }
    for (int index_m = 0; index_m < TM; index_m++) {
        for (int index_n = 0; index_n < TN; index_n++) {
            int reg_c_m = threadIdx.x * TM + index_m;
            int reg_c_n = threadIdx.y * TN + index_n;
            int global_m = indA + reg_c_m;
            int global_n = indB + reg_c_n;
            if (global_m < M && global_n < N) {
                c[global_m * N + global_n] = __float2bfloat16(tmp_sum[index_m * TN + index_n]);
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN, int GROUP_SIZE>
__global__ void Flex_FPMA_GEMM_comptable_group_fp8(uint8_t *a, uint8_t *b, __nv_bfloat16 *c,
                                            __nv_bfloat16 *scale_a, __nv_bfloat16 *scale_b,
                                            int M, int N, int K) {

    // Compute how many complete groups are covered by the current tile.
    const int num_groups_per_tile = (BK + GROUP_SIZE - 1) / GROUP_SIZE;

    __shared__ uint16_t comp_table_shared[COMP_TABLE_SIZE][COMP_TABLE_SIZE];
    if (threadIdx.x < COMP_TABLE_SIZE && threadIdx.y < COMP_TABLE_SIZE) {
        comp_table_shared[threadIdx.x][threadIdx.y] = comp_table_const_fp8_8x8_us[threadIdx.x][threadIdx.y];
    }
    __syncthreads();

    // A [M, K], B [N, K] (stored as [K, N] but transposed in kernel), C [M, N]
    // scale_a [M, K//GROUP_SIZE], scale_b [N, K//GROUP_SIZE]
    __shared__ uint8_t Sa[BM * (BK + 1)];
    __shared__ uint8_t Sb[BK * BN];

    // Store the scales for the current tile: each BM row has num_groups_per_tile scales.
    __shared__ __nv_bfloat16 scale_a_shared[BM * 8];  // Support up to 8 groups per tile.
    // Each BN column has num_groups_per_tile scales.
    __shared__ __nv_bfloat16 scale_b_shared[BN * 8];  // Support up to 8 groups per tile.

    int indA = TM * blockIdx.x * blockDim.x;
    int indB = TN * blockIdx.y * blockDim.y;
    int width = (K + BK - 1) / BK;
    float tmp_sum[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int tpb = blockDim.x * blockDim.y; // threads per block
    int num_groups_total = (K + GROUP_SIZE - 1) / GROUP_SIZE;

    for (int ph = 0; ph < width; ph++) {
        // Cooperative load of A tile [BM x BK] into shared memory (row-major with (BK+1) stride)
        for (int i = tid; i < BM * BK; i += tpb) {
            int m = i % BM;           // 0..BM-1
            int k = i / BM;           // 0..BK-1
            int g_m = indA + m;
            int g_k = k + ph * BK;
            if (g_m < M && g_k < K) {
                Sa[m * (BK + 1) + k] = a[g_m * K + g_k];
            } else {
                Sa[m * (BK + 1) + k] = 0;
            }
        }
        // Cooperative load of B tile [BK x BN] into shared memory (row-major with BN stride)
        for (int i = tid; i < BK * BN; i += tpb) {
            int k = i / BN;           // 0..BK-1
            int n = i % BN;           // 0..BN-1
            int g_k = k + ph * BK;
            int g_n = n + indB;
            if (g_k < K && g_n < N) {
                Sb[k * BN + n] = b[g_k * N + g_n];
            } else {
                Sb[k * BN + n] = 0;
            }
        }
        // Load scale_a: [M, K//GROUP_SIZE]
        for (int i = tid; i < BM * num_groups_per_tile; i += tpb) {
            int m = i / num_groups_per_tile;
            int g_idx = i % num_groups_per_tile;
            int g_m = indA + m;
            int global_g_idx = (ph * BK) / GROUP_SIZE + g_idx;
            if (g_m < M && global_g_idx < num_groups_total) {
                scale_a_shared[m * num_groups_per_tile + g_idx] =
                    scale_a[g_m * num_groups_total + global_g_idx];
            } else {
                scale_a_shared[m * num_groups_per_tile + g_idx] =
                    __float2bfloat16(1.0f);
            }
        }
        // Load scale_b: [N, K//GROUP_SIZE]
        for (int i = tid; i < BN * num_groups_per_tile; i += tpb) {
            int n = i / num_groups_per_tile;
            int g_idx = i % num_groups_per_tile;
            int g_n = indB + n;
            int global_g_idx = (ph * BK) / GROUP_SIZE + g_idx;
            if (g_n < N && global_g_idx < num_groups_total) {
                scale_b_shared[n * num_groups_per_tile + g_idx] =
                    scale_b[g_n * num_groups_total + global_g_idx];
            } else {
                scale_b_shared[n * num_groups_per_tile + g_idx] =
                    __float2bfloat16(1.0f);
            }
        }
        __syncthreads();

        // Compute and dequantize group by group.
        for (int g = 0; g < num_groups_per_tile; g++) {
            int k_start = g * GROUP_SIZE;
            int k_end = min(k_start + GROUP_SIZE, BK);

            #pragma unroll
            for (int index_k = k_start; index_k < k_end; index_k++) {
                for (int index_m = 0; index_m < TM; index_m++) {
                    for (int index_n = 0; index_n < TN; index_n++) {
                        int reg_c_m = threadIdx.x * TM + index_m;
                        int reg_c_n = threadIdx.y * TN + index_n;

                        __nv_bfloat16 result = FPMA_TC_fp8_us(Sa[reg_c_m * (BK + 1) + index_k], Sb[index_k * BN + reg_c_n], comp_table_shared);

                        // Dequantize immediately: result * scale_a * scale_b.
                        __nv_bfloat16 scale_a_val = scale_a_shared[reg_c_m * num_groups_per_tile + g];
                        __nv_bfloat16 scale_b_val = scale_b_shared[reg_c_n * num_groups_per_tile + g];

                        // float scaled_result = __bfloat162float(result) *
                        //                     __bfloat162float(scale_a_val) *
                        //                     __bfloat162float(scale_b_val);
                        __nv_bfloat16 scale_ab = __hmul(scale_a_val, scale_b_val);
                        __nv_bfloat16 scaled_result = __hmul(result, scale_ab);

                        tmp_sum[index_m * TN + index_n] += __bfloat162float(scaled_result);
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int index_m = 0; index_m < TM; index_m++) {
        for (int index_n = 0; index_n < TN; index_n++) {
            int reg_c_m = threadIdx.x * TM + index_m;
            int reg_c_n = threadIdx.y * TN + index_n;
            int global_m = indA + reg_c_m;
            int global_n = indB + reg_c_n;
            if (global_m < M && global_n < N) {
                c[global_m * N + global_n] = __float2bfloat16(
                    tmp_sum[index_m * TN + index_n]
                );
            }
        }
    }
}


template <int BM, int BN, int BK, int TM, int TN, int GROUP_SIZE, bool USE_UNICORE_COMP>
__global__ void Flex_FPMA_GEMM_comptable_group_fp8_fixed(
    uint8_t *a, uint8_t *b, __nv_bfloat16 *c,
    __nv_bfloat16 *scale_a, __nv_bfloat16 *scale_b,
    int M, int N, int K) {

    __shared__ uint16_t comp_table_shared[COMP_TABLE_SIZE][COMP_TABLE_SIZE];
    if (threadIdx.x < COMP_TABLE_SIZE && threadIdx.y < COMP_TABLE_SIZE) {
        if constexpr (USE_UNICORE_COMP) {
            comp_table_shared[threadIdx.x][threadIdx.y] =
                comp_table_const_fp8_8x8_us[threadIdx.x][threadIdx.y];
        } else {
            comp_table_shared[threadIdx.x][threadIdx.y] =
                comp_table_const_fp8_8x8[threadIdx.x][threadIdx.y];
        }
    }
    __syncthreads();

    // A [M, K], B [K, N] (transposed), C [M, N]
    // scale_a [M, K//GROUP_SIZE], scale_b [N, K//GROUP_SIZE]
    __shared__ uint8_t Sa[BM * (BK + 1)];
    __shared__ uint8_t Sb[BK * BN];

    // Key fix: dynamically compute how many groups each tile actually spans.
    // Support tiles spanning up to 8 groups (for very small group sizes).
    __shared__ __nv_bfloat16 scale_a_shared[BM * 8];
    __shared__ __nv_bfloat16 scale_b_shared[BN * 8];

    int indA = TM * blockIdx.x * blockDim.x;
    int indB = TN * blockIdx.y * blockDim.y;
    int width = (K + BK - 1) / BK;
    float tmp_sum[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int tpb = blockDim.x * blockDim.y;
    int num_groups_total = (K + GROUP_SIZE - 1) / GROUP_SIZE;

    for (int ph = 0; ph < width; ph++) {
        // Compute the global k-range covered by the current tile.
        int k_tile_start = ph * BK;
        int k_tile_end = min(k_tile_start + BK, K);

        // Compute the range of groups spanned by the current tile.
        int group_start = k_tile_start / GROUP_SIZE;
        int group_end = (k_tile_end - 1) / GROUP_SIZE;
        int num_groups_in_tile = group_end - group_start + 1;

        // Load A tile [BM x BK]
        for (int i = tid; i < BM * BK; i += tpb) {
            int m = i % BM;
            int k = i / BM;
            int g_m = indA + m;
            int g_k = k + ph * BK;
            if (g_m < M && g_k < K) {
                Sa[m * (BK + 1) + k] = a[g_m * K + g_k];
            } else {
                Sa[m * (BK + 1) + k] = 0;
            }
        }

        // Load B tile [BK x BN]
        for (int i = tid; i < BK * BN; i += tpb) {
            int k = i / BN;
            int n = i % BN;
            int g_k = k + ph * BK;
            int g_n = n + indB;
            if (g_k < K && g_n < N) {
                Sb[k * BN + n] = b[g_k * N + g_n];
            } else {
                Sb[k * BN + n] = 0;
            }
        }

        // Fix: correctly load the scales for every group spanned by the current tile.
        // Load scale_a: for BM rows, load scales from group_start to group_end.
        for (int i = tid; i < BM * num_groups_in_tile; i += tpb) {
            int m = i / num_groups_in_tile;
            int g_offset = i % num_groups_in_tile;
            int g_m = indA + m;
            int global_g_idx = group_start + g_offset;
            if (g_m < M && global_g_idx < num_groups_total) {
                scale_a_shared[m * num_groups_in_tile + g_offset] =
                    scale_a[g_m * num_groups_total + global_g_idx];
            } else {
                scale_a_shared[m * num_groups_in_tile + g_offset] =
                    __float2bfloat16(1.0f);
            }
        }

        // Load scale_b: for BN columns, load scales from group_start to group_end.
        for (int i = tid; i < BN * num_groups_in_tile; i += tpb) {
            int n = i / num_groups_in_tile;
            int g_offset = i % num_groups_in_tile;
            int g_n = indB + n;
            int global_g_idx = group_start + g_offset;
            if (g_n < N && global_g_idx < num_groups_total) {
                scale_b_shared[n * num_groups_in_tile + g_offset] =
                    scale_b[g_n * num_groups_total + global_g_idx];
            } else {
                scale_b_shared[n * num_groups_in_tile + g_offset] =
                    __float2bfloat16(1.0f);
            }
        }
        __syncthreads();

        // Fix: determine the group using the actual k-value within the tile.
        for (int index_k = 0; index_k < BK && (k_tile_start + index_k) < K; index_k++) {
            // Compute which global group the current k-value belongs to.
            int global_k = k_tile_start + index_k;
            int global_group_idx = global_k / GROUP_SIZE;
            // Compute the index within the current tile's group array.
            int local_group_idx = global_group_idx - group_start;

            #pragma unroll
            for (int index_m = 0; index_m < TM; index_m++) {
                for (int index_n = 0; index_n < TN; index_n++) {
                    int reg_c_m = threadIdx.x * TM + index_m;
                    int reg_c_n = threadIdx.y * TN + index_n;

                    if (reg_c_m < BM && reg_c_n < BN) {
                        __nv_bfloat16 result;
                        if constexpr (USE_UNICORE_COMP) {
                            result = FPMA_TC_fp8_us(
                                Sa[reg_c_m * (BK + 1) + index_k],
                                Sb[index_k * BN + reg_c_n],
                                comp_table_shared
                            );
                        } else {
                            result = FPMA_TC_fp8(
                                Sa[reg_c_m * (BK + 1) + index_k],
                                Sb[index_k * BN + reg_c_n],
                                comp_table_shared
                            );
                        }

                        // Use the correct scale.
                        __nv_bfloat16 scale_a_val =
                            scale_a_shared[reg_c_m * num_groups_in_tile + local_group_idx];
                        __nv_bfloat16 scale_b_val =
                            scale_b_shared[reg_c_n * num_groups_in_tile + local_group_idx];

                        float scaled_result = __bfloat162float(result) *
                                            __bfloat162float(scale_a_val) *
                                            __bfloat162float(scale_b_val);

                        tmp_sum[index_m * TN + index_n] += scaled_result;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write results
    for (int index_m = 0; index_m < TM; index_m++) {
        for (int index_n = 0; index_n < TN; index_n++) {
            int reg_c_m = threadIdx.x * TM + index_m;
            int reg_c_n = threadIdx.y * TN + index_n;
            int global_m = indA + reg_c_m;
            int global_n = indB + reg_c_n;
            if (global_m < M && global_n < N) {
                c[global_m * N + global_n] = __float2bfloat16(
                    tmp_sum[index_m * TN + index_n]
                );
            }
        }
    }
}


template <int BM, int BN, int BK, int TM, int TN>
__global__ void Flex_FPMA_GEMM_fp4(uint8_t *a, uint8_t *b, __nv_bfloat16 *c, int M, int N, int K) {

    // A [M, K], B [K, N], C [M, N]
    __shared__ uint8_t Sa[BM * (BK + 1)];
    __shared__ uint8_t Sb[BK * BN];

    int indA = TM * blockIdx.x * blockDim.x;
    int indB = TN * blockIdx.y * blockDim.y;
    int width = (K + BK - 1) / BK;
    float tmp_sum[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int tpb = blockDim.x * blockDim.y; // threads per block

    for (int ph = 0; ph < width; ph++) {
        // Cooperative load of A tile [BM x BK] into shared memory (row-major with (BK+1) stride)
        for (int i = tid; i < BM * BK; i += tpb) {
            int m = i % BM;           // 0..BM-1
            int k = i / BM;           // 0..BK-1
            int g_m = indA + m;
            int g_k = k + ph * BK;
            if (g_m < M && g_k < K) {
                Sa[m * (BK + 1) + k] = a[g_m * K + g_k];
            } else {
                Sa[m * (BK + 1) + k] = 0;
            }
        }
        // Cooperative load of B tile [BK x BN] into shared memory (row-major with BN stride)
        for (int i = tid; i < BK * BN; i += tpb) {
            int k = i / BN;           // 0..BK-1
            int n = i % BN;           // 0..BN-1
            int g_k = k + ph * BK;
            int g_n = n + indB;
            if (g_k < K && g_n < N) {
                Sb[k * BN + n] = b[g_k * N + g_n];
            } else {
                Sb[k * BN + n] = 0;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int index_k = 0; index_k < BK; index_k++) {
            for (int index_m = 0; index_m < TM; index_m++) {
                for (int index_n = 0; index_n < TN; index_n++) {
                    int reg_c_m = threadIdx.x * TM + index_m;
                    int reg_c_n = threadIdx.y * TN + index_n;
                    // tmp_sum[index_m * TN + index_n] += __bfloat162float(__hmul(Sa[reg_c_m * (BK + 1) + index_k], Sb[index_k * BN + reg_c_n]));
                    tmp_sum[index_m * TN + index_n] += __bfloat162float(FPMA_TC_fp4(Sa[reg_c_m * (BK + 1) + index_k], Sb[index_k * BN + reg_c_n]));
                }
            }
        }
        __syncthreads();
    }
    for (int index_m = 0; index_m < TM; index_m++) {
        for (int index_n = 0; index_n < TN; index_n++) {
            int reg_c_m = threadIdx.x * TM + index_m;
            int reg_c_n = threadIdx.y * TN + index_n;
            int global_m = indA + reg_c_m;
            int global_n = indB + reg_c_n;
            if (global_m < M && global_n < N) {
                c[global_m * N + global_n] = __float2bfloat16(tmp_sum[index_m * TN + index_n]);
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void Flex_FPMA_GEMM_fp4_comptable(uint8_t *a, uint8_t *b, __nv_bfloat16 *c, int M, int N, int K) {

    __shared__ uint16_t comp_table_shared[COMP_TABLE_SIZE][COMP_TABLE_SIZE];
    if (threadIdx.x < COMP_TABLE_SIZE && threadIdx.y < COMP_TABLE_SIZE) {
        comp_table_shared[threadIdx.x][threadIdx.y] = comp_table_const_fp4_8x8[threadIdx.x][threadIdx.y];
    }
    __syncthreads();

    // A [M, K], B [K, N], C [M, N]
    __shared__ uint8_t Sa[BM * (BK + 1)];
    __shared__ uint8_t Sb[BK * BN];

    int indA = TM * blockIdx.x * blockDim.x;
    int indB = TN * blockIdx.y * blockDim.y;
    int width = (K + BK - 1) / BK;
    float tmp_sum[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int tpb = blockDim.x * blockDim.y; // threads per block

    for (int ph = 0; ph < width; ph++) {
        for (int i = tid; i < BM * BK; i += tpb) {
            int m = i % BM;
            int k = i / BM;
            int g_m = indA + m;
            int g_k = k + ph * BK;
            if (g_m < M && g_k < K) {
                Sa[m * (BK + 1) + k] = a[g_m * K + g_k];
            } else {
                Sa[m * (BK + 1) + k] = 0;
            }
        }
        for (int i = tid; i < BK * BN; i += tpb) {
            int k = i / BN;
            int n = i % BN;
            int g_k = k + ph * BK;
            int g_n = n + indB;
            if (g_k < K && g_n < N) {
                Sb[k * BN + n] = b[g_k * N + g_n];
            } else {
                Sb[k * BN + n] = 0;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int index_k = 0; index_k < BK; index_k++) {
            for (int index_m = 0; index_m < TM; index_m++) {
                for (int index_n = 0; index_n < TN; index_n++) {
                    int reg_c_m = threadIdx.x * TM + index_m;
                    int reg_c_n = threadIdx.y * TN + index_n;
                    tmp_sum[index_m * TN + index_n] += __bfloat162float(
                        FPMA_TC_fp4_comptable(
                            Sa[reg_c_m * (BK + 1) + index_k],
                            Sb[index_k * BN + reg_c_n],
                            comp_table_shared
                        )
                    );
                }
            }
        }
        __syncthreads();
    }
    for (int index_m = 0; index_m < TM; index_m++) {
        for (int index_n = 0; index_n < TN; index_n++) {
            int reg_c_m = threadIdx.x * TM + index_m;
            int reg_c_n = threadIdx.y * TN + index_n;
            int global_m = indA + reg_c_m;
            int global_n = indB + reg_c_n;
            if (global_m < M && global_n < N) {
                c[global_m * N + global_n] = __float2bfloat16(tmp_sum[index_m * TN + index_n]);
            }
        }
    }
}

void unicore_core_gemm_bf16(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C,
                                    // __nv_bfloat16* SA, __nv_bfloat16* SB,
                                    int M, int N, int K, int device_id) {
    cudaSetDevice(device_id);
    const int compute_kernel = 0; // 0->16bit, 1->8bit, 2->6bit, 3->4bit
    const int comp_method = 1;

    const int TM = 4, TN = 4;
    const int BLOCK_DIM_x = 16, BLOCK_DIM_y = 16;
    const int BM = TM * BLOCK_DIM_x;
    const int BN = TN * BLOCK_DIM_y;
    const int BK = 64;

    int num_blocks_x = (M + BM - 1) / BM;
    int num_blocks_y = (N + BN - 1) / BN;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y);
    dim3 grid_dim(num_blocks_x, num_blocks_y);

    if (comp_method == 0) {
        Flex_FPMA_GEMM<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    } else if (comp_method == 1) {
        Flex_FPMA_GEMM_comptable<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    }

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void unicore_core_gemm_fp16(half* A, half* B, half* C,
                                // __nv_bfloat16* SA, __nv_bfloat16* SB,
                                int M, int N, int K, int device_id) {
    cudaSetDevice(device_id);
    const int compute_kernel = 0; // 0->16bit, 1->8bit, 2->6bit, 3->4bit
    const int comp_method = 0;

    const int TM = 4, TN = 4;
    const int BLOCK_DIM_x = 16, BLOCK_DIM_y = 16;
    const int BM = TM * BLOCK_DIM_x;
    const int BN = TN * BLOCK_DIM_y;
    const int BK = 64;

    int num_blocks_x = (M + BM - 1) / BM;
    int num_blocks_y = (N + BN - 1) / BN;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y);
    dim3 grid_dim(num_blocks_x, num_blocks_y);


    if (comp_method == 0) {
        Flex_FPMA_GEMM_fp16<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    } else if (comp_method == 1) {
        Flex_FPMA_GEMM_comptable_fp16<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    }

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}



void unicore_core_gemm_fp8(uint8_t* A, uint8_t* B, __nv_bfloat16* C,
                                // __nv_bfloat16* SA, __nv_bfloat16* SB,
                                int M, int N, int K, int comp_method, int device_id) {
    cudaSetDevice(device_id);

    const int TM = 4, TN = 4;
    const int BLOCK_DIM_x = 16, BLOCK_DIM_y = 16;
    const int BM = TM * BLOCK_DIM_x;
    const int BN = TN * BLOCK_DIM_y;
    const int BK = 64;

    int num_blocks_x = (M + BM - 1) / BM;
    int num_blocks_y = (N + BN - 1) / BN;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y);
    dim3 grid_dim(num_blocks_x, num_blocks_y);

    if (comp_method == FP8_COMPENSATION_FPMG_CG) {
        Flex_FPMA_GEMM_comptable_fp8<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    } else if (comp_method == FP8_COMPENSATION_UNICORE) {
        Flex_FPMA_GEMM_comptable_fp8_us<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    } else {
        throw std::runtime_error("Unsupported fp8 compensation method");
    }

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}


void unicore_core_gemm_fp8_grouped(uint8_t* A, uint8_t* B, __nv_bfloat16* C,
                                __nv_bfloat16* SA, __nv_bfloat16* SB,
                                int M, int N, int K, int group_size, int comp_method, int device_id) {
    cudaSetDevice(device_id);

    const int TM = 4, TN = 4;
    const int BLOCK_DIM_x = 16, BLOCK_DIM_y = 16;
    const int BM = TM * BLOCK_DIM_x;
    const int BN = TN * BLOCK_DIM_y;
    const int BK = 64;

    int num_blocks_x = (M + BM - 1) / BM;
    int num_blocks_y = (N + BN - 1) / BN;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y);
    dim3 grid_dim(num_blocks_x, num_blocks_y);

    const bool use_unicore_comp = (comp_method == FP8_COMPENSATION_UNICORE);
    if (!(comp_method == FP8_COMPENSATION_FPMG_CG || comp_method == FP8_COMPENSATION_UNICORE)) {
        throw std::runtime_error("Unsupported fp8 compensation method");
    }

    // Dispatch the matching kernel based on group_size.
    if (group_size == 16) {
        if (use_unicore_comp) {
            Flex_FPMA_GEMM_comptable_group_fp8_fixed<BM, BN, BK, TM, TN, 16, true><<<grid_dim, block_dim>>>(A, B, C, SA, SB, M, N, K);
        } else {
            Flex_FPMA_GEMM_comptable_group_fp8_fixed<BM, BN, BK, TM, TN, 16, false><<<grid_dim, block_dim>>>(A, B, C, SA, SB, M, N, K);
        }
    } else if (group_size == 32) {
        if (use_unicore_comp) {
            Flex_FPMA_GEMM_comptable_group_fp8_fixed<BM, BN, BK, TM, TN, 32, true><<<grid_dim, block_dim>>>(A, B, C, SA, SB, M, N, K);
        } else {
            Flex_FPMA_GEMM_comptable_group_fp8_fixed<BM, BN, BK, TM, TN, 32, false><<<grid_dim, block_dim>>>(A, B, C, SA, SB, M, N, K);
        }
    } else if (group_size == 64) {
        if (use_unicore_comp) {
            Flex_FPMA_GEMM_comptable_group_fp8_fixed<BM, BN, BK, TM, TN, 64, true><<<grid_dim, block_dim>>>(A, B, C, SA, SB, M, N, K);
        } else {
            Flex_FPMA_GEMM_comptable_group_fp8_fixed<BM, BN, BK, TM, TN, 64, false><<<grid_dim, block_dim>>>(A, B, C, SA, SB, M, N, K);
        }
    } else if (group_size == 128) {
        if (use_unicore_comp) {
            Flex_FPMA_GEMM_comptable_group_fp8_fixed<BM, BN, BK, TM, TN, 128, true><<<grid_dim, block_dim>>>(A, B, C, SA, SB, M, N, K);
        } else {
            Flex_FPMA_GEMM_comptable_group_fp8_fixed<BM, BN, BK, TM, TN, 128, false><<<grid_dim, block_dim>>>(A, B, C, SA, SB, M, N, K);
        }
    } else {
        throw std::runtime_error("Unsupported group_size. Supported values: 16, 32, 64, 128");
    }

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}


void unicore_core_gemm_fp4(uint8_t* A, uint8_t* B, __nv_bfloat16* C,
                                // __nv_bfloat16* SA, __nv_bfloat16* SB,
                                int M, int N, int K, int device_id) {
    cudaSetDevice(device_id);

    const int TM = 4, TN = 4;
    const int BLOCK_DIM_x = 16, BLOCK_DIM_y = 16;
    const int BM = TM * BLOCK_DIM_x;
    const int BN = TN * BLOCK_DIM_y;
    const int BK = 64;

    int num_blocks_x = (M + BM - 1) / BM;
    int num_blocks_y = (N + BN - 1) / BN;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y);
    dim3 grid_dim(num_blocks_x, num_blocks_y);


    Flex_FPMA_GEMM_fp4<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(A, B, C, M, N, K);


    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void unicore_core_gemm_fp4_comptable(uint8_t* A, uint8_t* B, __nv_bfloat16* C,
                                int M, int N, int K, int device_id) {
    cudaSetDevice(device_id);

    const int TM = 4, TN = 4;
    const int BLOCK_DIM_x = 16, BLOCK_DIM_y = 16;
    const int BM = TM * BLOCK_DIM_x;
    const int BN = TN * BLOCK_DIM_y;
    const int BK = 64;

    int num_blocks_x = (M + BM - 1) / BM;
    int num_blocks_y = (N + BN - 1) / BN;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y);
    dim3 grid_dim(num_blocks_x, num_blocks_y);

    Flex_FPMA_GEMM_fp4_comptable<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(A, B, C, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
