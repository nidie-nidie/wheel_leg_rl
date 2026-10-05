#ifndef SIM_PORT_ARM_MATH_H
#define SIM_PORT_ARM_MATH_H

#include <math.h>
#include <stdint.h>

typedef struct
{
    uint16_t numRows;
    uint16_t numCols;
    float *pData;
} arm_matrix_instance_f32;

typedef enum
{
    ARM_MATH_SUCCESS = 0,
    ARM_MATH_ARGUMENT_ERROR = -1,
    ARM_MATH_LENGTH_ERROR = -2,
    ARM_MATH_SIZE_MISMATCH = -3,
    ARM_MATH_NANINF = -4,
    ARM_MATH_SINGULAR = -5,
    ARM_MATH_TEST_FAILURE = -6
} arm_status;

static inline void arm_mat_init_f32(arm_matrix_instance_f32 *S, uint16_t nRows, uint16_t nColumns, float *pData)
{
    S->numRows = nRows;
    S->numCols = nColumns;
    S->pData = pData;
}

static inline arm_status arm_mat_add_f32(const arm_matrix_instance_f32 *pSrcA,
                                         const arm_matrix_instance_f32 *pSrcB,
                                         arm_matrix_instance_f32 *pDst)
{
    (void)pSrcA;
    (void)pSrcB;
    (void)pDst;
    return ARM_MATH_SUCCESS;
}

static inline arm_status arm_mat_sub_f32(const arm_matrix_instance_f32 *pSrcA,
                                         const arm_matrix_instance_f32 *pSrcB,
                                         arm_matrix_instance_f32 *pDst)
{
    (void)pSrcA;
    (void)pSrcB;
    (void)pDst;
    return ARM_MATH_SUCCESS;
}

static inline arm_status arm_mat_mult_f32(const arm_matrix_instance_f32 *pSrcA,
                                          const arm_matrix_instance_f32 *pSrcB,
                                          arm_matrix_instance_f32 *pDst)
{
    (void)pSrcA;
    (void)pSrcB;
    (void)pDst;
    return ARM_MATH_SUCCESS;
}

static inline arm_status arm_mat_trans_f32(const arm_matrix_instance_f32 *pSrc, arm_matrix_instance_f32 *pDst)
{
    (void)pSrc;
    (void)pDst;
    return ARM_MATH_SUCCESS;
}

static inline arm_status arm_mat_inverse_f32(const arm_matrix_instance_f32 *pSrc, arm_matrix_instance_f32 *pDst)
{
    (void)pSrc;
    (void)pDst;
    return ARM_MATH_SUCCESS;
}

static inline float arm_sin_f32(float x)
{
    return sinf(x);
}

static inline float arm_cos_f32(float x)
{
    return cosf(x);
}

static inline void arm_sqrt_f32(float in, float *out)
{
    *out = sqrtf(in);
}

#endif
