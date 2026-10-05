/*
 * File: _coder_LQR_K_api.h
 *
 * MATLAB Coder version            : 24.2
 * C/C++ source code generated on  : 2026-05-27 19:29:22
 */

#ifndef _CODER_LQR_K_API_H
#define _CODER_LQR_K_API_H

/* Include Files */
#include "emlrt.h"
#include "mex.h"
#include "tmwtypes.h"
#include <string.h>

/* Variable Declarations */
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
void LQR_K(real_T L_0, real_T K[12]);

void LQR_K_api(const mxArray *prhs, const mxArray **plhs);

void LQR_K_atexit(void);

void LQR_K_initialize(void);

void LQR_K_terminate(void);

void LQR_K_xil_shutdown(void);

void LQR_K_xil_terminate(void);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for _coder_LQR_K_api.h
 *
 * [EOF]
 */
