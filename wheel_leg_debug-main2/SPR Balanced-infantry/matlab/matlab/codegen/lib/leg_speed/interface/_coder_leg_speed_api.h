/*
 * File: _coder_leg_speed_api.h
 *
 * MATLAB Coder version            : 24.2
 * C/C++ source code generated on  : 2025-12-14 14:57:16
 */

#ifndef _CODER_LEG_SPEED_API_H
#define _CODER_LEG_SPEED_API_H

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
void leg_speed(real_T d_phi1, real_T d_phi4, real_T phi1, real_T phi4,
               real_T speed[2]);

void leg_speed_api(const mxArray *const prhs[4], const mxArray **plhs);

void leg_speed_atexit(void);

void leg_speed_initialize(void);

void leg_speed_terminate(void);

void leg_speed_xil_shutdown(void);

void leg_speed_xil_terminate(void);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for _coder_leg_speed_api.h
 *
 * [EOF]
 */
