/*
 * File: _coder_leg_position_api.h
 *
 * MATLAB Coder version            : 24.2
 * C/C++ source code generated on  : 2025-12-14 14:58:52
 */

#ifndef _CODER_LEG_POSITION_API_H
#define _CODER_LEG_POSITION_API_H

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
void leg_position(real_T phi1, real_T phi4, real_T position[2]);

void leg_position_api(const mxArray *const prhs[2], const mxArray **plhs);

void leg_position_atexit(void);

void leg_position_initialize(void);

void leg_position_terminate(void);

void leg_position_xil_shutdown(void);

void leg_position_xil_terminate(void);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for _coder_leg_position_api.h
 *
 * [EOF]
 */
