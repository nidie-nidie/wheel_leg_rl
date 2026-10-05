/*
 * File: _coder_leg_position_mex.h
 *
 * MATLAB Coder version            : 24.2
 * C/C++ source code generated on  : 2025-12-14 14:58:52
 */

#ifndef _CODER_LEG_POSITION_MEX_H
#define _CODER_LEG_POSITION_MEX_H

/* Include Files */
#include "emlrt.h"
#include "mex.h"
#include "tmwtypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
MEXFUNCTION_LINKAGE void mexFunction(int32_T nlhs, mxArray *plhs[],
                                     int32_T nrhs, const mxArray *prhs[]);

emlrtCTX mexFunctionCreateRootTLS(void);

void unsafe_leg_position_mexFunction(int32_T nlhs, mxArray *plhs[1],
                                     int32_T nrhs, const mxArray *prhs[2]);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for _coder_leg_position_mex.h
 *
 * [EOF]
 */
