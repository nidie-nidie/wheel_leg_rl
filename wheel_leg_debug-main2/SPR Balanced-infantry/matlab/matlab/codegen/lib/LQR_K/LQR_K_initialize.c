/*
 * File: LQR_K_initialize.c
 *
 * MATLAB Coder version            : 24.2
 * C/C++ source code generated on  : 2026-05-27 19:29:22
 */

/* Include Files */
#include "LQR_K_initialize.h"
#include "LQR_K_data.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : void
 * Return Type  : void
 */
void LQR_K_initialize(void)
{
  rt_InitInfAndNaN();
  isInitialized_LQR_K = true;
}

/*
 * File trailer for LQR_K_initialize.c
 *
 * [EOF]
 */
