/*
 * File: leg_position_initialize.c
 *
 * MATLAB Coder version            : 24.2
 * C/C++ source code generated on  : 2025-12-14 14:58:52
 */

/* Include Files */
#include "leg_position_initialize.h"
#include "leg_position_data.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : void
 * Return Type  : void
 */
void leg_position_initialize(void)
{
  rt_InitInfAndNaN();
  isInitialized_leg_position = true;
}

/*
 * File trailer for leg_position_initialize.c
 *
 * [EOF]
 */
