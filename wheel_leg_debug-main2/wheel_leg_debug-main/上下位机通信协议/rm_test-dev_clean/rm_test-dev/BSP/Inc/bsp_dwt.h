#ifndef BSP_DWT_H
#define BSP_DWT_H

/* Includes ------------------------------------------------------------------*/
#include "stdint.h"

/* Exported types ------------------------------------------------------------*/
/**
 * @brief typedef structure that contains the information for the DWT timer.
 */
typedef struct
{
  uint32_t s;
  uint16_t ms;
  uint16_t us;
} DWT_Time_Typedef;

extern void BSP_DWT_Init(uint32_t CPU_Freq_mHz);
extern float DWT_GetDeltaT(uint32_t *cnt_last);
extern double DWT_GetDeltaT64(uint32_t *cnt_last);
extern float DWT_GetTimeline_s(void);
extern float DWT_GetTimeline_ms(void);
extern uint64_t DWT_GetTimeline_us(void);
extern void DWT_Delay(float Delay);
extern void DWT_SysTimeUpdate(void);
extern void DWT_Delay_us(uint32_t us);

#endif