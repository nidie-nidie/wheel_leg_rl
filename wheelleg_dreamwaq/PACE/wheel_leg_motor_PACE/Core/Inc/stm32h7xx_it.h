#ifndef WHEEL_LEG_MOTOR_PACE_IT_H
#define WHEEL_LEG_MOTOR_PACE_IT_H

#ifdef __cplusplus
extern "C" {
#endif

void NMI_Handler(void);
void HardFault_Handler(void);
void MemManage_Handler(void);
void BusFault_Handler(void);
void UsageFault_Handler(void);
void DebugMon_Handler(void);
void DMA1_Stream3_IRQHandler(void);
void DMA1_Stream4_IRQHandler(void);
void FDCAN3_IT0_IRQHandler(void);
void FDCAN3_IT1_IRQHandler(void);
void TIM2_IRQHandler(void);
void TIM6_DAC_IRQHandler(void);
void UART7_IRQHandler(void);

#ifdef __cplusplus
}
#endif

#endif
