#include "sucker.h"

extern ARM_CMD_data_t ARM_CMD_data;
void SUCKER_INIT(void)
{
//	  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_3);
//    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_3, 2000);
//    DWT_Delay(2);
//    HAL_TIM_PWM_Stop(&htim1, TIM_CHANNEL_3);
//    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_3, 1000);
//    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_3);
}
	
void SUCKER_TASK(void)
{
//	    if (ARM_CMD_data.sucker_mode == SUCKER_ON) {
//        __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, 2000);
//    } else {
//        __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, 1000);
//    }
//		
//    if (ARM_CMD_data.sucker_mode == SUCKER_HALF) {
//        __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, 1300);
//    }
		
		if (ARM_CMD_data.sucker_mode == SUCKER_ON) {
//        __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, 39999);
			HAL_GPIO_WritePin(GPIOE,GPIO_PIN_13,1);
    } else {
        HAL_GPIO_WritePin(GPIOE,GPIO_PIN_13,0);
    }
		
    if (ARM_CMD_data.sucker_mode == SUCKER_HALF) {
//        __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, 0);
    }
	
}