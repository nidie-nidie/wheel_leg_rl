/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : MiniPC.h
  * @brief          : MiniPC interfaces functions 
  * @author         : GrassFan Wang
  * @date           : 2025/02/10
  * @version        : v1.0
  ******************************************************************************
  * @attention      : None
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef DEVICE_MINIPC_H
#define DEVICE_MINIPC_H


/* Includes ------------------------------------------------------------------*/
#include "stdint.h"
#include "stdbool.h" 


extern void MiniPC_Transmit_Info(uint8_t *Buff, uint16_t Len);

extern void MiniPC_Receive_Info(uint8_t *Buff, uint32_t Len);
extern void MiniPC_Recvive_Info(uint8_t* Buff, const uint32_t *Len);
extern void MiniPC_Send_IMU_Euler(float roll, float pitch, float yaw, float yaw_total);
extern uint32_t MiniPC_Get_RxPackets(void);
extern uint32_t MiniPC_Get_RxBytes(void);
extern uint16_t MiniPC_Get_LastRxLen(void);
#endif
