#ifndef SIM_PORT_BSP_CAN_H
#define SIM_PORT_BSP_CAN_H

#include <stdint.h>

typedef struct
{
    uint32_t Identifier;
    uint32_t DataLength;
} FDCAN_TxHeaderTypeDef;

typedef struct
{
    FDCAN_TxHeaderTypeDef Header;
    uint8_t Data[8];
} FDCAN_TxFrame_TypeDef;

extern FDCAN_TxFrame_TypeDef FDCAN3_TxFrame;

void USER_FDCAN_AddMessageToTxFifoQ(FDCAN_TxFrame_TypeDef *frame);

#endif

