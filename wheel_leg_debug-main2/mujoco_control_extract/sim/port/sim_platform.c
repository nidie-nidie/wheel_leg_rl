#include "bsp_can.h"
#include "cmsis_os.h"
#include "INS_Task.h"
#include "PS2_Task.h"
#include "Remote_Control.h"

FDCAN_TxFrame_TypeDef FDCAN3_TxFrame;
INS_t INS;
BMI088_t BMI088;

bool ps2_lost = false;
uint8_t vbus_low_warning = 0;
Remote_Info_Typedef remote_ctrl;

static uint32_t sim_tick_ms = 0;

void osDelay(uint32_t ms)
{
    sim_tick_ms += ms;
}

TickType_t osKernelSysTick(void)
{
    return sim_tick_ms;
}

uint32_t HAL_GetTick(void)
{
    return sim_tick_ms;
}

void USER_FDCAN_AddMessageToTxFifoQ(FDCAN_TxFrame_TypeDef *frame)
{
    (void)frame;
}

uint32_t GetPs2IdleTime(void)
{
    return 0;
}

bool GetRcOffline(void)
{
    return false;
}
