/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : MiniPC.c
  * @brief          : MiniPC interfaces functions 
  * @author         : GarssFan Wang
  * @date           : 2025/01/22
  * @version        : v1.0
  ******************************************************************************
  * @attention      : None
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "MiniPC.h"
#include "Sim2Real.h"
#include "usbd_cdc_if.h"
#include "Config.h"

#define MINIPC_RX_BUFFER_SIZE 256U
#define MINIPC_TX_BUFFER_SIZE 128U

static volatile uint32_t minipc_rx_packets = 0U;
static volatile uint32_t minipc_rx_bytes = 0U;
static volatile uint16_t minipc_last_rx_len = 0U;
static uint8_t minipc_last_rx_buf[MINIPC_RX_BUFFER_SIZE];

static uint8_t minipc_tx_buf[2][MINIPC_TX_BUFFER_SIZE];
static uint8_t minipc_tx_buf_index = 0U;

static uint16_t MiniPC_Append_Str(uint8_t *buf, uint16_t pos, uint16_t max_len, const char *str)
{
    while ((str != NULL) && (*str != '\0') && (pos < max_len))
    {
        buf[pos++] = (uint8_t)(*str++);
    }
    return pos;
}

static uint16_t MiniPC_Append_U32(uint8_t *buf, uint16_t pos, uint16_t max_len, uint32_t value)
{
    char tmp[10];
    uint8_t len = 0U;

    do
    {
        tmp[len++] = (char)('0' + (value % 10U));
        value /= 10U;
    } while ((value > 0U) && (len < sizeof(tmp)));

    while ((len > 0U) && (pos < max_len))
    {
        buf[pos++] = (uint8_t)tmp[--len];
    }

    return pos;
}

static uint16_t MiniPC_Append_Milli_Deg(uint8_t *buf, uint16_t pos, uint16_t max_len, float rad)
{
    int32_t milli_deg = (int32_t)(rad * 57295.779513f + ((rad >= 0.0f) ? 0.5f : -0.5f));
    uint32_t abs_milli_deg;

    if (milli_deg < 0)
    {
        if (pos < max_len)
        {
            buf[pos++] = '-';
        }
        abs_milli_deg = (uint32_t)(-milli_deg);
    }
    else
    {
        abs_milli_deg = (uint32_t)milli_deg;
    }

    pos = MiniPC_Append_U32(buf, pos, max_len, abs_milli_deg / 1000U);
    if (pos < max_len)
    {
        buf[pos++] = '.';
    }

    uint32_t frac = abs_milli_deg % 1000U;
    if ((frac < 100U) && (pos < max_len))
    {
        buf[pos++] = '0';
    }
    if ((frac < 10U) && (pos < max_len))
    {
        buf[pos++] = '0';
    }
    pos = MiniPC_Append_U32(buf, pos, max_len, frac);

    return pos;
}

void MiniPC_Transmit_Info(uint8_t *Buff, uint16_t Len)
{
    if (Buff == NULL || Len == 0U)
    {
        return;
    }

    CDC_Transmit_HS(Buff, Len);
}

//usbd_cdc_if.c -> CDC_Receive_HS
void MiniPC_Receive_Info(uint8_t *Buff, uint32_t Len)
{
    uint32_t copy_len;

    if (Buff == NULL || Len == 0U)
    {
        return;
    }

    copy_len = Len;
    if (copy_len > MINIPC_RX_BUFFER_SIZE)
    {
        copy_len = MINIPC_RX_BUFFER_SIZE;
    }

    for (uint32_t i = 0U; i < copy_len; i++)
    {
        minipc_last_rx_buf[i] = Buff[i];
    }

    minipc_last_rx_len = (uint16_t)copy_len;
    minipc_rx_bytes += Len;
    minipc_rx_packets++;

#if ENABLE_SIM2REAL_PROTOCOL
    Sim2Real_RxBytes(Buff, Len);
#endif
}

void MiniPC_Recvive_Info(uint8_t* Buff, const uint32_t *Len)
{
    if (Len == NULL)
    {
        return;
    }

    MiniPC_Receive_Info(Buff, *Len);
}

void MiniPC_Send_IMU_Euler(float roll, float pitch, float yaw, float yaw_total)
{
#if ENABLE_USB_IMU_EULER_TX
    uint8_t next_index = (uint8_t)(minipc_tx_buf_index ^ 1U);
    uint8_t *buf = minipc_tx_buf[next_index];
    uint16_t pos = 0U;
    uint8_t result;

    pos = MiniPC_Append_Str(buf, pos, MINIPC_TX_BUFFER_SIZE, "imu_euler_deg roll=");
    pos = MiniPC_Append_Milli_Deg(buf, pos, MINIPC_TX_BUFFER_SIZE, roll);
    pos = MiniPC_Append_Str(buf, pos, MINIPC_TX_BUFFER_SIZE, " pitch=");
    pos = MiniPC_Append_Milli_Deg(buf, pos, MINIPC_TX_BUFFER_SIZE, pitch);
    pos = MiniPC_Append_Str(buf, pos, MINIPC_TX_BUFFER_SIZE, " yaw=");
    pos = MiniPC_Append_Milli_Deg(buf, pos, MINIPC_TX_BUFFER_SIZE, yaw);
    pos = MiniPC_Append_Str(buf, pos, MINIPC_TX_BUFFER_SIZE, " yaw_total=");
    pos = MiniPC_Append_Milli_Deg(buf, pos, MINIPC_TX_BUFFER_SIZE, yaw_total);
    pos = MiniPC_Append_Str(buf, pos, MINIPC_TX_BUFFER_SIZE, " rx=");
    pos = MiniPC_Append_U32(buf, pos, MINIPC_TX_BUFFER_SIZE, minipc_rx_packets);
    pos = MiniPC_Append_Str(buf, pos, MINIPC_TX_BUFFER_SIZE, " last_len=");
    pos = MiniPC_Append_U32(buf, pos, MINIPC_TX_BUFFER_SIZE, minipc_last_rx_len);
    pos = MiniPC_Append_Str(buf, pos, MINIPC_TX_BUFFER_SIZE, "\r\n");

    result = CDC_Transmit_HS(buf, pos);
    if (result == USBD_OK)
    {
        minipc_tx_buf_index = next_index;
    }
#else
    (void)roll;
    (void)pitch;
    (void)yaw;
    (void)yaw_total;
#endif
}

uint32_t MiniPC_Get_RxPackets(void)
{
    return minipc_rx_packets;
}

uint32_t MiniPC_Get_RxBytes(void)
{
    return minipc_rx_bytes;
}

uint16_t MiniPC_Get_LastRxLen(void)
{
    return minipc_last_rx_len;
}
