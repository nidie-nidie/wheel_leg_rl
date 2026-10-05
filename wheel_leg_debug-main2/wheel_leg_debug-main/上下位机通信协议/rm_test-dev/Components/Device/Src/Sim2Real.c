#include "Sim2Real.h"

#include "Config.h"
#include "INS_Task.h"
#include "Motor.h"
#include "bsp_adc.h"
#include "bsp_dwt.h"
#include "stm32h7xx_hal.h"
#include "string.h"
#include "usbd_cdc_if.h"

#define SIM2REAL_RX_BUFFER_SIZE 512U
#define SIM2REAL_MAX_PAYLOAD_LEN SIM2REAL_STATE_PAYLOAD_LEN

static uint8_t sim2real_rx_buf[SIM2REAL_RX_BUFFER_SIZE];
static uint16_t sim2real_rx_len = 0U;

static Sim2Real_Command_t sim2real_last_cmd;
static bool sim2real_cmd_valid = false;
static uint32_t sim2real_valid_cmd_count = 0U;
static uint32_t sim2real_crc_error_count = 0U;
static uint32_t sim2real_bad_frame_count = 0U;
static uint16_t sim2real_fault_latched = SIM2REAL_FAULT_NONE;

static uint32_t sim2real_state_seq = 0U;
static uint8_t sim2real_state_tx_buf[2][SIM2REAL_STATE_FRAME_LEN];
static uint8_t sim2real_state_tx_index = 0U;

static uint8_t Sim2Real_CurrentStatus(void);
static bool Sim2Real_IsEstopHoldActive(void);
static void Sim2Real_HandleIdleMode(void);
static void Sim2Real_HandleReadyMode(void);
static void Sim2Real_HandleRunMode(void);
static void Sim2Real_HandleDampingMode(void);
static void Sim2Real_HandleEstopMode(void);
static void Sim2Real_HandleFaultMode(void);

static const uint8_t sim2real_leg_dm_index[SIM2REAL_LEG_NUM] = {
    SIM2REAL_LEG_JIO_DM_IDX,
    SIM2REAL_LEG_JAG_DM_IDX,
    SIM2REAL_LEG_JIJ_DM_IDX,
    SIM2REAL_LEG_JAB_DM_IDX,
};

static const float sim2real_leg_sign[SIM2REAL_LEG_NUM] = {1.0f, 1.0f, 1.0f, 1.0f};
static const float sim2real_leg_zero_offset[SIM2REAL_LEG_NUM] = {0.0f, 0.0f, 0.0f, 0.0f};
static const float sim2real_wheel_sign[SIM2REAL_WHEEL_NUM] = {1.0f, 1.0f};

static bool sim2real_motors_enabled = false;
static uint8_t sim2real_enable_repeat_remaining = 0U;
static uint8_t sim2real_enable_actuator_index = 0U;
static uint16_t sim2real_motor_tx_tick = 0U;
static bool sim2real_run_blocked_not_armed = false;
static bool sim2real_estop_hold_active = false;
static uint32_t sim2real_estop_hold_start_ms = 0U;
static bool sim2real_control_status_valid = false;
static uint8_t sim2real_last_control_status = SIM2REAL_MODE_IDLE;
static bool sim2real_control_status_changed = true;

static DM_Motor_Info_Typedef *Sim2Real_GetLegMotor(uint8_t protocol_leg_index)
{
    if (protocol_leg_index >= SIM2REAL_LEG_NUM)
    {
        protocol_leg_index = 0U;
    }

    return &DM_8009_Motor[sim2real_leg_dm_index[protocol_leg_index]];
}

static float Sim2Real_Clamp(float value, float min_value, float max_value)
{
    if (value > max_value)
    {
        return max_value;
    }
    if (value < min_value)
    {
        return min_value;
    }
    return value;
}

static float Sim2Real_AbsLimit(float requested_limit, float hard_limit)
{
    float limit = requested_limit;
    if (limit < 0.0f)
    {
        limit = -limit;
    }
    if (limit > hard_limit)
    {
        limit = hard_limit;
    }
    return limit;
}

static float Sim2Real_LegMotorPositionToProtocol(uint8_t protocol_leg_index)
{
    return sim2real_leg_sign[protocol_leg_index] *
           (Sim2Real_GetLegMotor(protocol_leg_index)->Data.Position - sim2real_leg_zero_offset[protocol_leg_index]);
}

static float Sim2Real_LegMotorVelocityToProtocol(uint8_t protocol_leg_index)
{
    return sim2real_leg_sign[protocol_leg_index] * Sim2Real_GetLegMotor(protocol_leg_index)->Data.Velocity;
}

static float Sim2Real_LegMotorTorqueToProtocol(uint8_t protocol_leg_index)
{
    return sim2real_leg_sign[protocol_leg_index] * Sim2Real_GetLegMotor(protocol_leg_index)->Data.Torque;
}

static float Sim2Real_LegProtocolPositionToMotor(uint8_t protocol_leg_index, float position)
{
    return sim2real_leg_sign[protocol_leg_index] * position + sim2real_leg_zero_offset[protocol_leg_index];
}

static float Sim2Real_LegProtocolVelocityToMotor(uint8_t protocol_leg_index, float velocity)
{
    return sim2real_leg_sign[protocol_leg_index] * velocity;
}

static float Sim2Real_WheelVelocityToProtocol(uint8_t wheel_index)
{
    return sim2real_wheel_sign[wheel_index] * LK_9025_Motor[wheel_index].Data.Velocity;
}

static float Sim2Real_WheelProtocolVelocityToMotor(uint8_t wheel_index, float velocity)
{
    return sim2real_wheel_sign[wheel_index] * velocity;
}

static uint16_t Sim2Real_ReadBE16(const uint8_t *p)
{
    return (uint16_t)(((uint16_t)p[0] << 8) | (uint16_t)p[1]);
}

static uint32_t Sim2Real_ReadBE32(const uint8_t *p)
{
    return ((uint32_t)p[0] << 24) |
           ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) |
           ((uint32_t)p[3]);
}

static uint64_t Sim2Real_ReadBE64(const uint8_t *p)
{
    return ((uint64_t)Sim2Real_ReadBE32(p) << 32) | (uint64_t)Sim2Real_ReadBE32(p + 4);
}

static float Sim2Real_ReadBEFloat(const uint8_t *p)
{
    uint32_t raw = Sim2Real_ReadBE32(p);
    float value;
    memcpy(&value, &raw, sizeof(value));
    return value;
}

static void Sim2Real_WriteBE16(uint8_t *p, uint16_t value)
{
    p[0] = (uint8_t)(value >> 8);
    p[1] = (uint8_t)value;
}

static void Sim2Real_WriteBE32(uint8_t *p, uint32_t value)
{
    p[0] = (uint8_t)(value >> 24);
    p[1] = (uint8_t)(value >> 16);
    p[2] = (uint8_t)(value >> 8);
    p[3] = (uint8_t)value;
}

static void Sim2Real_WriteBE64(uint8_t *p, uint64_t value)
{
    Sim2Real_WriteBE32(p, (uint32_t)(value >> 32));
    Sim2Real_WriteBE32(p + 4, (uint32_t)value);
}

static void Sim2Real_WriteBEFloat(uint8_t *p, float value)
{
    uint32_t raw;
    memcpy(&raw, &value, sizeof(raw));
    Sim2Real_WriteBE32(p, raw);
}

static uint32_t Sim2Real_Crc32(const uint8_t *data, uint32_t len)
{
    uint32_t crc = 0xFFFFFFFFU;

    for (uint32_t i = 0U; i < len; i++)
    {
        crc ^= data[i];
        for (uint8_t bit = 0U; bit < 8U; bit++)
        {
            if ((crc & 1U) != 0U)
            {
                crc = (crc >> 1) ^ 0xEDB88320U;
            }
            else
            {
                crc >>= 1;
            }
        }
    }

    return crc ^ 0xFFFFFFFFU;
}

static void Sim2Real_RemoveRxBytes(uint16_t len)
{
    if (len >= sim2real_rx_len)
    {
        sim2real_rx_len = 0U;
        return;
    }

    memmove(sim2real_rx_buf, sim2real_rx_buf + len, sim2real_rx_len - len);
    sim2real_rx_len = (uint16_t)(sim2real_rx_len - len);
}

static void Sim2Real_ParseCommandPayload(const uint8_t *payload, uint32_t seq, uint64_t t_us)
{
    uint16_t off = 0U;
    Sim2Real_Command_t cmd;

    memset(&cmd, 0, sizeof(cmd));
    cmd.mode = payload[off++];
    cmd.reserved = payload[off++];
    cmd.ttl_ms = Sim2Real_ReadBE16(payload + off);
    off += 2U;

    for (uint8_t i = 0U; i < 4U; i++, off += 4U)
    {
        cmd.leg_q_des_rad[i] = Sim2Real_ReadBEFloat(payload + off);
    }
    for (uint8_t i = 0U; i < 4U; i++, off += 4U)
    {
        cmd.leg_dq_des_rad_s[i] = Sim2Real_ReadBEFloat(payload + off);
    }
    for (uint8_t i = 0U; i < 2U; i++, off += 4U)
    {
        cmd.wheel_dq_des_rad_s[i] = Sim2Real_ReadBEFloat(payload + off);
    }
    for (uint8_t i = 0U; i < 4U; i++, off += 4U)
    {
        cmd.leg_kp[i] = Sim2Real_ReadBEFloat(payload + off);
    }
    for (uint8_t i = 0U; i < 4U; i++, off += 4U)
    {
        cmd.leg_kd[i] = Sim2Real_ReadBEFloat(payload + off);
    }
    for (uint8_t i = 0U; i < 2U; i++, off += 4U)
    {
        cmd.wheel_kp[i] = Sim2Real_ReadBEFloat(payload + off);
    }
    for (uint8_t i = 0U; i < 2U; i++, off += 4U)
    {
        cmd.wheel_kd[i] = Sim2Real_ReadBEFloat(payload + off);
    }
    for (uint8_t i = 0U; i < 6U; i++, off += 4U)
    {
        cmd.tau_limit_nm[i] = Sim2Real_ReadBEFloat(payload + off);
    }
    for (uint8_t i = 0U; i < 6U; i++, off += 4U)
    {
        cmd.dq_limit_rad_s[i] = Sim2Real_ReadBEFloat(payload + off);
    }

    cmd.seq = seq;
    cmd.t_us = t_us;
    cmd.rx_tick_ms = HAL_GetTick();

    sim2real_last_cmd = cmd;
    if (cmd.mode == SIM2REAL_MODE_ESTOP)
    {
        sim2real_estop_hold_active = true;
        sim2real_estop_hold_start_ms = cmd.rx_tick_ms;
    }

    sim2real_cmd_valid = true;
    sim2real_valid_cmd_count++;
}

static void Sim2Real_ParseRxBuffer(void)
{
    while (sim2real_rx_len >= SIM2REAL_HEADER_LEN)
    {
        uint16_t sof = Sim2Real_ReadBE16(sim2real_rx_buf);
        if (sof != SIM2REAL_SOF)
        {
            uint16_t next = 1U;
            while ((next + 1U) < sim2real_rx_len)
            {
                if (Sim2Real_ReadBE16(sim2real_rx_buf + next) == SIM2REAL_SOF)
                {
                    break;
                }
                next++;
            }
            Sim2Real_RemoveRxBytes(next);
            continue;
        }

        uint8_t ver = sim2real_rx_buf[2];
        uint8_t msg_type = sim2real_rx_buf[3];
        uint16_t payload_len = Sim2Real_ReadBE16(sim2real_rx_buf + 4);
        uint32_t seq = Sim2Real_ReadBE32(sim2real_rx_buf + 6);
        uint64_t t_us = Sim2Real_ReadBE64(sim2real_rx_buf + 10);

        if (payload_len > SIM2REAL_MAX_PAYLOAD_LEN)
        {
            sim2real_bad_frame_count++;
            sim2real_fault_latched |= SIM2REAL_FAULT_BAD_FRAME;
            Sim2Real_RemoveRxBytes(1U);
            continue;
        }

        uint16_t frame_len = (uint16_t)(SIM2REAL_HEADER_LEN + payload_len + SIM2REAL_CRC_LEN);
        if (sim2real_rx_len < frame_len)
        {
            break;
        }

        uint32_t rx_crc = Sim2Real_ReadBE32(sim2real_rx_buf + SIM2REAL_HEADER_LEN + payload_len);
        uint32_t calc_crc = Sim2Real_Crc32(sim2real_rx_buf, SIM2REAL_HEADER_LEN + payload_len);
        if (rx_crc != calc_crc)
        {
            sim2real_crc_error_count++;
            sim2real_fault_latched |= SIM2REAL_FAULT_CRC;
            Sim2Real_RemoveRxBytes(1U);
            continue;
        }

        if ((ver == SIM2REAL_VER) && (msg_type == SIM2REAL_MSG_COMMAND) && (payload_len == SIM2REAL_CMD_PAYLOAD_LEN))
        {
            Sim2Real_ParseCommandPayload(sim2real_rx_buf + SIM2REAL_HEADER_LEN, seq, t_us);
        }
        else
        {
            sim2real_bad_frame_count++;
            sim2real_fault_latched |= SIM2REAL_FAULT_BAD_FRAME;
        }

        Sim2Real_RemoveRxBytes(frame_len);
    }
}

void Sim2Real_RxBytes(const uint8_t *data, uint32_t len)
{
    if ((data == NULL) || (len == 0U))
    {
        return;
    }

    for (uint32_t i = 0U; i < len; i++)
    {
        if (sim2real_rx_len < SIM2REAL_RX_BUFFER_SIZE)
        {
            sim2real_rx_buf[sim2real_rx_len++] = data[i];
        }
        else
        {
            sim2real_rx_len = 0U;
            sim2real_fault_latched |= SIM2REAL_FAULT_RX_OVERFLOW;
        }
    }

    Sim2Real_ParseRxBuffer();
}

static void Sim2Real_SendOneEnableCommand(uint8_t actuator_index)
{
    switch (actuator_index)
    {
    case 0U:
        DM_Motor_Command(&FDCAN3_TxFrame, &DM_8009_Motor[0], DM_Motor_Enable);
        break;
    case 1U:
        DM_Motor_Command(&FDCAN3_TxFrame, &DM_8009_Motor[1], DM_Motor_Enable);
        break;
    case 2U:
        LK_Motor_Command(&FDCAN3_TxFrame, &LK_9025_Motor[0], LK_Motor_Enable);
        break;
    case 3U:
        DM_Motor_Command(&FDCAN3_TxFrame, &DM_8009_Motor[2], DM_Motor_Enable);
        break;
    case 4U:
        DM_Motor_Command(&FDCAN3_TxFrame, &DM_8009_Motor[3], DM_Motor_Enable);
        break;
    case 5U:
        LK_Motor_Command(&FDCAN3_TxFrame, &LK_9025_Motor[1], LK_Motor_Enable);
        break;
    default:
        break;
    }
}

static void Sim2Real_SendDisableCommands(void)
{
    for (uint8_t i = 0U; i < SIM2REAL_LEG_NUM; i++)
    {
        DM_Motor_Command(&FDCAN3_TxFrame, &DM_8009_Motor[i], DM_Motor_Disable);
    }
    for (uint8_t i = 0U; i < SIM2REAL_WHEEL_NUM; i++)
    {
        LK_Motor_Command(&FDCAN3_TxFrame, &LK_9025_Motor[i], LK_Motor_Disable);
    }
}

static void Sim2Real_RequestEnable(void)
{
    if (!sim2real_motors_enabled && (sim2real_enable_repeat_remaining == 0U))
    {
        sim2real_enable_repeat_remaining = SIM2REAL_ENABLE_REPEAT_COUNT;
        sim2real_enable_actuator_index = 0U;
    }
}

static bool Sim2Real_PumpEnable(void)
{
    if (sim2real_enable_repeat_remaining == 0U)
    {
        return false;
    }

    Sim2Real_SendOneEnableCommand(sim2real_enable_actuator_index);
    sim2real_enable_repeat_remaining--;

    if (sim2real_enable_repeat_remaining == 0U)
    {
        sim2real_enable_actuator_index++;

        if (sim2real_enable_actuator_index >= SIM2REAL_ACTUATOR_NUM)
        {
            sim2real_enable_actuator_index = 0U;
            sim2real_motors_enabled = true;
        }
        else
        {
            sim2real_enable_repeat_remaining = SIM2REAL_ENABLE_REPEAT_COUNT;
        }
    }

    return true;
}

static void Sim2Real_DisableMotors(void)
{
    Sim2Real_SendDisableCommands();
    sim2real_motors_enabled = false;
    sim2real_enable_repeat_remaining = 0U;
    sim2real_enable_actuator_index = 0U;
    sim2real_run_blocked_not_armed = false;
}

static void Sim2Real_SendDampingCommands(void)
{
    for (uint8_t i = 0U; i < SIM2REAL_LEG_NUM; i++)
    {
        DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, Sim2Real_GetLegMotor(i), 0.0f, 0.0f, 0.0f, SIM2REAL_DAMPING_DM_KD, 0.0f);
    }
    for (uint8_t i = 0U; i < SIM2REAL_WHEEL_NUM; i++)
    {
        LK_Motor_CAN_TxTorqueMessage(&FDCAN3_TxFrame, &LK_9025_Motor[i], 0.0f);
    }
}

#if ENABLE_SIM2REAL_IDLE_CAN_KEEPALIVE
static void Sim2Real_SendZeroOutputCommands(void)
{
    for (uint8_t i = 0U; i < SIM2REAL_LEG_NUM; i++)
    {
        DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, Sim2Real_GetLegMotor(i), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (uint8_t i = 0U; i < SIM2REAL_WHEEL_NUM; i++)
    {
        LK_Motor_CAN_TxTorqueMessage(&FDCAN3_TxFrame, &LK_9025_Motor[i], 0.0f);
    }
}
#endif

static void Sim2Real_SendRunCommands(void)
{
    for (uint8_t i = 0U; i < SIM2REAL_LEG_NUM; i++)
    {
        DM_Motor_Info_Typedef *motor = Sim2Real_GetLegMotor(i);
        float dq_limit = Sim2Real_AbsLimit(sim2real_last_cmd.dq_limit_rad_s[i], SIM2REAL_HARD_LEG_DQ_LIMIT_RAD_S);
        float q_des = Sim2Real_LegProtocolPositionToMotor(i, sim2real_last_cmd.leg_q_des_rad[i]);
        float dq_des = (dq_limit > 0.0f) ? Sim2Real_LegProtocolVelocityToMotor(i, Sim2Real_Clamp(sim2real_last_cmd.leg_dq_des_rad_s[i], -dq_limit, dq_limit)) : 0.0f;
        float kp = Sim2Real_Clamp(sim2real_last_cmd.leg_kp[i], 0.0f, DM_KP_MAX);
        float kd = Sim2Real_Clamp(sim2real_last_cmd.leg_kd[i], 0.0f, DM_KD_MAX);

        q_des = Sim2Real_Clamp(q_des, motor->Param_Range.P_MIN, motor->Param_Range.P_MAX);
        dq_des = Sim2Real_Clamp(dq_des, motor->Param_Range.V_MIN, motor->Param_Range.V_MAX);
        DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, motor, q_des, dq_des, kp, kd, 0.0f);
    }

    for (uint8_t i = 0U; i < SIM2REAL_WHEEL_NUM; i++)
    {
        uint8_t actuator_index = (uint8_t)(SIM2REAL_LEG_NUM + i);
        float dq_limit = Sim2Real_AbsLimit(sim2real_last_cmd.dq_limit_rad_s[actuator_index], SIM2REAL_HARD_WHEEL_DQ_LIMIT_RAD_S);
        float dq_des = (dq_limit > 0.0f) ? Sim2Real_WheelProtocolVelocityToMotor(i, Sim2Real_Clamp(sim2real_last_cmd.wheel_dq_des_rad_s[i], -dq_limit, dq_limit)) : 0.0f;

        LK_Motor_CAN_TxVelocityMessage(&FDCAN3_TxFrame, &LK_9025_Motor[i], dq_des);
    }
}

static bool Sim2Real_IsEstopHoldActive(void)
{
    if (!sim2real_estop_hold_active)
    {
        return false;
    }

    if ((HAL_GetTick() - sim2real_estop_hold_start_ms) < SIM2REAL_ESTOP_HOLD_MS)
    {
        return true;
    }

    sim2real_estop_hold_active = false;
    return false;
}

static void Sim2Real_HandleIdleMode(void)
{
    if (sim2real_control_status_changed || sim2real_motors_enabled || (sim2real_enable_repeat_remaining != 0U))
    {
        Sim2Real_DisableMotors();
    }
}

static void Sim2Real_HandleReadyMode(void)
{
    sim2real_run_blocked_not_armed = false;
    Sim2Real_RequestEnable();

    if (Sim2Real_PumpEnable())
    {
        return;
    }

    if (sim2real_motors_enabled)
    {
        Sim2Real_SendDampingCommands();
    }
}

static void Sim2Real_HandleRunMode(void)
{
    if (!sim2real_motors_enabled)
    {
        sim2real_run_blocked_not_armed = true;
        if (sim2real_control_status_changed || (sim2real_enable_repeat_remaining != 0U))
        {
            Sim2Real_DisableMotors();
        }
        return;
    }

    sim2real_run_blocked_not_armed = false;
#if ENABLE_SIM2REAL_RUN_OUTPUT
    Sim2Real_SendRunCommands();
#else
    Sim2Real_SendDampingCommands();
#endif
}

static void Sim2Real_HandleDampingMode(void)
{
    sim2real_enable_repeat_remaining = 0U;
    sim2real_enable_actuator_index = 0U;
    sim2real_run_blocked_not_armed = false;
    if (sim2real_motors_enabled)
    {
        Sim2Real_SendDampingCommands();
    }
}

static void Sim2Real_HandleEstopMode(void)
{
    if (sim2real_control_status_changed || sim2real_motors_enabled || (sim2real_enable_repeat_remaining != 0U))
    {
        Sim2Real_DisableMotors();
    }
}

static void Sim2Real_HandleFaultMode(void)
{
    if (sim2real_control_status_changed || sim2real_motors_enabled || (sim2real_enable_repeat_remaining != 0U))
    {
        Sim2Real_DisableMotors();
    }
}

void Sim2Real_ControlStep(void)
{
#if ENABLE_SIM2REAL_PROTOCOL
#if ENABLE_SIM2REAL_MOTOR_OUTPUT
    uint8_t status = Sim2Real_CurrentStatus();
    uint16_t tx_period_ms = SIM2REAL_MOTOR_TX_PERIOD_MS;
    sim2real_control_status_changed = (!sim2real_control_status_valid) || (status != sim2real_last_control_status);

    if ((status == SIM2REAL_MODE_READY) && !sim2real_motors_enabled)
    {
        tx_period_ms = SIM2REAL_ENABLE_TX_PERIOD_MS;
    }

    sim2real_motor_tx_tick++;
    if (sim2real_motor_tx_tick < tx_period_ms)
    {
        return;
    }
    sim2real_motor_tx_tick = 0U;

    switch (status)
    {
    case SIM2REAL_MODE_IDLE:
        Sim2Real_HandleIdleMode();
        break;
    case SIM2REAL_MODE_READY:
        Sim2Real_HandleReadyMode();
        break;
    case SIM2REAL_MODE_RUN:
        Sim2Real_HandleRunMode();
        break;
    case SIM2REAL_MODE_DAMPING:
        Sim2Real_HandleDampingMode();
        break;
    case SIM2REAL_MODE_ESTOP:
        Sim2Real_HandleEstopMode();
        break;
    case SIM2REAL_MODE_FAULT:
    default:
        Sim2Real_HandleFaultMode();
        break;
    }

    sim2real_last_control_status = status;
    sim2real_control_status_valid = true;
    sim2real_control_status_changed = false;
#else
    Sim2Real_DisableMotors();
#endif
#endif
}

static uint8_t Sim2Real_CurrentStatus(void)
{
    if (!sim2real_cmd_valid)
    {
        return SIM2REAL_MODE_IDLE;
    }

    if (Sim2Real_IsEstopHoldActive())
    {
        return SIM2REAL_MODE_ESTOP;
    }

    if (Sim2Real_GetFaultCode() & SIM2REAL_FAULT_CMD_TIMEOUT)
    {
        return SIM2REAL_MODE_DAMPING;
    }

    if (sim2real_last_cmd.mode <= SIM2REAL_MODE_ESTOP)
    {
        return sim2real_last_cmd.mode;
    }

    return SIM2REAL_MODE_FAULT;
}

uint16_t Sim2Real_GetFaultCode(void)
{
    uint16_t fault = sim2real_fault_latched;

    if (sim2real_cmd_valid)
    {
        uint32_t ttl_ms = (sim2real_last_cmd.ttl_ms == 0U) ? 50U : (uint32_t)sim2real_last_cmd.ttl_ms;
        if ((HAL_GetTick() - sim2real_last_cmd.rx_tick_ms) > ttl_ms)
        {
            fault |= SIM2REAL_FAULT_CMD_TIMEOUT;
        }
    }

#if (!ENABLE_TASK_CHASSIS_CONTROL) && (!ENABLE_SIM2REAL_MOTOR_OUTPUT)
    fault |= SIM2REAL_FAULT_CONTROL_DISABLED;
#endif

#if (!ENABLE_SIM2REAL_RUN_OUTPUT)
    if (sim2real_cmd_valid &&
        ((fault & SIM2REAL_FAULT_CMD_TIMEOUT) == 0U) &&
        (sim2real_last_cmd.mode == SIM2REAL_MODE_RUN))
    {
        fault |= SIM2REAL_FAULT_RUN_DISABLED;
    }
#endif

    if (sim2real_run_blocked_not_armed)
    {
        fault |= SIM2REAL_FAULT_MOTOR_NOT_ARMED;
    }

    return fault;
}

uint8_t Sim2Real_GetStatus(void)
{
    return Sim2Real_CurrentStatus();
}

static void Sim2Real_WriteStatePayload(uint8_t *payload)
{
    uint16_t off = 0U;
    uint16_t fault = Sim2Real_GetFaultCode();

    payload[off++] = Sim2Real_CurrentStatus();
    payload[off++] = 0U;
    Sim2Real_WriteBE16(payload + off, fault);
    off += 2U;
    Sim2Real_WriteBE32(payload + off, sim2real_cmd_valid ? sim2real_last_cmd.seq : 0U);
    off += 4U;

    for (uint8_t i = 0U; i < 3U; i++, off += 4U)
    {
        Sim2Real_WriteBEFloat(payload + off, INS.Gyro[i]);
    }
    for (uint8_t i = 0U; i < 4U; i++, off += 4U)
    {
        Sim2Real_WriteBEFloat(payload + off, INS.q[i]);
    }

    for (uint8_t i = 0U; i < SIM2REAL_LEG_NUM; i++, off += 4U)
    {
        Sim2Real_WriteBEFloat(payload + off, Sim2Real_LegMotorPositionToProtocol(i));
    }
    for (uint8_t i = 0U; i < SIM2REAL_LEG_NUM; i++, off += 4U)
    {
        Sim2Real_WriteBEFloat(payload + off, Sim2Real_LegMotorVelocityToProtocol(i));
    }
    for (uint8_t i = 0U; i < SIM2REAL_WHEEL_NUM; i++, off += 4U)
    {
        Sim2Real_WriteBEFloat(payload + off, Sim2Real_WheelVelocityToProtocol(i));
    }
    for (uint8_t i = 0U; i < SIM2REAL_LEG_NUM; i++, off += 4U)
    {
        Sim2Real_WriteBEFloat(payload + off, Sim2Real_LegMotorTorqueToProtocol(i));
    }
    for (uint8_t i = 0U; i < SIM2REAL_WHEEL_NUM; i++, off += 4U)
    {
        Sim2Real_WriteBEFloat(payload + off, sim2real_wheel_sign[i] * LK_9025_Motor[i].Data.Current * LK_TORQUE_COEFFICIENT);
    }
    for (uint8_t i = 0U; i < SIM2REAL_LEG_NUM; i++, off += 4U)
    {
        Sim2Real_WriteBEFloat(payload + off, 0.0f);
    }
    for (uint8_t i = 0U; i < SIM2REAL_WHEEL_NUM; i++, off += 4U)
    {
        Sim2Real_WriteBEFloat(payload + off, LK_9025_Motor[i].Data.Current);
    }

    Sim2Real_WriteBEFloat(payload + off, USER_ADC_Voltage_Update());
    off += 4U;

    for (uint8_t i = 0U; i < SIM2REAL_LEG_NUM; i++, off += 4U)
    {
        Sim2Real_WriteBEFloat(payload + off, Sim2Real_GetLegMotor(i)->Data.Temperature_Rotor);
    }
    for (uint8_t i = 0U; i < SIM2REAL_WHEEL_NUM; i++, off += 4U)
    {
        Sim2Real_WriteBEFloat(payload + off, LK_9025_Motor[i].Data.Temperature);
    }
}

void Sim2Real_SendState(void)
{
#if ENABLE_SIM2REAL_PROTOCOL
    uint8_t next_index = (uint8_t)(sim2real_state_tx_index ^ 1U);
    uint8_t *frame = sim2real_state_tx_buf[next_index];
    uint8_t *payload = frame + SIM2REAL_HEADER_LEN;

    Sim2Real_WriteBE16(frame, SIM2REAL_SOF);
    frame[2] = SIM2REAL_VER;
    frame[3] = SIM2REAL_MSG_STATE;
    Sim2Real_WriteBE16(frame + 4, SIM2REAL_STATE_PAYLOAD_LEN);
    Sim2Real_WriteBE32(frame + 6, sim2real_state_seq++);
    Sim2Real_WriteBE64(frame + 10, DWT_GetTimeline_us());

    Sim2Real_WriteStatePayload(payload);

    uint32_t crc = Sim2Real_Crc32(frame, SIM2REAL_HEADER_LEN + SIM2REAL_STATE_PAYLOAD_LEN);
    Sim2Real_WriteBE32(frame + SIM2REAL_HEADER_LEN + SIM2REAL_STATE_PAYLOAD_LEN, crc);

    if (CDC_Transmit_HS(frame, SIM2REAL_STATE_FRAME_LEN) == USBD_OK)
    {
        sim2real_state_tx_index = next_index;
    }
#endif
}

bool Sim2Real_GetLastCommand(Sim2Real_Command_t *cmd)
{
    if ((cmd == NULL) || !sim2real_cmd_valid)
    {
        return false;
    }

    *cmd = sim2real_last_cmd;
    return true;
}

uint32_t Sim2Real_GetValidCommandCount(void)
{
    return sim2real_valid_cmd_count;
}

uint32_t Sim2Real_GetCrcErrorCount(void)
{
    return sim2real_crc_error_count;
}

uint32_t Sim2Real_GetBadFrameCount(void)
{
    return sim2real_bad_frame_count;
}
