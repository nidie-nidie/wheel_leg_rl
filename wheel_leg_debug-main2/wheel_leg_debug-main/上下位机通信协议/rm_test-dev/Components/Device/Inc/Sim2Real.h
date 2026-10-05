#ifndef DEVICE_SIM2REAL_H
#define DEVICE_SIM2REAL_H

#include "stdbool.h"
#include "stdint.h"

/*
 * Sim2Real USB CDC protocol notes.
 *
 * Wire format is big-endian/network byte order. Do not memcpy C structs to the
 * USB frame directly, because C structs can contain padding and the STM32 is
 * little-endian. Sim2Real.c packs/unpacks every field explicitly.
 *
 * Frame header, 18 bytes:
 *   sof:uint16 = 0xAA55, ver:uint8 = 1, msg_type:uint8,
 *   payload_len:uint16, seq:uint32, t_us:uint64.
 * CRC32 is appended after payload and covers header + payload only.
 *
 * Command payload order:
 *   mode, reserved, ttl_ms,
 *   leg_q_des_rad[4]       = [jIO, jAG, jIJ, jAB],
 *   leg_dq_des_rad_s[4]    = [jIO, jAG, jIJ, jAB],
 *   wheel_dq_des_rad_s[2]  = [left wheel, right wheel],
 *   leg_kp[4], leg_kd[4], wheel_kp[2], wheel_kd[2],
 *   tau_limit_nm[6]        = [jIO, jAG, jIJ, jAB, left wheel, right wheel],
 *   dq_limit_rad_s[6]      = [jIO, jAG, jIJ, jAB, left wheel, right wheel].
 *
 * State payload order:
 *   status, reserved, fault_code, cmd_seq_echo,
 *   imu_gyro_rad_s[3], imu_quat_wxyz[4],
 *   active_leg_q_rad[4], active_leg_dq_rad_s[4],
 *   wheel_dq_rad_s[2],
 *   joint_tau_nm[6], motor_current_a[6], bus_voltage_v,
 *   temperature_c[6].
 *
 * Current physical mapping notes:
 *   wheel[0] -> LK_9025_Motor[0], CAN ID 4, left wheel.
 *   wheel[1] -> LK_9025_Motor[1], CAN ID 5, right wheel.
 *   leg[0] jIO -> left thigh, left-rear DM motor, DM_8009_Motor[1], CAN ID 2.
 *   leg[1] jAG -> right thigh, right-front DM motor, DM_8009_Motor[2], CAN ID 3.
 *   leg[2] jIJ -> left shank, left-front DM motor, DM_8009_Motor[0], CAN ID 1.
 *   leg[3] jAB -> right shank, right-rear DM motor, DM_8009_Motor[3], CAN ID 6.
 *   Joint sign and zero offset still need real-car verification before RUN
 *   motor output is enabled.
 */

#define SIM2REAL_SOF 0xAA55U
#define SIM2REAL_VER 1U

#define SIM2REAL_MSG_COMMAND 1U
#define SIM2REAL_MSG_STATE 2U

#define SIM2REAL_HEADER_LEN 18U
#define SIM2REAL_CRC_LEN 4U
#define SIM2REAL_CMD_PAYLOAD_LEN 140U
#define SIM2REAL_STATE_PAYLOAD_LEN 152U
#define SIM2REAL_CMD_FRAME_LEN (SIM2REAL_HEADER_LEN + SIM2REAL_CMD_PAYLOAD_LEN + SIM2REAL_CRC_LEN)
#define SIM2REAL_STATE_FRAME_LEN (SIM2REAL_HEADER_LEN + SIM2REAL_STATE_PAYLOAD_LEN + SIM2REAL_CRC_LEN)

#define SIM2REAL_LEG_NUM 4U
#define SIM2REAL_WHEEL_NUM 2U
#define SIM2REAL_ACTUATOR_NUM 6U

#define SIM2REAL_LEG_JIO_IDX 0U
#define SIM2REAL_LEG_JAG_IDX 1U
#define SIM2REAL_LEG_JIJ_IDX 2U
#define SIM2REAL_LEG_JAB_IDX 3U

#define SIM2REAL_LEG_JIO_DM_IDX 1U
#define SIM2REAL_LEG_JAG_DM_IDX 2U
#define SIM2REAL_LEG_JIJ_DM_IDX 0U
#define SIM2REAL_LEG_JAB_DM_IDX 3U

#define SIM2REAL_LEG_JIO_CAN_ID 2U
#define SIM2REAL_LEG_JAG_CAN_ID 3U
#define SIM2REAL_LEG_JIJ_CAN_ID 1U
#define SIM2REAL_LEG_JAB_CAN_ID 6U

#define SIM2REAL_WHEEL_LEFT_IDX 0U
#define SIM2REAL_WHEEL_RIGHT_IDX 1U
#define SIM2REAL_WHEEL_LEFT_CAN_ID 4U
#define SIM2REAL_WHEEL_RIGHT_CAN_ID 5U

typedef enum
{
    SIM2REAL_MODE_IDLE = 0,    /* Host is idle; firmware sends no active control. */
    SIM2REAL_MODE_READY = 1,   /* Host requests motor enable and damping hold. */
    SIM2REAL_MODE_RUN = 2,     /* Host requests actuator control output. */
    SIM2REAL_MODE_DAMPING = 3, /* Timeout or requested damping/safe output. */
    SIM2REAL_MODE_ESTOP = 4,   /* Emergency stop; highest priority, timed hold, then recoverable. */
    SIM2REAL_MODE_FAULT = 5,   /* Protocol or lower-level control fault. */
} Sim2Real_Mode_e;

typedef enum
{
    SIM2REAL_FAULT_NONE = 0x0000U,             /* No fault. */
    SIM2REAL_FAULT_CRC = 0x0001U,              /* RX frame CRC32 check failed. */
    SIM2REAL_FAULT_BAD_FRAME = 0x0002U,        /* RX frame header/type/length is invalid. */
    SIM2REAL_FAULT_RX_OVERFLOW = 0x0004U,      /* RX byte buffer overflowed before parsing. */
    SIM2REAL_FAULT_CMD_TIMEOUT = 0x0008U,      /* Last valid command exceeded ttl_ms. */
    SIM2REAL_FAULT_CONTROL_DISABLED = 0x0010U, /* No motor output path is enabled. */
    SIM2REAL_FAULT_RUN_DISABLED = 0x0020U,     /* RUN command accepted, but RUN output is gated off. */
    SIM2REAL_FAULT_MOTOR_NOT_ARMED = 0x0040U,  /* RUN command arrived before READY finished motor enable. */
} Sim2Real_Fault_e;

typedef struct
{
    uint8_t mode;                                  /* Sim2Real_Mode_e received from command payload. */
    uint8_t reserved;                              /* Reserved for future protocol extension. */
    uint16_t ttl_ms;                               /* Command timeout; 0 means use firmware default. */
    float leg_q_des_rad[SIM2REAL_LEG_NUM];         /* [jIO, jAG, jIJ, jAB], desired joint position. */
    float leg_dq_des_rad_s[SIM2REAL_LEG_NUM];      /* [jIO, jAG, jIJ, jAB], desired joint velocity. */
    float wheel_dq_des_rad_s[SIM2REAL_WHEEL_NUM];  /* [left wheel(ID4), right wheel(ID5)]. */
    float leg_kp[SIM2REAL_LEG_NUM];                /* [jIO, jAG, jIJ, jAB], leg position gain. */
    float leg_kd[SIM2REAL_LEG_NUM];                /* [jIO, jAG, jIJ, jAB], leg velocity gain. */
    float wheel_kp[SIM2REAL_WHEEL_NUM];            /* [left wheel(ID4), right wheel(ID5)]. */
    float wheel_kd[SIM2REAL_WHEEL_NUM];            /* [left wheel(ID4), right wheel(ID5)]. */
    float tau_limit_nm[SIM2REAL_ACTUATOR_NUM];     /* [jIO, jAG, jIJ, jAB, left wheel(ID4), right wheel(ID5)]. */
    float dq_limit_rad_s[SIM2REAL_ACTUATOR_NUM];   /* [jIO, jAG, jIJ, jAB, left wheel(ID4), right wheel(ID5)]. */
    uint32_t seq;                                  /* Header sequence number copied from the valid command. */
    uint64_t t_us;                                 /* Header timestamp copied from the valid command. */
    uint32_t rx_tick_ms;                           /* Local HAL_GetTick() when this command was accepted. */
} Sim2Real_Command_t;

void Sim2Real_RxBytes(const uint8_t *data, uint32_t len);      /* Feed raw USB CDC RX bytes into the frame parser. */
void Sim2Real_ControlStep(void);                               /* Run one 1 ms motor-output safety/control step. */
void Sim2Real_SendState(void);                                 /* Pack and send one state frame over USB CDC. */
bool Sim2Real_GetLastCommand(Sim2Real_Command_t *cmd);         /* Copy the latest valid command into cmd. */
uint32_t Sim2Real_GetValidCommandCount(void);                  /* Number of valid command frames accepted. */
uint32_t Sim2Real_GetCrcErrorCount(void);                      /* Number of command frames rejected by CRC32. */
uint32_t Sim2Real_GetBadFrameCount(void);                      /* Number of rejected malformed frames. */
uint16_t Sim2Real_GetFaultCode(void);                          /* Current Sim2Real_Fault_e bitmask. */
uint8_t Sim2Real_GetStatus(void);                              /* Current Sim2Real_Mode_e status returned in state. */

#endif
