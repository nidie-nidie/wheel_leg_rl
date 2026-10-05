#ifndef DAMIAOMOTORS_H
#define DAMIAOMOTORS_H

#include "bsp_fdcan.h"

#define MIT_MODE 0x000
#define POS_MODE 0x100
#define SPEED_MODE 0x200
//调试助手读到8009出厂时的参数
#define P_MIN -12.5f
#define P_MAX 12.5f
#define V_MIN -45.0f
#define V_MAX 45.0f
#define KP_MIN 0.0f
#define KP_MAX 500.0f
#define KD_MIN 0.0f
#define KD_MAX 5.0f
#define T_MIN -54.0f
#define T_MAX 54.0f

#ifndef uint8_t
typedef unsigned char uint8_t;
#endif // uint8_t
#ifndef uint16_t
typedef unsigned short uint16_t;
#endif // uint16_t
#ifndef uint32_t
typedef unsigned int uint32_t;
#endif // uint32_t

typedef struct
{
    int id;
    int state;
    int p_int;
    int v_int;
    int t_int;
    int kp_int;
    int kd_int;
    float pos;
    float vel;
    float tor;
    float Kp;
    float Kd;
    float Tmos;
    float Tcoil;
} motor_fbpara_t;

typedef struct
{
    char mode;
    float pos_set;
    float vel_set;
    float tor_set;
    float kp_set;
    float kd_set;
} motor_ctrl_t;

typedef struct
{
    char id;
    unsigned char start_flag;
	  uint8_t timeout_cnt;//失联计数
    motor_fbpara_t para;
    motor_ctrl_t cmd;
} dmmotor_t;

void enable_motor_mode(FDCAN_HandleTypeDef *hcan, uint16_t motor_id, uint16_t mode_id);
void disable_motor_mode(FDCAN_HandleTypeDef *hcan, uint16_t motor_id, uint16_t mode_id);
void save_pos_zero(FDCAN_HandleTypeDef *hcan, uint16_t motor_id, uint16_t mode_id);
void clear_err(FDCAN_HandleTypeDef *hcan, uint16_t motor_id, uint16_t mode_id);
void mit_ctrl(FDCAN_HandleTypeDef *hcan, uint16_t motor_id, float pos, float vel, float kp, float kd, float torq);
void pos_speed_ctrl(FDCAN_HandleTypeDef *hcan, uint16_t motor_id, float pos, float vel);
void speed_ctrl(FDCAN_HandleTypeDef *hcan, uint16_t motor_id, float vel);
void dm_fdkdata(dmmotor_t *motor, uint8_t *rx_data);
extern dmmotor_t max_motor,min_motor,finesse_motor,pitch_motor;
#endif // DAMIAOMOTORS_H
