#include "sim_adapter.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rm_third_party/glfw.h"
#include "rm_third_party/mujoco.h"

#include "Chassis_Task.h"
#include "jump_telemetry.h"
#include "Motor.h"
#include "robot_param.h"
#include "steer_state_machine.h"

typedef struct
{
    int id;
    int qpos;
    int qvel;
} JointRef;

typedef struct
{
    JointRef joint[4];
    JointRef wheel[2];
    int actuator[6];
    int base_body;
    int base_freejoint;
    int floor_geom;
    int base_contact_geom;
    int rotate_control_frame;
} ModelMap;

typedef enum
{
    DRIVE_STAND = 0,
    DRIVE_FORWARD = 1,
    DRIVE_JUMP = 2,
} DriveMode;

typedef struct
{
    DriveMode mode;
    float leg_set;
    float roll_set;
    float forward_speed;
    float current_speed;
    float target_x;
    float target_y;
    float position_hold_blend;
    float yaw_hold;
    float yaw_rate_ref;
    int yaw_lock;
    int planar_hold;
    int hold_position_pending;
    int hold_yaw_pending;
} DriveCommand;

typedef struct
{
    double samples[4];
} SupportForceObserver;

static mjModel *g_model = 0;
static mjData *g_data = 0;
static mjvCamera g_camera;
static mjvOption g_option;
static mjvScene g_scene;
static mjrContext g_context;
static int g_button_left = 0;
static int g_button_middle = 0;
static int g_button_right = 0;
static int g_paused = 0;
static int g_key_forward = 0;
static int g_key_backward = 0;
static int g_key_leg_up = 0;
static int g_key_leg_down = 0;
static int g_key_turn_left = 0;
static int g_key_turn_right = 0;
static float g_keyboard_steer_input = 0.0f;
static double g_last_x = 0.0;
static double g_last_y = 0.0;
static double g_body_z_min = 1.0e9;
static double g_body_z_max = -1.0e9;
static int g_body_z_range_valid = 0;
static double g_wheel_clearance_min = 1.0e9;
static double g_wheel_clearance_max = -1.0e9;
static double g_airborne_time = 0.0;
static double g_airborne_current_duration = 0.0;
static double g_airborne_max_duration = 0.0;
static float g_airborne_abs_roll_max = 0.0f;
static float g_airborne_abs_pitch_max = 0.0f;
static float g_airborne_abs_yaw_error_max = 0.0f;
static float g_airborne_rpy_last_error[3];
static int g_airborne_metrics_valid = 0;
static float g_xml_initial_rpy[3];
static mjtNum *g_xml_initial_qpos = 0;
static double g_airborne_full_pose_best_rms = 1.0e9;
static double g_airborne_full_pose_last_rms = 0.0;
static double g_airborne_full_pose_max_abs = 0.0;
static float g_airborne_pose_target[4];
static float g_airborne_joint_abs_max[4];
static float g_airborne_joint_last_error[4];
static float g_airborne_joint_best_error[4];
static double g_airborne_joint_error_sq_sum = 0.0;
static double g_airborne_joint_best_rms = 1.0e9;
static int g_airborne_joint_error_samples = 0;
static double g_jump_attitude_time = 0.0;
static double g_jump_pitch_min_time = 0.0;
static double g_jump_pitch_max_time = 0.0;
static float g_jump_abs_roll_max = 0.0f;
static float g_jump_abs_pitch_max = 0.0f;
static float g_jump_pitch_min = 0.0f;
static float g_jump_pitch_max = 0.0f;
static int g_jump_attitude_active = 0;
static int g_jump_attitude_valid = 0;
static double g_jump_base_contact_peak_n = 0.0;
static double g_jump_base_contact_peak_time = 0.0;
static mjtNum g_joint_ctrl_sign[4] = {1.0, 1.0, 1.0, 1.0};
static mjtNum g_wheel_ctrl_sign[2] = {1.0, 1.0};
static int g_use_wheel_balance_override = 1;
static float g_balance_pitch_target = MUJOCO_WHEEL_BALANCE_PITCH_TARGET;
static float g_balance_pitch_kp = MUJOCO_WHEEL_BALANCE_PITCH_KP;
static float g_balance_pitch_kd = MUJOCO_WHEEL_BALANCE_PITCH_KD;
static float g_balance_pos_kp = MUJOCO_WHEEL_BALANCE_POS_KP;
static float g_balance_vel_kd = MUJOCO_WHEEL_BALANCE_VEL_KD;
static float g_balance_pos_ramp_time = 0.5f;
static float g_balance_drive_kff = MUJOCO_WHEEL_BALANCE_DRIVE_KFF;
static double g_auto_stop_time = -1.0;
static double g_auto_jump_time = -1.0;
static float g_jump_thrust_ff = MUJOCO_JUMP_THRUST_FF;
static float g_jump_pitch_wheel_kp = MUJOCO_JUMP_PITCH_WHEEL_KP;
static float g_jump_pitch_wheel_kd = MUJOCO_JUMP_PITCH_WHEEL_KD;
static float g_jump_pitch_wheel_limit = MUJOCO_JUMP_PITCH_WHEEL_LIMIT;
static float g_jump_motor_wheel_limit = LK_MAX_MF_TORQUE;
static float g_jump_takeoff_yaw_kp = MUJOCO_JUMP_TAKEOFF_YAW_KP;
static float g_jump_takeoff_yaw_kd = MUJOCO_JUMP_TAKEOFF_YAW_KD;
static float g_jump_takeoff_yaw_limit = MUJOCO_JUMP_TAKEOFF_YAW_LIMIT;
static float g_jump_landing_yaw_kp = MUJOCO_JUMP_LANDING_YAW_KP;
static float g_jump_landing_yaw_kd = MUJOCO_JUMP_LANDING_YAW_KD;
static float g_jump_landing_yaw_limit = MUJOCO_JUMP_LANDING_YAW_LIMIT;
static int g_jump_landing_yaw_start_phase = MUJOCO_JUMP_LANDING_YAW_START_PHASE;
static float g_jump_pitch_target = MUJOCO_JUMP_PITCH_TARGET;
static float g_jump_pitch_offset = MUJOCO_JUMP_PITCH_OFFSET;
static int g_jump_pitch_target_overridden = 0;
static float g_jump_landing_pitch_target = MUJOCO_JUMP_PITCH_TARGET;
static float g_jump_landing_pitch_offset = MUJOCO_JUMP_LANDING_PITCH_OFFSET;
static int g_jump_landing_pitch_target_overridden = 0;
static float g_jump_recover_pitch_target = MUJOCO_JUMP_RECOVER_PITCH_TARGET;
static float g_jump_pitch_tp_kp = MUJOCO_JUMP_PITCH_TP_KP;
static float g_jump_pitch_tp_kd = MUJOCO_JUMP_PITCH_TP_KD;
static float g_jump_pitch_tp_limit = MUJOCO_JUMP_PITCH_TP_LIMIT;
static int g_jump_compression_enabled = 1;
static float g_jump_compress_target = MUJOCO_JUMP_COMPRESS_TARGET;
static float g_jump_compress_rate = MUJOCO_JUMP_COMPRESS_RATE;
static float g_jump_compress_support_scale = MUJOCO_JUMP_COMPRESS_SUPPORT_SCALE;
static float g_jump_compress_tolerance = MUJOCO_JUMP_COMPRESS_TOLERANCE;
static float g_jump_compress_hold_time = MUJOCO_JUMP_COMPRESS_HOLD_TIME;
static float g_jump_compress_timeout = MUJOCO_JUMP_COMPRESS_TIMEOUT;
static float g_jump_leg_swing_offset = MUJOCO_JUMP_LEG_SWING_OFFSET;
static float g_jump_leg_swing_kp = MUJOCO_JUMP_LEG_SWING_KP;
static float g_jump_leg_swing_kd = MUJOCO_JUMP_LEG_SWING_KD;
static float g_jump_leg_swing_limit = MUJOCO_JUMP_LEG_SWING_LIMIT;
static float g_jump_extend_l0 = MUJOCO_JUMP_EXTEND_L0;
static float g_jump_extend_end_margin = MUJOCO_JUMP_EXTEND_END_MARGIN;
static float g_jump_extend_rate = MUJOCO_JUMP_EXTEND_RATE;
static float g_jump_preland_clearance = MUJOCO_JUMP_PRELAND_CLEARANCE;
static float g_jump_preland_l0 = MUJOCO_JUMP_PRELAND_L0;
static float g_jump_preland_rate = MUJOCO_JUMP_PRELAND_RATE;
static float g_jump_buffer_l0 = MUJOCO_JUMP_BUFFER_L0;
static float g_jump_buffer_rate = MUJOCO_JUMP_BUFFER_RATE;
static float g_jump_buffer_support_scale = MUJOCO_JUMP_BUFFER_SUPPORT_SCALE;
static float g_jump_buffer_pid_scale = MUJOCO_JUMP_BUFFER_PID_SCALE;
static float g_jump_preland_pid_scale = MUJOCO_JUMP_PRELAND_PID_SCALE;
static float g_jump_landing_roll_f0_kp = MUJOCO_JUMP_LANDING_ROLL_F0_KP;
static float g_jump_landing_roll_f0_kd = MUJOCO_JUMP_LANDING_ROLL_F0_KD;
static float g_jump_landing_contact_f0_kp = MUJOCO_JUMP_LANDING_CONTACT_F0_KP;
static float g_jump_landing_balance_f0_limit = MUJOCO_JUMP_LANDING_BALANCE_F0_LIMIT;
static float g_jump_landing_roll_l0_kp = MUJOCO_JUMP_LANDING_ROLL_L0_KP;
static float g_jump_landing_roll_l0_kd = MUJOCO_JUMP_LANDING_ROLL_L0_KD;
static float g_jump_landing_balance_l0_limit = MUJOCO_JUMP_LANDING_BALANCE_L0_LIMIT;
static float g_jump_landing_clearance_l0_kp = MUJOCO_JUMP_LANDING_CLEARANCE_L0_KP;
static float g_jump_landing_clearance_l0_rate = MUJOCO_JUMP_LANDING_CLEARANCE_L0_RATE;
static float g_jump_landing_clearance_l0_limit = MUJOCO_JUMP_LANDING_CLEARANCE_L0_LIMIT;
static float g_jump_wheel_recover_blend_time = MUJOCO_JUMP_WHEEL_RECOVER_BLEND_TIME;
static float g_jump_lqr_tp_weight = MUJOCO_JUMP_LQR_TP_WEIGHT;
static float g_jump_split_tp_weight = MUJOCO_JUMP_SPLIT_TP_WEIGHT;
static float g_jump_pitch_tp_weight = MUJOCO_JUMP_PITCH_TP_WEIGHT;
static float g_jump_leg_swing_tp_weight = MUJOCO_JUMP_LEG_SWING_TP_WEIGHT;
static float g_jump_tuck_kp = MUJOCO_JUMP_TUCK_KP;
static float g_jump_tuck_kd = MUJOCO_JUMP_TUCK_KD;
static float g_jump_tuck_torque_limit = MUJOCO_JUMP_TUCK_TORQUE_LIMIT;
static float g_balance_yaw_kp = MUJOCO_WHEEL_BALANCE_YAW_KP;
static float g_balance_yaw_kd = MUJOCO_WHEEL_BALANCE_YAW_KD;
static float g_balance_wheel_limit = MUJOCO_WHEEL_BALANCE_LIMIT;
static float g_steer_pitch_target = -0.12f;
static float g_steer_yaw_kp = 3.0f;
static float g_steer_yaw_rate_kd = 3.0f;
static float g_steer_yaw_torque_limit = 1.45f;
static float g_keyboard_leg_rate = 0.04f;
static float g_initial_leg_length = INIT_LEG_LENGTH;
static SteerStateMachine g_steer_machine;
static int g_steer_command_was_active = 0;
static int g_steer_pose_hold_active = 0;
static SteerStateMachineConfig g_steer_config = {
    .turn_leg_length = 0.20f,
    .leg_rate = 0.30f,
    .yaw_rate_max = 0.50f,
    .yaw_accel_limit = 1.00f,
    .active_time_limit = 0.0f,
    .active_planar_error_limit = 0.60f,
    .input_deadband = 0.05f,
    .prepare_l0_tolerance = 0.012f,
    .brake_gyro_tolerance = 0.08f,
    .brake_linear_velocity_tolerance = 0.04f,
    .brake_hold_time = 0.12f,
    .recover_linear_velocity_tolerance = 0.05f,
    .recover_l0_tolerance = 0.002f,
    .abort_roll_limit = 0.30f,
    .abort_pitch_limit = 0.35f,
};
static double g_auto_steer_time = -1.0;
static double g_auto_steer_duration = 0.0;
static float g_auto_steer_input = 1.0f;
static int g_steer_metrics_valid = 0;
static double g_steer_metrics_time = 0.0;
static float g_steer_abs_roll_max = 0.0f;
static float g_steer_abs_pitch_max = 0.0f;
static float g_steer_abs_yaw_rate_max = 0.0f;
static float g_steer_planar_error_max = 0.0f;
static float g_steer_planar_speed_max = 0.0f;
static float g_steer_yaw_start = 0.0f;
static float g_steer_yaw_end = 0.0f;
static JumpTelemetry *g_jump_telemetry = 0;
static FILE *g_control_csv = 0;
static double g_control_csv_period_s = 0.010;
static double g_control_csv_next_time = 0.0;
static int g_controller_period_steps = 3;
static float g_controller_dt_s = 0.003f;
static SimControllerOutput g_held_controller_output;
static SimControllerControlBreakdown g_held_control_breakdown;
static double g_held_requested_torque[6];
static int g_held_controller_output_valid = 0;
static SupportForceObserver g_support_force_observer[2];
static float g_visual_forward_to_controller_sign = 1.0f;
static DriveCommand g_drive_command = {
    .mode = DRIVE_STAND,
    .leg_set = INIT_LEG_LENGTH,
    .roll_set = INIT_ROLL,
    .forward_speed = 0.2f,
    .current_speed = 0.0f,
    .target_x = 0.0f,
    .target_y = 0.0f,
    .position_hold_blend = 1.0f,
    .yaw_hold = 0.0f,
    .yaw_rate_ref = 0.0f,
    .yaw_lock = 1,
    .planar_hold = 0,
    .hold_position_pending = 1,
    .hold_yaw_pending = 1,
};






static int find_required_id(const mjModel *m, int type, const char *name)
{
    int id = mj_name2id(m, type, name);
    if (id < 0)
    {
        fprintf(stderr, "Missing MuJoCo object: %s\n", name);
        exit(2);
    }
    return id;
}








static int find_optional_id(const mjModel *m, int type, const char *name)
{
    return mj_name2id(m, type, name);
}

static int file_exists(const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (fp != 0)
    {
        fclose(fp);
        return 1;
    }
    return 0;
}

static const char *resolve_default_model_path(const char *model_path)
{
    static char prefixed_path[1024];

    if (model_path == 0 || model_path[0] == '\0' || file_exists(model_path))
    {
        return model_path;
    }

    if (model_path[0] != '/')
    {
        int written = snprintf(prefixed_path, sizeof(prefixed_path), "mujoco_control_extract/%s", model_path);
        if (written > 0 && written < (int)sizeof(prefixed_path) && file_exists(prefixed_path))
        {
            return prefixed_path;
        }
    }

    return model_path;
}

static void rotate_xy_into_controller_frame(double in_x, double in_y, float *out_x, float *out_y)
{
    if (out_x != 0)
    {
        *out_x = (float)(-in_y);
    }
    if (out_y != 0)
    {
        *out_y = (float)in_x;
    }
}

static void rotate_vec3_into_controller_frame(const mjtNum in[3], float out[3])
{
    out[0] = (float)(-in[1]);
    out[1] = (float)in[0];
    out[2] = (float)in[2];
}

static void rotate_quat_into_controller_frame(const mjtNum in[4], mjtNum out[4])
{
    const mjtNum c = 0.7071067811865476;
    const mjtNum z90[4] = {c, 0.0, 0.0, -c};

    out[0] = in[0] * z90[0] - in[1] * z90[1] - in[2] * z90[2] - in[3] * z90[3];
    out[1] = in[0] * z90[1] + in[1] * z90[0] + in[2] * z90[3] - in[3] * z90[2];
    out[2] = in[0] * z90[2] - in[1] * z90[3] + in[2] * z90[0] + in[3] * z90[1];
    out[3] = in[0] * z90[3] + in[1] * z90[2] - in[2] * z90[1] + in[3] * z90[0];
}




static JointRef find_joint(const mjModel *m, const char *name)
{
    int id = find_required_id(m, mjOBJ_JOINT, name);
    JointRef ref = {id, m->jnt_qposadr[id], m->jnt_dofadr[id]};
    return ref;
}

static double clamp_double(double value, double min_value, double max_value)
{
    if (value < min_value)
    {
        return min_value;
    }
    if (value > max_value)
    {
        return max_value;
    }
    return value;
}

static const char *drive_command_name(const DriveCommand *command)
{
    if (command->mode == DRIVE_JUMP)
    {
        return "jump";
    }
    if (command->mode == DRIVE_FORWARD)
    {
        return command->forward_speed * g_visual_forward_to_controller_sign < 0.0f ? "backward" : "forward";
    }
    return "stand";
}

static float visual_to_controller_speed(float visual_speed)
{
    return g_visual_forward_to_controller_sign * visual_speed;
}

static float controller_to_visual_speed(float controller_speed)
{
    return g_visual_forward_to_controller_sign * controller_speed;
}

static float drive_speed_magnitude(const DriveCommand *command)
{
    const float speed = fabsf(command->forward_speed);
    return speed > 1.0e-4f ? speed : 0.2f;
}

static void queue_drive_mode(DriveCommand *command, DriveMode mode)
{
    if (command->mode != mode)
    {
        const DriveMode previous_mode = command->mode;
        command->mode = mode;
        command->hold_position_pending = 1;
        if (mode == DRIVE_FORWARD || previous_mode == DRIVE_FORWARD || mode == DRIVE_JUMP || previous_mode == DRIVE_JUMP)
        {
            command->position_hold_blend = 0.0f;
        }
    }
    command->hold_yaw_pending = 1;
}

static void request_drive_speed(DriveCommand *command, float desired_speed)
{
    if (fabsf(desired_speed) < 1.0e-6f)
    {
        command->forward_speed = drive_speed_magnitude(command);
        queue_drive_mode(command, DRIVE_STAND);
        return;
    }

    if (command->mode != DRIVE_FORWARD || command->forward_speed * desired_speed <= 0.0f)
    {
        command->hold_position_pending = 1;
    }

    queue_drive_mode(command, DRIVE_FORWARD);
    command->forward_speed = desired_speed;
}

static void request_jump(DriveCommand *command, double event_time)
{
    if (SimController_RequestJump())
    {
        g_jump_attitude_time = 0.0;
        g_jump_pitch_min_time = event_time;
        g_jump_pitch_max_time = event_time;
        g_jump_abs_roll_max = 0.0f;
        g_jump_abs_pitch_max = 0.0f;
        g_jump_pitch_min = 0.0f;
        g_jump_pitch_max = 0.0f;
        g_jump_base_contact_peak_n = 0.0;
        g_jump_base_contact_peak_time = event_time;
        g_jump_attitude_active = 1;
        g_jump_attitude_valid = 1;
        queue_drive_mode(command, DRIVE_JUMP);
        printf("Drive mode -> jump at t=%6.3f\n", event_time);
        fflush(stdout);
    }
}

static void sync_keyboard_drive_command(GLFWwindow *window, double dt)
{
    static int jump_was_down = 0;
    const int forward_down = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS ||
                             glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS;
    const int backward_down = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
    const int leg_up_down = glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS;
    const int leg_down_down = glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS;
    const int turn_left_down = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
    const int turn_right_down = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;
    const int jump_down = glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS;
    const char *before = drive_command_name(&g_drive_command);
    float desired_speed = 0.0f;

    g_key_forward = forward_down;
    g_key_backward = backward_down;
    g_key_leg_up = leg_up_down;
    g_key_leg_down = leg_down_down;
    g_key_turn_left = turn_left_down;
    g_key_turn_right = turn_right_down;
    g_keyboard_steer_input = 0.0f;
    if (turn_left_down && !turn_right_down)
    {
        g_keyboard_steer_input = 1.0f;
    }
    else if (turn_right_down && !turn_left_down)
    {
        g_keyboard_steer_input = -1.0f;
    }

    if (jump_down && !jump_was_down)
    {
        request_jump(&g_drive_command, g_data ? g_data->time : 0.0);
    }
    jump_was_down = jump_down;

    if (g_drive_command.mode == DRIVE_JUMP)
    {
        desired_speed = 0.0f;
    }
    else if (forward_down && !backward_down)
    {
        desired_speed = visual_to_controller_speed(drive_speed_magnitude(&g_drive_command));
    }
    else if (backward_down && !forward_down)
    {
        desired_speed = visual_to_controller_speed(-drive_speed_magnitude(&g_drive_command));
    }

    if (g_drive_command.mode != DRIVE_JUMP)
    {
        request_drive_speed(&g_drive_command, desired_speed);
    }
    if (leg_up_down && !leg_down_down)
    {
        g_drive_command.leg_set += (float)dt * g_keyboard_leg_rate;
    }
    else if (leg_down_down && !leg_up_down)
    {
        g_drive_command.leg_set -= (float)dt * g_keyboard_leg_rate;
    }
    g_drive_command.leg_set = (float)clamp_double(g_drive_command.leg_set,
                                                  MIN_LEG_LENGTH,
                                                  MAX_LEG_LENGTH);

    if (strcmp(before, drive_command_name(&g_drive_command)) != 0)
    {
        printf("Drive mode -> %s at t=%6.3f\n",
               drive_command_name(&g_drive_command),
               g_data ? g_data->time : 0.0);
        fflush(stdout);
    }
}




static void quat_to_euler(const mjtNum q[4], float *roll, float *pitch, float *yaw)
{
    double w = q[0];
    double x = q[1];
    double y = q[2];
    double z = q[3];

    double sinr_cosp = 2.0 * (w * x + y * z);
    double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    double sinp = 2.0 * (w * y - z * x);
    double siny_cosp = 2.0 * (w * z + x * y);
    double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);

    if (sinp > 1.0)
    {
        sinp = 1.0;
    }
    else if (sinp < -1.0)
    {
        sinp = -1.0;
    }

    *roll = (float)atan2(sinr_cosp, cosr_cosp);
    *pitch = (float)asin(sinp);
    *yaw = (float)atan2(siny_cosp, cosy_cosp);
}




static void build_model_map(const mjModel *m, ModelMap *map)
{
    memset(map, 0, sizeof(*map));
    map->floor_geom = find_optional_id(m, mjOBJ_GEOM, "floor");
    map->base_contact_geom = find_optional_id(m, mjOBJ_GEOM, "base_proxy");
    map->rotate_control_frame = 0;

    if (find_optional_id(m, mjOBJ_BODY, "base_link") >= 0)
    {
        map->base_body = find_required_id(m, mjOBJ_BODY, "base_link");
        map->base_freejoint = find_required_id(m, mjOBJ_JOINT, "base_freejoint");

        map->joint[0] = find_joint(m, "left_leg");
        map->joint[1] = find_joint(m, "left_small_leg");
        map->joint[2] = find_joint(m, "right_leg");
        map->joint[3] = find_joint(m, "right_small_leg");
        map->wheel[0] = find_joint(m, "left_wheel");
        map->wheel[1] = find_joint(m, "right_wheel");

        map->actuator[0] = find_required_id(m, mjOBJ_ACTUATOR, "left_leg_motor");
        map->actuator[1] = find_required_id(m, mjOBJ_ACTUATOR, "left_small_leg_motor");
        map->actuator[2] = find_required_id(m, mjOBJ_ACTUATOR, "right_leg_motor");
        map->actuator[3] = find_required_id(m, mjOBJ_ACTUATOR, "right_small_leg_motor");
        map->actuator[4] = find_required_id(m, mjOBJ_ACTUATOR, "left_wheel_motor");
        map->actuator[5] = find_required_id(m, mjOBJ_ACTUATOR, "right_wheel_motor");
        return;
    }




    if (find_optional_id(m, mjOBJ_BODY, "base") >= 0)
    {
        map->base_body = find_required_id(m, mjOBJ_BODY, "base");
        map->base_freejoint = find_required_id(m, mjOBJ_JOINT, "base_free");
        map->rotate_control_frame = 1;

        // joint[0]=jIJ 左前/J0/phi4, joint[1]=jIO 左后/J1/phi1
        // joint[2]=jAG 右后/J2/phi1, joint[3]=jAB 右前/J3/phi4
        map->joint[0] = find_joint(m, "jIJ");
        map->joint[1] = find_joint(m, "jIO");
        map->joint[2] = find_joint(m, "jAG");
        map->joint[3] = find_joint(m, "jAB");
        map->wheel[0] = find_joint(m, "jwheel_left");
        map->wheel[1] = find_joint(m, "jwheel_right");

        map->actuator[0] = find_required_id(m, mjOBJ_ACTUATOR, "Left_front_joint_act");
        map->actuator[1] = find_required_id(m, mjOBJ_ACTUATOR, "Left_rear_joint_act");
        map->actuator[2] = find_required_id(m, mjOBJ_ACTUATOR, "Right_rear_joint_act");
        map->actuator[3] = find_required_id(m, mjOBJ_ACTUATOR, "Right_front_joint_act");
        map->actuator[4] = find_required_id(m, mjOBJ_ACTUATOR, "Left_Wheel_act");
        map->actuator[5] = find_required_id(m, mjOBJ_ACTUATOR, "Right_Wheel_act");
        return;
    }

    fprintf(stderr, "Unsupported MuJoCo model: expected body 'base_link' or 'base'.\n");
    exit(2);
}

static int is_closed_chain_model(const mjModel *m)
{
    return find_optional_id(m, mjOBJ_JOINT, "jBE") >= 0 &&
           find_optional_id(m, mjOBJ_JOINT, "jJM") >= 0;
}

static double wrap_pi(double value)
{
    while (value > M_PI)
    {
        value -= 2.0 * M_PI;
    }
    while (value < -M_PI)
    {
        value += 2.0 * M_PI;
    }
    return value;
}

static SteerDriveMode steer_drive_mode_from_drive(DriveMode mode)
{
    switch (mode)
    {
    case DRIVE_FORWARD:
        return STEER_DRIVE_FORWARD;
    case DRIVE_JUMP:
        return STEER_DRIVE_JUMP;
    case DRIVE_STAND:
    default:
        return STEER_DRIVE_STAND;
    }
}

static float scheduled_steer_input(double time)
{
    if (g_auto_steer_time < 0.0 || g_auto_steer_duration <= 0.0)
    {
        return 0.0f;
    }
    if (time < g_auto_steer_time ||
        time >= g_auto_steer_time + g_auto_steer_duration)
    {
        return 0.0f;
    }
    return (float)clamp_double(g_auto_steer_input, -1.0, 1.0);
}

static float current_steer_input(double time)
{
    if (fabsf(g_keyboard_steer_input) > 1.0e-5f)
    {
        return g_keyboard_steer_input;
    }
    return scheduled_steer_input(time);
}

static void update_steer_metrics(const SteerCommand *command,
                                 const SimControllerState *state,
                                 double dt)
{
    if (command == 0 || state == 0)
    {
        return;
    }

    if (!command->active)
    {
        return;
    }

    if (!g_steer_metrics_valid)
    {
        g_steer_metrics_valid = 1;
        g_steer_metrics_time = 0.0;
        g_steer_abs_roll_max = 0.0f;
        g_steer_abs_pitch_max = 0.0f;
        g_steer_abs_yaw_rate_max = 0.0f;
        g_steer_planar_error_max = 0.0f;
        g_steer_planar_speed_max = 0.0f;
        g_steer_yaw_start = state->yaw;
    }

    const float planar_error_x = state->body_x - command->target_x;
    const float planar_error_y = state->body_y - command->target_y;
    const float planar_error =
        sqrtf(planar_error_x * planar_error_x +
              planar_error_y * planar_error_y);
    const float planar_speed =
        sqrtf(state->body_v * state->body_v +
              state->body_v_y * state->body_v_y);

    g_steer_metrics_time += dt;
    g_steer_yaw_end = state->yaw;
    g_steer_abs_roll_max =
        fmaxf(g_steer_abs_roll_max, fabsf(state->roll));
    g_steer_abs_pitch_max =
        fmaxf(g_steer_abs_pitch_max, fabsf(state->pitch));
    g_steer_abs_yaw_rate_max =
        fmaxf(g_steer_abs_yaw_rate_max, fabsf(state->gyro[2]));
    g_steer_planar_error_max =
        fmaxf(g_steer_planar_error_max, planar_error);
    g_steer_planar_speed_max =
        fmaxf(g_steer_planar_speed_max, planar_speed);
}

static double sim_theta_transform(double angle, double dangle, int direction)
{
    return wrap_pi((angle + dangle) * (double)direction);
}

static int calc_phi1_phi4(double phi0, double leg_length, double phi1_phi4[2])
{
    const double cos_beta1 = (LEG_L1 * LEG_L1 + leg_length * leg_length - LEG_L2 * LEG_L2) /
                             (2.0 * LEG_L1 * leg_length);
    const double cos_beta2 = (LEG_L4 * LEG_L4 + leg_length * leg_length - LEG_L3 * LEG_L3) /
                             (2.0 * LEG_L4 * leg_length);

    if (cos_beta1 < -1.0 || cos_beta1 > 1.0 || cos_beta2 < -1.0 || cos_beta2 > 1.0)
    {
        return 0;
    }

    phi1_phi4[0] = phi0 + acos(cos_beta1);
    phi1_phi4[1] = phi0 - acos(cos_beta2);
    return 1;
}

static void calc_initial_stand_joint_qpos(mjtNum joint_qpos[4])
{
    const double leg_length = g_initial_leg_length;
    const double phi0 = INIT_L0_PITCH;
    double phi1_phi4[2] = {0.0, 0.0};

    if (!calc_phi1_phi4(phi0, leg_length, phi1_phi4))
    {
        fprintf(stderr, "Failed to calculate initial stand pose.\n");
        exit(2);
    }

    joint_qpos[0] = -sim_theta_transform(phi1_phi4[1], -J0_ANGLE_OFFSET, J0_DIRECTION);
    joint_qpos[1] = -sim_theta_transform(phi1_phi4[0], -J1_ANGLE_OFFSET, J1_DIRECTION);
    joint_qpos[2] = -sim_theta_transform(phi1_phi4[0], -J2_ANGLE_OFFSET, J2_DIRECTION);
    joint_qpos[3] = -sim_theta_transform(phi1_phi4[1], -J3_ANGLE_OFFSET, J3_DIRECTION);
}

static void zero_fixed_velocities(const mjModel *m, mjData *d, const ModelMap *map)
{
    int base_qvel = m->jnt_dofadr[map->base_freejoint];
    for (int i = 0; i < 6; ++i)
    {
        d->qvel[base_qvel + i] = 0.0;
    }

    for (int i = 0; i < 4; ++i)
    {
        d->qvel[map->joint[i].qvel] = 0.0;
    }
}

static void relax_closed_chain_pose(mjModel *m, mjData *d, const ModelMap *map)
{
    if (!is_closed_chain_model(m))
    {
        return;
    }

    int base_qpos = m->jnt_qposadr[map->base_freejoint];
    mjtNum base_pose[7];
    mjtNum joint_pose[4];
    mjtNum gravity[3];
    mjtNum saved_time = d->time;

    for (int i = 0; i < 7; ++i)
    {
        base_pose[i] = d->qpos[base_qpos + i];
    }
    for (int i = 0; i < 4; ++i)
    {
        joint_pose[i] = d->qpos[map->joint[i].qpos];
    }
    for (int i = 0; i < 3; ++i)
    {
        gravity[i] = m->opt.gravity[i];
        m->opt.gravity[i] = 0.0;
    }

    for (int iter = 0; iter < 2000; ++iter)
    {
        for (int i = 0; i < 7; ++i)
        {
            d->qpos[base_qpos + i] = base_pose[i];
        }
        for (int i = 0; i < 4; ++i)
        {
            d->qpos[map->joint[i].qpos] = joint_pose[i];
        }
        zero_fixed_velocities(m, d, map);

        mj_step(m, d);

        for (int i = 0; i < m->nv; ++i)
        {
            d->qvel[i] *= 0.96;
        }
    }

    for (int i = 0; i < 7; ++i)
    {
        d->qpos[base_qpos + i] = base_pose[i];
    }
    for (int i = 0; i < 4; ++i)
    {
        d->qpos[map->joint[i].qpos] = joint_pose[i];
    }
    memset(d->qvel, 0, sizeof(mjtNum) * m->nv);
    memset(d->ctrl, 0, sizeof(mjtNum) * m->nu);
    d->time = saved_time;

    for (int i = 0; i < 3; ++i)
    {
        m->opt.gravity[i] = gravity[i];
    }
    mj_forward(m, d);
}

static void settle_closed_chain_pose(mjModel *m, mjData *d, const ModelMap *map, const mjtNum target_joint_pose[4])
{
    if (!is_closed_chain_model(m))
    {
        return;
    }

    const int base_qpos = m->jnt_qposadr[map->base_freejoint];
    mjtNum base_pose[7];
    mjtNum start_joint_pose[4];
    mjtNum gravity[3];
    const mjtNum saved_time = d->time;
    const int settle_steps = 3000;

    for (int i = 0; i < 7; ++i)
    {
        base_pose[i] = d->qpos[base_qpos + i];
    }
    for (int i = 0; i < 4; ++i)
    {
        start_joint_pose[i] = d->qpos[map->joint[i].qpos];
    }
    for (int i = 0; i < 3; ++i)
    {
        gravity[i] = m->opt.gravity[i];
        m->opt.gravity[i] = 0.0;
    }

    for (int iter = 0; iter < settle_steps; ++iter)
    {
        const mjtNum alpha = (mjtNum)(iter + 1) / (mjtNum)settle_steps;

        for (int i = 0; i < 7; ++i)
        {
            d->qpos[base_qpos + i] = base_pose[i];
        }
        for (int i = 0; i < 4; ++i)
        {
            d->qpos[map->joint[i].qpos] =
                (1.0 - alpha) * start_joint_pose[i] + alpha * target_joint_pose[i];
        }
        zero_fixed_velocities(m, d, map);

        mj_step(m, d);

        for (int i = 0; i < m->nv; ++i)
        {
            d->qvel[i] *= 0.94;
        }
    }

    for (int i = 0; i < 7; ++i)
    {
        d->qpos[base_qpos + i] = base_pose[i];
    }
    for (int i = 0; i < 4; ++i)
    {
        d->qpos[map->joint[i].qpos] = target_joint_pose[i];
    }
    memset(d->qvel, 0, sizeof(mjtNum) * m->nv);
    memset(d->ctrl, 0, sizeof(mjtNum) * m->nu);
    d->time = saved_time;

    for (int i = 0; i < 3; ++i)
    {
        m->opt.gravity[i] = gravity[i];
    }
    mj_forward(m, d);
}

static void apply_closed_chain_initial_pose(mjModel *m, mjData *d, const ModelMap *map, int ground_init)
{
    const int base_qpos = m->jnt_qposadr[map->base_freejoint];
    const int left_wheel_body = m->jnt_bodyid[map->wheel[0].id];
    const int right_wheel_body = m->jnt_bodyid[map->wheel[1].id];
    mjtNum joint_qpos[4];

    mj_resetData(m, d);

    const mjtNum base_x = d->qpos[base_qpos + 0];
    const mjtNum base_y = d->qpos[base_qpos + 1];
    calc_initial_stand_joint_qpos(joint_qpos);

    d->qpos[base_qpos + 0] = base_x;
    d->qpos[base_qpos + 1] = base_y;
    d->qpos[base_qpos + 2] = 1.0;
    d->qpos[base_qpos + 3] = 1.0;
    d->qpos[base_qpos + 4] = 0.0;
    d->qpos[base_qpos + 5] = 0.0;
    d->qpos[base_qpos + 6] = 0.0;

    for (int i = 0; i < 4; ++i)
    {
        d->qpos[map->joint[i].qpos] = joint_qpos[i];
    }
    for (int i = 0; i < 2; ++i)
    {
        d->qpos[map->wheel[i].qpos] = 0.0;
    }

    memset(d->qvel, 0, sizeof(mjtNum) * m->nv);
    memset(d->ctrl, 0, sizeof(mjtNum) * m->nu);
    mj_forward(m, d);
    settle_closed_chain_pose(m, d, map, joint_qpos);

    const double left_clearance = (double)d->xpos[3 * left_wheel_body + 2] - WHEEL_RADIUS;
    const double right_clearance = (double)d->xpos[3 * right_wheel_body + 2] - WHEEL_RADIUS;
    const double ground_margin = 0.001;
    d->qpos[base_qpos + 2] += -fmin(left_clearance, right_clearance) + ground_margin;
    if (!ground_init)
    {
        d->qpos[base_qpos + 2] += 0.5;
    }

    memset(d->qvel, 0, sizeof(mjtNum) * m->nv);
    memset(d->ctrl, 0, sizeof(mjtNum) * m->nu);
    d->time = 0.0;
    mj_forward(m, d);
}

static void apply_model_initial_pose(mjModel *m, mjData *d, const ModelMap *map, const char *init_key_name)
{
    const int init_key = init_key_name ? find_optional_id(m, mjOBJ_KEY, init_key_name) : -1;
    if (init_key >= 0)
    {
        mj_resetDataKeyframe(m, d, init_key);
    }
    else if (find_optional_id(m, mjOBJ_BODY, "base") >= 0 &&
             init_key_name != 0 &&
             (strcmp(init_key_name, "pos_debug_ground") == 0 ||
              strcmp(init_key_name, "pos_debug_hang") == 0))
    {
        apply_closed_chain_initial_pose(m, d, map, strcmp(init_key_name, "pos_debug_ground") == 0);
    }
    else
    {
        mj_resetData(m, d);
    }

    memset(d->qvel, 0, sizeof(mjtNum) * m->nv);
    memset(d->ctrl, 0, sizeof(mjtNum) * m->nu);
    mj_forward(m, d);
}




static void read_state(const mjModel *m, const mjData *d, const ModelMap *map, SimControllerState *state)
{
    memset(state, 0, sizeof(*state));

    for (int i = 0; i < 4; ++i)
    {
        state->joint_pos[i] = (float)d->qpos[map->joint[i].qpos];
        state->joint_vel[i] = (float)d->qvel[map->joint[i].qvel];
    }

    for (int i = 0; i < 2; ++i)
    {
        state->wheel_vel[i] = (float)(g_wheel_ctrl_sign[i] * d->qvel[map->wheel[i].qvel]);
    }

    int base_qpos = m->jnt_qposadr[map->base_freejoint];
    int base_qvel = m->jnt_dofadr[map->base_freejoint];
    if (map->rotate_control_frame)
    {
        mjtNum rotated_quat[4];
        mjtNum base_com_velocity[3];
        mjtNum base_com_jacobian[3 * m->nv];
        mjtNum raw_gyro[3] = {
            d->qvel[base_qvel + 3],
            d->qvel[base_qvel + 4],
            d->qvel[base_qvel + 5],
        };
        float rotated_gyro[3];
        float body_x = 0.0f;
        float body_y = 0.0f;
        float body_v = 0.0f;
        float body_v_y = 0.0f;

        rotate_quat_into_controller_frame(&d->qpos[base_qpos + 3], rotated_quat);
        quat_to_euler(rotated_quat, &state->roll, &state->pitch, &state->yaw);

        rotate_vec3_into_controller_frame(raw_gyro, rotated_gyro);
        state->gyro[0] = rotated_gyro[0];
        state->gyro[1] = rotated_gyro[1];
        state->gyro[2] = rotated_gyro[2];

        mj_jacBodyCom(m,
                      d,
                      base_com_jacobian,
                      0,
                      map->base_body);
        mju_mulMatVec(base_com_velocity,
                      base_com_jacobian,
                      d->qvel,
                      3,
                      m->nv);
        rotate_xy_into_controller_frame(d->xipos[3 * map->base_body + 0],
                                        d->xipos[3 * map->base_body + 1],
                                        &body_x,
                                        &body_y);
        state->body_x = body_x;
        state->body_y = body_y;
        state->body_z = (float)d->qpos[base_qpos + 2];
        state->body_z_vel = (float)d->qvel[base_qvel + 2];

        rotate_xy_into_controller_frame(base_com_velocity[0],
                                        base_com_velocity[1],
                                        &body_v,
                                        &body_v_y);
        state->body_v = body_v;
        state->body_v_y = body_v_y;
    }
    else
    {
        quat_to_euler(&d->qpos[base_qpos + 3], &state->roll, &state->pitch, &state->yaw);

        state->gyro[0] = (float)d->qvel[base_qvel + 3];
        state->gyro[1] = (float)d->qvel[base_qvel + 4];
        state->gyro[2] = (float)d->qvel[base_qvel + 5];
        state->body_x = (float)d->qpos[base_qpos + 0];
        state->body_y = (float)d->qpos[base_qpos + 1];
        state->body_z = (float)d->qpos[base_qpos + 2];
        state->body_z_vel = (float)d->qvel[base_qvel + 2];
        state->body_v = (float)d->qvel[base_qvel + 0];
        state->body_v_y = (float)d->qvel[base_qvel + 1];
    }
}

static void update_drive_command(DriveCommand *command, const SimControllerState *state, double dt)
{
    if (command->hold_position_pending)
    {
        command->target_x = state->body_x;
        command->target_y = state->body_y;
        command->hold_position_pending = 0;
    }
    if (command->hold_yaw_pending)
    {
        command->yaw_hold = state->yaw;
        command->hold_yaw_pending = 0;
    }

    const float target_speed = command->mode == DRIVE_FORWARD ? command->forward_speed : 0.0f;
    const float max_speed_delta = (float)(dt * 0.8);
    float speed_delta = target_speed - command->current_speed;
    speed_delta = (float)clamp_double(speed_delta, -max_speed_delta, max_speed_delta);
    command->current_speed += speed_delta;

    if (command->mode == DRIVE_STAND)
    {
        if (command->position_hold_blend < 1.0f)
        {
            const float blend_delta = g_balance_pos_ramp_time > 1.0e-4f
                                          ? (float)(dt / g_balance_pos_ramp_time)
                                          : 1.0f;
            command->position_hold_blend = (float)clamp_double(command->position_hold_blend + blend_delta,
                                                               0.0,
                                                               1.0);
            command->target_x = state->body_x +
                                command->position_hold_blend * (command->target_x - state->body_x);
            command->target_y = state->body_y +
                                command->position_hold_blend * (command->target_y - state->body_y);
        }
    }
    else
    {
        command->position_hold_blend = 0.0f;
    }
}

static void update_body_z_range(const SimControllerState *state)
{
    if (!g_body_z_range_valid)
    {
        g_body_z_min = state->body_z;
        g_body_z_max = state->body_z;
        g_body_z_range_valid = 1;
        return;
    }
    if (state->body_z < g_body_z_min)
    {
        g_body_z_min = state->body_z;
    }
    if (state->body_z > g_body_z_max)
    {
        g_body_z_max = state->body_z;
    }
}

static void measure_wheel_clearances(const mjModel *m,
                                     const mjData *d,
                                     const ModelMap *map,
                                     double clearance[2])
{
    const int left_wheel_body = m->jnt_bodyid[map->wheel[0].id];
    const int right_wheel_body = m->jnt_bodyid[map->wheel[1].id];
    clearance[0] = d->xpos[3 * left_wheel_body + 2] - WHEEL_RADIUS;
    clearance[1] = d->xpos[3 * right_wheel_body + 2] - WHEEL_RADIUS;
}

static void measure_wheel_clearance(const mjModel *m,
                                    const mjData *d,
                                    const ModelMap *map,
                                    double *min_clearance,
                                    double *max_clearance)
{
    double clearance[2];
    measure_wheel_clearances(m, d, map, clearance);

    *min_clearance = fmin(clearance[0], clearance[1]);
    *max_clearance = fmax(clearance[0], clearance[1]);
}

static int wheels_are_airborne(const mjModel *m, const mjData *d, const ModelMap *map)
{
    double min_clearance;
    double max_clearance;

    measure_wheel_clearance(m, d, map, &min_clearance, &max_clearance);
    (void)max_clearance;
    return min_clearance > 0.005;
}

static void measure_wheel_contact_normal_force(const mjModel *m,
                                               const mjData *d,
                                               const ModelMap *map,
                                               double force_n[2])
{
    const int wheel_body[2] = {
        m->jnt_bodyid[map->wheel[0].id],
        m->jnt_bodyid[map->wheel[1].id],
    };
    force_n[0] = 0.0;
    force_n[1] = 0.0;

    for (int contact_index = 0; contact_index < d->ncon; ++contact_index)
    {
        const mjContact *contact = &d->contact[contact_index];
        const int body1 = m->geom_bodyid[contact->geom1];
        const int body2 = m->geom_bodyid[contact->geom2];
        const int geom1_is_floor =
            contact->geom1 == map->floor_geom || body1 == 0;
        const int geom2_is_floor =
            contact->geom2 == map->floor_geom || body2 == 0;

        for (int wheel = 0; wheel < 2; ++wheel)
        {
            if (!((body1 == wheel_body[wheel] && geom2_is_floor) ||
                  (body2 == wheel_body[wheel] && geom1_is_floor)))
            {
                continue;
            }

            mjtNum contact_force[6] = {0};
            mj_contactForce(m, d, contact_index, contact_force);
            force_n[wheel] += fabs((double)contact_force[0]);
        }
    }
}

static double measure_base_contact_normal_force(const mjModel *m,
                                                const mjData *d,
                                                const ModelMap *map)
{
    if (map->base_contact_geom < 0 || map->floor_geom < 0)
    {
        return 0.0;
    }

    double force_n = 0.0;
    for (int contact_index = 0; contact_index < d->ncon; ++contact_index)
    {
        const mjContact *contact = &d->contact[contact_index];
        if (!((contact->geom1 == map->base_contact_geom &&
               contact->geom2 == map->floor_geom) ||
              (contact->geom2 == map->base_contact_geom &&
               contact->geom1 == map->floor_geom)))
        {
            continue;
        }

        mjtNum contact_force[6] = {0};
        mj_contactForce(m, d, contact_index, contact_force);
        force_n += fabs((double)contact_force[0]);
    }
    return force_n;
}

static double estimate_vmc_support_force(const vmc_leg_t *leg,
                                         double base_accel_z_mps2)
{
    const double theta = leg->theta;
    const double sin_theta = sin(theta);
    const double cos_theta = cos(theta);

    return leg->F0 * cos_theta +
           leg->Tp * sin_theta / leg->L0 +
           0.6 *
               (base_accel_z_mps2 -
                leg->dd_L0 * cos_theta +
                2.0 * leg->d_L0 * leg->d_theta * sin_theta +
                leg->L0 * leg->dd_theta * sin_theta +
                leg->L0 * leg->d_theta * leg->d_theta * cos_theta);
}

static double update_support_force_observer(SupportForceObserver *observer,
                                            double raw_force_n)
{
    observer->samples[0] = observer->samples[1];
    observer->samples[1] = observer->samples[2];
    observer->samples[2] = observer->samples[3];
    observer->samples[3] = raw_force_n;

    return 0.25 *
           (observer->samples[0] +
            observer->samples[1] +
            observer->samples[2] +
            observer->samples[3]);
}

static void update_airborne_metrics(const mjModel *m, const mjData *d, const ModelMap *map, double dt)
{
    const int base_qpos = m->jnt_qposadr[map->base_freejoint];
    double min_clearance;
    double max_clearance;

    measure_wheel_clearance(m, d, map, &min_clearance, &max_clearance);

    if (!g_airborne_metrics_valid)
    {
        g_wheel_clearance_min = min_clearance;
        g_wheel_clearance_max = max_clearance;
        g_airborne_metrics_valid = 1;
    }
    else
    {
        if (min_clearance < g_wheel_clearance_min)
        {
            g_wheel_clearance_min = min_clearance;
        }
        if (max_clearance > g_wheel_clearance_max)
        {
            g_wheel_clearance_max = max_clearance;
        }
    }

    if (min_clearance > 0.015)
    {
        float roll;
        float pitch;
        float yaw;
        mjtNum rotated_quat[4];
        const mjtNum *attitude_quat = &d->qpos[base_qpos + 3];

        g_airborne_time += dt;
        g_airborne_current_duration += dt;
        if (g_airborne_current_duration > g_airborne_max_duration)
        {
            g_airborne_max_duration = g_airborne_current_duration;
        }

        if (map->rotate_control_frame)
        {
            rotate_quat_into_controller_frame(attitude_quat, rotated_quat);
            attitude_quat = rotated_quat;
        }
        quat_to_euler(attitude_quat, &roll, &pitch, &yaw);
        const float roll_error = (float)wrap_pi((double)roll - g_xml_initial_rpy[0]);
        const float pitch_error = (float)wrap_pi((double)pitch - g_xml_initial_rpy[1]);
        const float yaw_error = (float)wrap_pi((double)yaw - g_xml_initial_rpy[2]);
        g_airborne_rpy_last_error[0] = roll_error;
        g_airborne_rpy_last_error[1] = pitch_error;
        g_airborne_rpy_last_error[2] = yaw_error;
        if (fabsf(roll_error) > g_airborne_abs_roll_max)
        {
            g_airborne_abs_roll_max = fabsf(roll_error);
        }
        if (fabsf(pitch_error) > g_airborne_abs_pitch_max)
        {
            g_airborne_abs_pitch_max = fabsf(pitch_error);
        }
        if (fabsf(yaw_error) > g_airborne_abs_yaw_error_max)
        {
            g_airborne_abs_yaw_error_max = fabsf(yaw_error);
        }
        for (int i = 0; i < 4; ++i)
        {
            const float error = (float)d->qpos[map->joint[i].qpos] - g_airborne_pose_target[i];
            g_airborne_joint_last_error[i] = error;
            if (fabsf(error) > g_airborne_joint_abs_max[i])
            {
                g_airborne_joint_abs_max[i] = fabsf(error);
            }
            g_airborne_joint_error_sq_sum += (double)error * (double)error;
            ++g_airborne_joint_error_samples;
        }
        const double sample_rms = sqrt(((double)g_airborne_joint_last_error[0] * g_airborne_joint_last_error[0] +
                                        (double)g_airborne_joint_last_error[1] * g_airborne_joint_last_error[1] +
                                        (double)g_airborne_joint_last_error[2] * g_airborne_joint_last_error[2] +
                                        (double)g_airborne_joint_last_error[3] * g_airborne_joint_last_error[3]) /
                                       4.0);
        if (sample_rms < g_airborne_joint_best_rms)
        {
            g_airborne_joint_best_rms = sample_rms;
            memcpy(g_airborne_joint_best_error,
                   g_airborne_joint_last_error,
                   sizeof(g_airborne_joint_best_error));
        }
        if (g_xml_initial_qpos != 0)
        {
            double full_pose_error_sq = 0.0;
            double full_pose_max_abs = 0.0;
            int full_pose_joint_count = 0;

            for (int joint = 0; joint < m->njnt; ++joint)
            {
                if (m->jnt_type[joint] != mjJNT_HINGE ||
                    joint == map->wheel[0].id ||
                    joint == map->wheel[1].id)
                {
                    continue;
                }

                const int qpos = m->jnt_qposadr[joint];
                const double error = wrap_pi((double)d->qpos[qpos] -
                                             (double)g_xml_initial_qpos[qpos]);
                full_pose_error_sq += error * error;
                if (fabs(error) > full_pose_max_abs)
                {
                    full_pose_max_abs = fabs(error);
                }
                ++full_pose_joint_count;
            }

            if (full_pose_joint_count > 0)
            {
                g_airborne_full_pose_last_rms =
                    sqrt(full_pose_error_sq / (double)full_pose_joint_count);
                if (g_airborne_full_pose_last_rms < g_airborne_full_pose_best_rms)
                {
                    g_airborne_full_pose_best_rms = g_airborne_full_pose_last_rms;
                }
                if (full_pose_max_abs > g_airborne_full_pose_max_abs)
                {
                    g_airborne_full_pose_max_abs = full_pose_max_abs;
                }
            }
        }
    }
    else
    {
        g_airborne_current_duration = 0.0;
    }
}

static void update_jump_attitude_metrics(const SimControllerState *state, double sim_time, double dt)
{
    if (!g_jump_attitude_active)
    {
        return;
    }

    g_jump_attitude_time += dt;
    if (fabsf(state->roll) > g_jump_abs_roll_max)
    {
        g_jump_abs_roll_max = fabsf(state->roll);
    }
    if (fabsf(state->pitch) > g_jump_abs_pitch_max)
    {
        g_jump_abs_pitch_max = fabsf(state->pitch);
    }
    if (state->pitch < g_jump_pitch_min)
    {
        g_jump_pitch_min = state->pitch;
        g_jump_pitch_min_time = sim_time;
    }
    if (state->pitch > g_jump_pitch_max)
    {
        g_jump_pitch_max = state->pitch;
        g_jump_pitch_max_time = sim_time;
    }
}

static void apply_auto_stop_schedule(double sim_time)
{
    if (g_auto_stop_time >= 0.0 &&
        sim_time >= g_auto_stop_time &&
        g_drive_command.mode == DRIVE_FORWARD)
    {
        request_drive_speed(&g_drive_command, 0.0f);
        g_auto_stop_time = -1.0;
        printf("Auto stop -> stand at t=%6.3f\n", sim_time);
        fflush(stdout);
    }
}

static void apply_auto_jump_schedule(double sim_time)
{
    if (g_auto_jump_time >= 0.0 &&
        sim_time >= g_auto_jump_time &&
        !SimController_IsJumping())
    {
        request_jump(&g_drive_command, sim_time);
        g_auto_jump_time = -1.0;
    }
}

static double measure_leg_length(const mjModel *m, const mjData *d, const ModelMap *map, int left_leg)
{
    const int hip_a_joint = left_leg ? map->joint[0].id : map->joint[2].id;
    const int hip_b_joint = left_leg ? map->joint[1].id : map->joint[3].id;
    const int wheel_joint = left_leg ? map->wheel[0].id : map->wheel[1].id;
    mjtNum hip_anchor[3];
    mjtNum wheel_anchor[3];
    mjtNum hip_axis[3];
    double delta[3];
    double along_axis = 0.0;
    double normal_sq = 0.0;

    (void)m;
    for (int i = 0; i < 3; ++i)
    {
        hip_anchor[i] =
            0.5 * (d->xanchor[3 * hip_a_joint + i] + d->xanchor[3 * hip_b_joint + i]);
        wheel_anchor[i] = d->xanchor[3 * wheel_joint + i];
        hip_axis[i] = d->xaxis[3 * hip_a_joint + i];
        delta[i] = (double)wheel_anchor[i] - (double)hip_anchor[i];
        along_axis += delta[i] * (double)hip_axis[i];
    }

    for (int i = 0; i < 3; ++i)
    {
        const double normal = delta[i] - along_axis * (double)hip_axis[i];
        normal_sq += normal * normal;
    }
    return sqrt(normal_sq);
}

static int measure_leg_axis_geometry(const mjModel *m,
                                     const mjData *d,
                                     const ModelMap *map,
                                     int left_leg,
                                     double *axis_length,
                                     double *axis_phi0)
{
    const int hip_a_joint = left_leg ? map->joint[0].id : map->joint[2].id;
    const int hip_b_joint = left_leg ? map->joint[1].id : map->joint[3].id;
    const int wheel_joint = left_leg ? map->wheel[0].id : map->wheel[1].id;
    const double *base_xmat = &d->xmat[9 * map->base_body];
    /* 与 CAD VMC 报告一致：腿平面 s=+base Y，d=-base Z。 */
    const double base_s_axis[3] = {base_xmat[1], base_xmat[4], base_xmat[7]};
    const double base_z_axis[3] = {base_xmat[2], base_xmat[5], base_xmat[8]};
    mjtNum hip_anchor[3];
    mjtNum wheel_anchor[3];
    mjtNum hip_axis[3];
    double delta[3];
    double normal[3];
    double along_axis = 0.0;
    double normal_sq = 0.0;
    double local_s = 0.0;
    double local_down = 0.0;

    if (map->base_body < 0)
    {
        return 0;
    }

    for (int i = 0; i < 3; ++i)
    {
        hip_anchor[i] =
            0.5 * (d->xanchor[3 * hip_a_joint + i] + d->xanchor[3 * hip_b_joint + i]);
        wheel_anchor[i] = d->xanchor[3 * wheel_joint + i];
        hip_axis[i] = d->xaxis[3 * hip_a_joint + i];
        delta[i] = (double)wheel_anchor[i] - (double)hip_anchor[i];
        along_axis += delta[i] * (double)hip_axis[i];
    }

    for (int i = 0; i < 3; ++i)
    {
        normal[i] = delta[i] - along_axis * (double)hip_axis[i];
        normal_sq += normal[i] * normal[i];
        local_s += normal[i] * base_s_axis[i];
        local_down -= normal[i] * base_z_axis[i];
    }

    if (axis_length != 0)
    {
        *axis_length = sqrt(normal_sq);
    }
    if (axis_phi0 != 0)
    {
        *axis_phi0 = atan2(local_down, local_s);
    }
    return 1;
}

static double distance3(const mjtNum *a, const mjtNum *b)
{
    const double dx = (double)a[0] - (double)b[0];
    const double dy = (double)a[1] - (double)b[1];
    const double dz = (double)a[2] - (double)b[2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}

static int get_site_pos(const mjModel *m, const mjData *d, const char *name, const mjtNum **pos)
{
    int site = find_optional_id(m, mjOBJ_SITE, name);
    if (site < 0)
    {
        return 0;
    }

    *pos = &d->site_xpos[3 * site];
    return 1;
}

static double measure_mid_to_site(const mjModel *m, const mjData *d, const ModelMap *map, int left_leg, const char *site_name)
{
    const int hip_a_joint = left_leg ? map->joint[0].id : map->joint[2].id;
    const int hip_b_joint = left_leg ? map->joint[1].id : map->joint[3].id;
    const int hip_a_body = m->jnt_bodyid[hip_a_joint];
    const int hip_b_body = m->jnt_bodyid[hip_b_joint];
    const mjtNum *site_pos = 0;
    mjtNum hip_mid[3];

    if (!get_site_pos(m, d, site_name, &site_pos))
    {
        return NAN;
    }

    for (int i = 0; i < 3; ++i)
    {
        hip_mid[i] = 0.5 * (d->xpos[3 * hip_a_body + i] + d->xpos[3 * hip_b_body + i]);
    }

    return distance3(hip_mid, site_pos);
}

static double joint_axis_distance(const mjModel *m, const mjData *d, const char *joint_a_name, const char *joint_b_name)
{
    const int joint_a = find_optional_id(m, mjOBJ_JOINT, joint_a_name);
    const int joint_b = find_optional_id(m, mjOBJ_JOINT, joint_b_name);
    double delta[3];
    double along_axis = 0.0;
    double normal_sq = 0.0;

    if (joint_a < 0 || joint_b < 0)
    {
        return NAN;
    }

    for (int i = 0; i < 3; ++i)
    {
        delta[i] = (double)d->xanchor[3 * joint_b + i] - (double)d->xanchor[3 * joint_a + i];
        along_axis += delta[i] * (double)d->xaxis[3 * joint_a + i];
    }

    for (int i = 0; i < 3; ++i)
    {
        const double normal = delta[i] - along_axis * (double)d->xaxis[3 * joint_a + i];
        normal_sq += normal * normal;
    }
    return sqrt(normal_sq);
}

static double joint_axis_alignment(const mjModel *m, const mjData *d, const char *joint_a_name, const char *joint_b_name)
{
    const int joint_a = find_optional_id(m, mjOBJ_JOINT, joint_a_name);
    const int joint_b = find_optional_id(m, mjOBJ_JOINT, joint_b_name);
    double dot = 0.0;

    if (joint_a < 0 || joint_b < 0)
    {
        return NAN;
    }

    for (int i = 0; i < 3; ++i)
    {
        dot += (double)d->xaxis[3 * joint_a + i] * (double)d->xaxis[3 * joint_b + i];
    }
    return dot;
}

static double measure_virtual_leg_length(const mjModel *m, const mjData *d, const ModelMap *map, int left_leg)
{
    const int hip_a_joint = left_leg ? map->joint[0].id : map->joint[2].id;
    const int hip_b_joint = left_leg ? map->joint[1].id : map->joint[3].id;
    const int hip_a_body = m->jnt_bodyid[hip_a_joint];
    const int hip_b_body = m->jnt_bodyid[hip_b_joint];
    const char *site_a_name = left_leg ? "OP-N" : "GH-F";
    const char *site_b_name = left_leg ? "KN-N" : "CF-F";
    const mjtNum *site_a = 0;
    const mjtNum *site_b = 0;
    mjtNum hip_mid[3];
    mjtNum foot_mid[3];

    if (!get_site_pos(m, d, site_a_name, &site_a) || !get_site_pos(m, d, site_b_name, &site_b))
    {
        return NAN;
    }

    for (int i = 0; i < 3; ++i)
    {
        hip_mid[i] = 0.5 * (d->xpos[3 * hip_a_body + i] + d->xpos[3 * hip_b_body + i]);
        foot_mid[i] = 0.5 * (site_a[i] + site_b[i]);
    }

    return distance3(hip_mid, foot_mid);
}

static void print_joint_qpos(const mjModel *m, const mjData *d, const char *name)
{
    const int joint = find_optional_id(m, mjOBJ_JOINT, name);
    if (joint >= 0)
    {
        printf("  %-14s qpos=% .6f\n", name, d->qpos[m->jnt_qposadr[joint]]);
    }
}

static void print_site_error(const mjModel *m, const mjData *d, const char *a, const char *b)
{
    const mjtNum *pa = 0;
    const mjtNum *pb = 0;
    if (get_site_pos(m, d, a, &pa) && get_site_pos(m, d, b, &pb))
    {
        printf("  %-6s <-> %-6s err=% .6f\n", a, b, distance3(pa, pb));
    }
}

static double measure_site_error(const mjModel *m,
                                 const mjData *d,
                                 const char *a,
                                 const char *b)
{
    const mjtNum *pa = 0;
    const mjtNum *pb = 0;
    if (!get_site_pos(m, d, a, &pa) || !get_site_pos(m, d, b, &pb))
    {
        return NAN;
    }
    return distance3(pa, pb);
}

static void print_geometry_debug(const mjModel *m, const mjData *d, const ModelMap *map, const char *tag)
{
    double left_axis_length = NAN;
    double right_axis_length = NAN;
    double left_axis_phi0 = NAN;
    double right_axis_phi0 = NAN;
    measure_leg_axis_geometry(m, d, map, 1, &left_axis_length, &left_axis_phi0);
    measure_leg_axis_geometry(m, d, map, 0, &right_axis_length, &right_axis_phi0);

    printf("geometry[%s]\n", tag);
    printf("  length left:  wheel_axis=% .6f OP-N=% .6f KN-N=% .6f\n",
           left_axis_length,
           measure_mid_to_site(m, d, map, 1, "OP-N"),
           measure_mid_to_site(m, d, map, 1, "KN-N"));
    printf("  length right: wheel_axis=% .6f GH-F=% .6f CF-F=% .6f\n",
           right_axis_length,
           measure_mid_to_site(m, d, map, 0, "GH-F"),
           measure_mid_to_site(m, d, map, 0, "CF-F"));
    printf("  axis phi0: left=% .6f right=% .6f INIT_L0_PITCH=% .6f\n",
           left_axis_phi0,
           right_axis_phi0,
           (double)INIT_L0_PITCH);
    printf("  axis distance left:  jIO-wheel=% .6f jIJ-wheel=% .6f jOP-wheel=% .6f jKN-wheel=% .6f\n",
           joint_axis_distance(m, d, "jIO", "jwheel_left"),
           joint_axis_distance(m, d, "jIJ", "jwheel_left"),
           joint_axis_distance(m, d, "jOP", "jwheel_left"),
           joint_axis_distance(m, d, "jKN", "jwheel_left"));
    printf("  axis distance right: jAG-wheel=% .6f jAB-wheel=% .6f jGH-wheel=% .6f jCF-wheel=% .6f\n",
           joint_axis_distance(m, d, "jAG", "jwheel_right"),
           joint_axis_distance(m, d, "jAB", "jwheel_right"),
           joint_axis_distance(m, d, "jGH", "jwheel_right"),
           joint_axis_distance(m, d, "jCF", "jwheel_right"));
    printf("  axis dot left:  jOP-wheel=% .6f jKN-wheel=% .6f\n",
           joint_axis_alignment(m, d, "jOP", "jwheel_left"),
           joint_axis_alignment(m, d, "jKN", "jwheel_left"));
    printf("  axis dot right: jGH-wheel=% .6f jCF-wheel=% .6f\n",
           joint_axis_alignment(m, d, "jGH", "jwheel_right"),
           joint_axis_alignment(m, d, "jCF", "jwheel_right"));
    printf("  equality residuals:\n");
    print_site_error(m, d, "IO-L", "MK-L");
    print_site_error(m, d, "OP-N", "KN-N");
    print_site_error(m, d, "AG-D", "EC-D");
    print_site_error(m, d, "GH-F", "CF-F");
    printf("  left joints:\n");
    print_joint_qpos(m, d, "jIO");
    print_joint_qpos(m, d, "jOP");
    print_joint_qpos(m, d, "jIJ");
    print_joint_qpos(m, d, "jJM");
    print_joint_qpos(m, d, "jMK");
    print_joint_qpos(m, d, "jKN");
    printf("  right joints:\n");
    print_joint_qpos(m, d, "jAG");
    print_joint_qpos(m, d, "jGH");
    print_joint_qpos(m, d, "jAB");
    print_joint_qpos(m, d, "jBE");
    print_joint_qpos(m, d, "jEC");
    print_joint_qpos(m, d, "jCF");
}

static void print_joint_axis_heights(const mjModel *m, const mjData *d)
{
    printf("axis_z=");
    for (int joint = 0; joint < m->njnt; ++joint)
    {
        if (m->jnt_type[joint] == mjJNT_FREE)
        {
            continue;
        }

        const char *name = mj_id2name(m, mjOBJ_JOINT, joint);
        if (name == 0)
        {
            name = "(unnamed)";
        }

        printf("%s:% .3f ", name, d->xanchor[3 * joint + 2]);
    }
    printf("\n");
}





static void write_output(mjData *d, const ModelMap *map, const SimControllerOutput *output)
{
    d->ctrl[map->actuator[0]] = g_joint_ctrl_sign[0] * output->joint_torque[0];
    d->ctrl[map->actuator[1]] = g_joint_ctrl_sign[1] * output->joint_torque[1];
    d->ctrl[map->actuator[2]] = g_joint_ctrl_sign[2] * output->joint_torque[2];
    d->ctrl[map->actuator[3]] = g_joint_ctrl_sign[3] * output->joint_torque[3];
    d->ctrl[map->actuator[4]] = g_wheel_ctrl_sign[0] * output->wheel_torque[0];
    d->ctrl[map->actuator[5]] = g_wheel_ctrl_sign[1] * output->wheel_torque[1];
}

static mjtNum clamp_actuator_ctrl(const mjModel *m, int actuator, mjtNum value)
{
    if (m->actuator_ctrllimited[actuator])
    {
        const mjtNum min_value = m->actuator_ctrlrange[2 * actuator + 0];
        const mjtNum max_value = m->actuator_ctrlrange[2 * actuator + 1];
        if (value < min_value)
        {
            return min_value;
        }
        if (value > max_value)
        {
            return max_value;
        }
    }
    return value;
}

static void clamp_output_to_model(const mjModel *m, const ModelMap *map, SimControllerOutput *output)
{
    output->joint_torque[0] = clamp_actuator_ctrl(m, map->actuator[0], output->joint_torque[0]);
    output->joint_torque[1] = clamp_actuator_ctrl(m, map->actuator[1], output->joint_torque[1]);
    output->joint_torque[2] = clamp_actuator_ctrl(m, map->actuator[2], output->joint_torque[2]);
    output->joint_torque[3] = clamp_actuator_ctrl(m, map->actuator[3], output->joint_torque[3]);
    output->wheel_torque[0] = clamp_actuator_ctrl(m, map->actuator[4], output->wheel_torque[0]);
    output->wheel_torque[1] = clamp_actuator_ctrl(m, map->actuator[5], output->wheel_torque[1]);
}

static void clamp_output_to_motor_limits(SimControllerOutput *output, int jump_phase)
{
    const float joint_limit =
        jump_phase > 0 ? MAX_JOINT_TORQUE_JUMP : MAX_JOINT_TORQUE;

    for (int joint = 0; joint < 4; ++joint)
    {
        output->joint_torque[joint] =
            (float)clamp_double(output->joint_torque[joint],
                                -joint_limit,
                                joint_limit);
    }
    const float wheel_limit =
        jump_phase > 0 ? fabsf(g_jump_motor_wheel_limit) : LK_MAX_MF_TORQUE;
    const float wheel_min =
        jump_phase > 0 ? -wheel_limit : LK_MIN_MF_TORQUE;
    const float wheel_max =
        jump_phase > 0 ? wheel_limit : LK_MAX_MF_TORQUE;
    for (int wheel = 0; wheel < 2; ++wheel)
    {
        output->wheel_torque[wheel] =
            (float)clamp_double(output->wheel_torque[wheel],
                                wheel_min,
                                wheel_max);
    }
}

static void apply_wheel_balance_override(const SimControllerState *state,
                                         SimControllerOutput *output)
{
    const int steer_level_pitch =
        g_drive_command.planar_hold &&
        g_steer_machine.phase != STEER_PHASE_RECOVER;
    const float pitch_target =
        steer_level_pitch ? g_steer_pitch_target : g_balance_pitch_target;
    float position_error = state->body_x - g_drive_command.target_x;
    float velocity_error = state->body_v - g_drive_command.current_speed;

    if (g_drive_command.mode == DRIVE_STAND &&
        g_drive_command.yaw_lock)
    {
        const float forward_x = -sinf(state->yaw);
        const float forward_y = cosf(state->yaw);
        const float error_x = state->body_x - g_drive_command.target_x;
        const float error_y = state->body_y - g_drive_command.target_y;
        position_error = error_x * forward_x + error_y * forward_y;
        position_error = (float)clamp_double(position_error, -0.12, 0.12);
        velocity_error =
            state->body_v * forward_x +
            state->body_v_y * forward_y -
            g_drive_command.current_speed;
    }

    const float pitch_term = g_balance_pitch_kp * (state->pitch - pitch_target) +
                             g_balance_pitch_kd * state->gyro[1];
    const float pos_term = g_drive_command.mode == DRIVE_FORWARD
                               ? 0.0f
                               : g_balance_pos_kp * g_drive_command.position_hold_blend *
                                     position_error;
    const float vel_term = g_balance_vel_kd * velocity_error;
    const float drive_term = g_balance_drive_kff * g_drive_command.current_speed;
    float yaw_term = 0.0f;
    const float common =
        (float)clamp_double(pitch_term + pos_term + vel_term + drive_term,
                            -g_balance_wheel_limit,
                            g_balance_wheel_limit);
    if (g_drive_command.yaw_lock)
    {
        const float yaw_error =
            (float)wrap_pi((double)state->yaw - g_drive_command.yaw_hold);
        const float yaw_kp =
            g_drive_command.planar_hold ? g_steer_yaw_kp : g_balance_yaw_kp;
        const float yaw_rate_kd =
            g_drive_command.planar_hold ? g_steer_yaw_rate_kd : g_balance_yaw_kd;
        const float yaw_rate_error =
            state->gyro[2] -
            (g_drive_command.planar_hold ? g_drive_command.yaw_rate_ref : 0.0f);
        const float yaw_limit =
            g_drive_command.planar_hold ? fabsf(g_steer_yaw_torque_limit)
                                        : g_balance_wheel_limit;
        yaw_term =
            (float)clamp_double(yaw_kp * yaw_error +
                                    yaw_rate_kd * yaw_rate_error,
                                -yaw_limit,
                                yaw_limit);
    }

    output->wheel_torque[0] = common - yaw_term;
    output->wheel_torque[1] = common + yaw_term;
}

static void apply_jump_pitch_wheel_hold(const SimControllerState *state,
                                        int jump_phase,
                                        const double wheel_contact_normal_n[2],
                                        SimControllerOutput *output)
{
    float wheel_torque = 0.0f;
    if (jump_phase >= 3)
    {
        const float pitch_target =
            jump_phase == 6
                ? g_jump_recover_pitch_target
                : g_jump_landing_pitch_target;
        const float pitch_term =
            g_jump_pitch_wheel_kp * (state->pitch - pitch_target) +
            g_jump_pitch_wheel_kd * state->gyro[1];
        wheel_torque =
            (float)clamp_double(pitch_term,
                                -g_jump_pitch_wheel_limit,
                                g_jump_pitch_wheel_limit);
    }
    float yaw_term = 0.0f;
    const int stable_wheel_contact =
        wheel_contact_normal_n != 0 &&
        wheel_contact_normal_n[0] >= MUJOCO_JUMP_TOUCHDOWN_FORCE &&
        wheel_contact_normal_n[1] >= MUJOCO_JUMP_TOUCHDOWN_FORCE;
    const int takeoff_yaw_control =
        jump_phase == 2 &&
        wheel_contact_normal_n != 0 &&
        (wheel_contact_normal_n[0] >= MUJOCO_JUMP_TOUCHDOWN_FORCE ||
         wheel_contact_normal_n[1] >= MUJOCO_JUMP_TOUCHDOWN_FORCE);
    const int landing_yaw_control =
        jump_phase >= g_jump_landing_yaw_start_phase &&
        jump_phase <= 5 &&
        (jump_phase < 5 || stable_wheel_contact);
    if ((takeoff_yaw_control || landing_yaw_control) &&
        g_drive_command.yaw_lock)
    {
        const float yaw_error =
            (float)wrap_pi((double)state->yaw - g_drive_command.yaw_hold);
        const float yaw_kp =
            takeoff_yaw_control ? g_jump_takeoff_yaw_kp
                                : g_jump_landing_yaw_kp;
        const float yaw_kd =
            takeoff_yaw_control ? g_jump_takeoff_yaw_kd
                                : g_jump_landing_yaw_kd;
        const float yaw_limit =
            takeoff_yaw_control ? g_jump_takeoff_yaw_limit
                                : g_jump_landing_yaw_limit;
        yaw_term =
            (float)clamp_double(yaw_kp * yaw_error +
                                    yaw_kd * state->gyro[2],
                                -yaw_limit,
                                yaw_limit);
    }

    output->wheel_torque[0] =
        (float)clamp_double(wheel_torque - yaw_term,
                            -g_jump_pitch_wheel_limit,
                            g_jump_pitch_wheel_limit);
    output->wheel_torque[1] =
        (float)clamp_double(wheel_torque + yaw_term,
                            -g_jump_pitch_wheel_limit,
                            g_jump_pitch_wheel_limit);
}




static void mouse_button_callback(GLFWwindow *window, int button, int act, int mods)
{
    (void)button;
    (void)act;
    (void)mods;

    g_button_left = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    g_button_middle = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    g_button_right = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    glfwGetCursorPos(window, &g_last_x, &g_last_y);
}

static void mouse_move_callback(GLFWwindow *window, double xpos, double ypos)
{
    if (!g_button_left && !g_button_middle && !g_button_right)
    {
        return;
    }

    double dx = xpos - g_last_x;
    double dy = ypos - g_last_y;
    g_last_x = xpos;
    g_last_y = ypos;

    int width = 0;
    int height = 0;
    glfwGetWindowSize(window, &width, &height);

    int shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;

    mjtMouse action;
    if (g_button_right)
    {
        action = shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    }
    else if (g_button_left)
    {
        action = shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    }
    else
    {
        action = mjMOUSE_ZOOM;
    }

    mjv_moveCamera(g_model, action, dx / (double)height, dy / (double)height, &g_scene, &g_camera);
}

static void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    (void)window;
    (void)xoffset;
    mjv_moveCamera(g_model, mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, &g_scene, &g_camera);
}

static void keyboard_callback(GLFWwindow *window, int key, int scancode, int act, int mods)
{
    (void)scancode;
    (void)mods;

    if (act == GLFW_PRESS && key == GLFW_KEY_ESCAPE)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    else if (act == GLFW_PRESS && key == GLFW_KEY_SPACE)
    {
        g_paused = !g_paused;
        printf("Simulation %s at t=%6.3f\n", g_paused ? "paused" : "resumed", g_data ? g_data->time : 0.0);
        fflush(stdout);
    }
}

static int open_control_csv(const char *path)
{
    if (path == 0)
    {
        return 1;
    }

    g_control_csv = fopen(path, "w");
    if (g_control_csv == 0)
    {
        fprintf(stderr, "Failed to open control CSV: %s\n", path);
        return 0;
    }

    fprintf(g_control_csv,
            "t_s,controller_tick,mode,body_x_m,body_y_m,body_z_m,body_vx_mps,body_vy_mps,body_vz_mps,roll_rad,pitch_rad,yaw_rad,pitch_rate_radps,");
    for (int joint = 0; joint < 4; ++joint)
    {
        fprintf(g_control_csv, "q%d_rad,", joint);
    }
    for (int joint = 0; joint < 4; ++joint)
    {
        fprintf(g_control_csv, "qd%d_radps,", joint);
    }
    fprintf(g_control_csv,
            "q_front_left_rad,q_rear_left_rad,q_front_right_rad,q_rear_right_rad,"
            "l0_set_left_m,l0_set_right_m,l0_left_m,l0_right_m,axis_l0_left_m,axis_l0_right_m,"
            "d_l0_left_mps,d_l0_right_mps,phi0_left_rad,phi0_right_rad,axis_phi0_left_rad,axis_phi0_right_rad,"
            "d_phi0_left_radps,d_phi0_right_radps,theta_left_rad,theta_right_rad,d_theta_left_radps,d_theta_right_radps,"
            "kin_left,kin_right,closure_io_mk_left_m,closure_op_kn_left_m,closure_ag_ec_right_m,closure_gh_cf_right_m,"
            "f0_total_left_n,f0_total_right_n,f0_gravity_left_n,f0_gravity_right_n,"
            "f0_pid_residual_left_n,f0_pid_residual_right_n,tp_left_nm,tp_right_nm,");
    for (int joint = 0; joint < 4; ++joint)
    {
        fprintf(g_control_csv, "tau_vmc_j%d_nm,", joint);
    }
    for (int joint = 0; joint < 4; ++joint)
    {
        fprintf(g_control_csv, "tau_mit_j%d_nm,", joint);
    }
    for (int joint = 0; joint < 4; ++joint)
    {
        fprintf(g_control_csv, "tau_requested_j%d_nm,", joint);
    }
    for (int joint = 0; joint < 4; ++joint)
    {
        fprintf(g_control_csv, "tau_final_j%d_nm,", joint);
    }
    for (int joint = 0; joint < 4; ++joint)
    {
        fprintf(g_control_csv, "joint_sat_j%d,", joint);
    }
    fprintf(g_control_csv,
            "wheel_requested_left_nm,wheel_requested_right_nm,wheel_final_left_nm,wheel_final_right_nm,"
            "wheel_sat_left,wheel_sat_right,wheel_vel_left_radps,wheel_vel_right_radps\n");
    g_control_csv_next_time = 0.0;
    return 1;
}

static void record_control_csv(const mjModel *m,
                               const mjData *d,
                               const ModelMap *map,
                               const SimControllerState *state,
                               const SimControllerOutput *output,
                               const SimControllerControlBreakdown *breakdown,
                               const double requested_torque[6],
                               int controller_tick)
{
    if (g_control_csv == 0 || d->time + 1.0e-12 < g_control_csv_next_time)
    {
        return;
    }
    while (g_control_csv_next_time <= d->time + 1.0e-12)
    {
        g_control_csv_next_time += g_control_csv_period_s;
    }

    double axis_l0[2] = {NAN, NAN};
    double axis_phi0[2] = {NAN, NAN};
    measure_leg_axis_geometry(m, d, map, 1, &axis_l0[0], &axis_phi0[0]);
    measure_leg_axis_geometry(m, d, map, 0, &axis_l0[1], &axis_phi0[1]);

    fprintf(g_control_csv,
            "%.9f,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,",
            d->time,
            controller_tick,
            (int)chassis_move.mode,
            state->body_x,
            state->body_y,
            state->body_z,
            state->body_v,
            state->body_v_y,
            state->body_z_vel,
            state->roll,
            state->pitch,
            state->yaw,
            state->gyro[1]);
    for (int joint = 0; joint < 4; ++joint)
    {
        fprintf(g_control_csv, "%.9f,", state->joint_pos[joint]);
    }
    for (int joint = 0; joint < 4; ++joint)
    {
        fprintf(g_control_csv, "%.9f,", state->joint_vel[joint]);
    }
    fprintf(g_control_csv,
            "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,"
            "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,"
            "%u,%u,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,",
            left.q_front,
            left.q_rear,
            right.q_front,
            right.q_rear,
            breakdown->l0_set[0],
            breakdown->l0_set[1],
            breakdown->l0[0],
            breakdown->l0[1],
            axis_l0[0],
            axis_l0[1],
            breakdown->d_l0[0],
            breakdown->d_l0[1],
            breakdown->phi0[0],
            breakdown->phi0[1],
            axis_phi0[0],
            axis_phi0[1],
            breakdown->d_phi0[0],
            breakdown->d_phi0[1],
            breakdown->theta[0],
            breakdown->theta[1],
            breakdown->d_theta[0],
            breakdown->d_theta[1],
            (unsigned int)left.kinematics_valid,
            (unsigned int)right.kinematics_valid,
            measure_site_error(m, d, "IO-L", "MK-L"),
            measure_site_error(m, d, "OP-N", "KN-N"),
            measure_site_error(m, d, "AG-D", "EC-D"),
            measure_site_error(m, d, "GH-F", "CF-F"),
            breakdown->f0_total[0],
            breakdown->f0_total[1],
            breakdown->f0_gravity[0],
            breakdown->f0_gravity[1],
            breakdown->f0_leg_pid[0],
            breakdown->f0_leg_pid[1],
            breakdown->tp_total[0],
            breakdown->tp_total[1]);
    for (int joint = 0; joint < 4; ++joint)
    {
        fprintf(g_control_csv, "%.9f,", breakdown->joint_torque_vmc[joint]);
    }
    for (int joint = 0; joint < 4; ++joint)
    {
        fprintf(g_control_csv, "%.9f,", breakdown->joint_torque_mit[joint]);
    }
    for (int joint = 0; joint < 4; ++joint)
    {
        fprintf(g_control_csv, "%.9f,", requested_torque[joint]);
    }
    for (int joint = 0; joint < 4; ++joint)
    {
        fprintf(g_control_csv, "%.9f,", output->joint_torque[joint]);
    }
    for (int joint = 0; joint < 4; ++joint)
    {
        fprintf(g_control_csv,
                "%d,",
                fabs(requested_torque[joint] - output->joint_torque[joint]) > 1.0e-6);
    }
    fprintf(g_control_csv,
            "%.9f,%.9f,%.9f,%.9f,%d,%d,%.9f,%.9f\n",
            requested_torque[4],
            requested_torque[5],
            output->wheel_torque[0],
            output->wheel_torque[1],
            fabs(requested_torque[4] - output->wheel_torque[0]) > 1.0e-6,
            fabs(requested_torque[5] - output->wheel_torque[1]) > 1.0e-6,
            state->wheel_vel[0],
            state->wheel_vel[1]);
}

static void step_controller(const mjModel *m,
                            mjData *d,
                            const ModelMap *map,
                            int print_line,
                            int zero_control,
                            int zero_wheels,
                            int invert_right_joints,
                            int start_mode,
                            double standup_time,
                            int controller_tick,
                            int *switched_to_safe)
{
    SimControllerState state;
    SimControllerOutput output;
    SimControllerControlBreakdown control_breakdown;
    SteerCommand steer_command;
    double requested_torque[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    read_state(m, d, map, &state);
    memset(&control_breakdown, 0, sizeof(control_breakdown));
    memset(&steer_command, 0, sizeof(steer_command));
    update_body_z_range(&state);
    update_jump_attitude_metrics(&state, d->time, m->opt.timestep);
    if (zero_control)
    {
        memset(&output, 0, sizeof(output));
    }
    else if (!controller_tick && g_held_controller_output_valid)
    {
        output = g_held_controller_output;
        control_breakdown = g_held_control_breakdown;
        memcpy(requested_torque,
               g_held_requested_torque,
               sizeof(requested_torque));
    }
    else
    {
        if (start_mode == CHASSIS_STAND_UP && standup_time >= 0.0 && d->time >= standup_time)
        {
            SimController_SetMode(CHASSIS_SAFE);
            if (switched_to_safe != 0 && *switched_to_safe == 0)
            {
                g_drive_command.hold_position_pending = 1;
                g_drive_command.position_hold_blend = 0.0f;
                printf("Switch STAND_UP -> SAFE at t=%6.3f\n", d->time);
                fflush(stdout);
                *switched_to_safe = 1;
            }
        }
        apply_auto_stop_schedule(d->time);
        double observation_clearance[2];
        double observation_contact_force[2];
        measure_wheel_clearances(m, d, map, observation_clearance);
        measure_wheel_contact_normal_force(m,
                                           d,
                                           map,
                                           observation_contact_force);
        const double observation_min_clearance =
            fmin(observation_clearance[0], observation_clearance[1]);
        const int airborne_before_step = observation_min_clearance > 0.005;
        const float observation_clearance_f[2] = {
            (float)observation_clearance[0],
            (float)observation_clearance[1],
        };
        const float observation_contact_force_f[2] = {
            (float)observation_contact_force[0],
            (float)observation_contact_force[1],
        };
        SimController_SetFlightObservation(airborne_before_step,
                                           observation_clearance_f,
                                           observation_contact_force_f);
        SimController_SetState(&state);
        apply_auto_jump_schedule(d->time);
        update_drive_command(&g_drive_command, &state, g_controller_dt_s);
        {
            SteerObservation steer_observation = {
                .dt = g_controller_dt_s,
                .input = current_steer_input(d->time),
                .current_yaw = state.yaw,
                .gyro_z = state.gyro[2],
                .left_l0 = left.L0,
                .right_l0 = right.L0,
                .body_x = state.body_x,
                .body_y = state.body_y,
                .body_v_x = state.body_v,
                .body_v_y = state.body_v_y,
                .base_leg_set = g_drive_command.leg_set,
                .base_yaw_hold = g_drive_command.yaw_hold,
                .roll = state.roll,
                .pitch = state.pitch,
                .drive_mode = steer_drive_mode_from_drive(g_drive_command.mode),
                .chassis_safe = chassis_move.mode == CHASSIS_SAFE,
                .start_enabled = chassis_move.start_flag == 1,
                .jump_active = SimController_IsJumping(),
            };
            SteerStateMachine_Step(&g_steer_machine,
                                   &steer_observation,
                                   &steer_command);
            if (steer_command.active)
            {
                g_steer_pose_hold_active = 1;
                g_drive_command.leg_set = steer_command.leg_set;
                g_drive_command.yaw_hold = steer_command.yaw_hold;
                g_drive_command.yaw_rate_ref = steer_command.yaw_rate_ref;
                g_drive_command.yaw_lock = steer_command.yaw_lock;
                g_drive_command.planar_hold = steer_command.planar_lock;
                g_drive_command.target_x = steer_command.target_x;
                g_drive_command.target_y = steer_command.target_y;
                g_drive_command.position_hold_blend = 1.0f;
                if (steer_command.force_stand)
                {
                    g_drive_command.mode = DRIVE_STAND;
                    g_drive_command.current_speed = 0.0f;
                }
            }
            else
            {
                g_drive_command.yaw_rate_ref = 0.0f;
                g_drive_command.planar_hold = 0;
                if (g_steer_command_was_active)
                {
                    const int normal_steer_exit =
                        g_steer_machine.phase == STEER_PHASE_IDLE;
                    g_drive_command.target_x =
                        normal_steer_exit ? g_steer_machine.anchor_x
                                          : state.body_x;
                    g_drive_command.target_y =
                        normal_steer_exit ? g_steer_machine.anchor_y
                                          : state.body_y;
                    g_drive_command.yaw_hold = state.yaw;
                    g_drive_command.current_speed = 0.0f;
                    g_drive_command.position_hold_blend =
                        normal_steer_exit ? 1.0f : 0.0f;
                    if (!normal_steer_exit)
                    {
                        g_steer_pose_hold_active = 0;
                    }
                    g_drive_command.hold_position_pending = 0;
                    g_drive_command.hold_yaw_pending = 0;
                }
            }
            g_steer_command_was_active = steer_command.active;
            update_steer_metrics(&steer_command,
                                 &state,
                                 g_controller_dt_s);
        }
        SimController_SetDriveContext(g_drive_command.mode == DRIVE_FORWARD,
                                      g_drive_command.position_hold_blend,
                                      g_drive_command.yaw_lock);
        SimController_SetCommand(g_drive_command.current_speed,
                                 g_drive_command.target_x,
                                 g_drive_command.leg_set,
                                 g_drive_command.roll_set,
                                 g_drive_command.yaw_lock ? g_drive_command.yaw_hold : 0.0f);
        SimController_Step(g_controller_dt_s);
        SimController_GetOutput(&output);
        SimController_GetControlBreakdown(&control_breakdown);
        const int airborne = wheels_are_airborne(m, d, map);
        const int jump_phase = chassis_move.jump_flag > chassis_move.jump_flag2
                                   ? chassis_move.jump_flag
                                   : chassis_move.jump_flag2;
        const int jump_launch_active = jump_phase >= 2 && jump_phase <= 5;
        const int suppress_wheel_balance = jump_launch_active || airborne;
        if (g_drive_command.mode == DRIVE_JUMP && !SimController_IsJumping())
        {
            queue_drive_mode(&g_drive_command, DRIVE_STAND);
            printf("Drive mode -> stand at t=%6.3f after jump\n", d->time);
            fflush(stdout);
        }
        if (g_use_wheel_balance_override &&
            !suppress_wheel_balance &&
            (chassis_move.mode == CHASSIS_SAFE || chassis_move.mode == CHASSIS_STAND_UP))
        {
            apply_wheel_balance_override(&state, &output);
        }
        const float native_wheel_torque[2] = {
            output.wheel_torque[0],
            output.wheel_torque[1],
        };
        if (suppress_wheel_balance)
        {
            output.wheel_torque[0] = 0.0f;
            output.wheel_torque[1] = 0.0f;
        }
        if (airborne || (jump_phase >= 2 && jump_phase <= 5))
        {
            apply_jump_pitch_wheel_hold(&state,
                                        jump_phase,
                                        observation_contact_force,
                                        &output);
        }
        else if (jump_phase == 6)
        {
            SimControllerOutput jump_hold_output = output;
            apply_jump_pitch_wheel_hold(&state,
                                        jump_phase,
                                        observation_contact_force,
                                        &jump_hold_output);
            const float native_blend =
                (float)clamp_double(
                    SimController_GetJumpWheelBalanceBlend(),
                    0.0,
                    1.0);
            output.wheel_torque[0] =
                (1.0f - native_blend) * jump_hold_output.wheel_torque[0] +
                native_blend * native_wheel_torque[0];
            output.wheel_torque[1] =
                (1.0f - native_blend) * jump_hold_output.wheel_torque[1] +
                native_blend * native_wheel_torque[1];
        }
        if (zero_wheels)
        {
            output.wheel_torque[0] = 0.0f;
            output.wheel_torque[1] = 0.0f;
        }
        if (invert_right_joints)
        {
            output.joint_torque[2] = -output.joint_torque[2];
            output.joint_torque[3] = -output.joint_torque[3];
        }
        for (int joint = 0; joint < 4; ++joint)
        {
            requested_torque[joint] = output.joint_torque[joint];
        }
        requested_torque[4] = output.wheel_torque[0];
        requested_torque[5] = output.wheel_torque[1];
        clamp_output_to_motor_limits(&output, jump_phase);
        clamp_output_to_model(m, map, &output);
        g_held_controller_output = output;
        g_held_control_breakdown = control_breakdown;
        memcpy(g_held_requested_torque,
               requested_torque,
               sizeof(g_held_requested_torque));
        g_held_controller_output_valid = 1;
    }
    write_output(d, map, &output);
    record_control_csv(m,
                       d,
                       map,
                       &state,
                       &output,
                       &control_breakdown,
                       requested_torque,
                       controller_tick);

    mj_step(m, d);
    if (g_jump_telemetry != 0)
    {
        const int base_qpos = m->jnt_qposadr[map->base_freejoint];
        const int base_qvel = m->jnt_dofadr[map->base_freejoint];
        float base_roll = 0.0f;
        float base_pitch = 0.0f;
        float base_yaw = 0.0f;
        double min_clearance = 0.0;
        double max_clearance = 0.0;
        double wheel_clearance_lr[2];
        const double base_accel_z_mps2 = d->qacc[base_qvel + 2];
        double base_rpy_rad[3];
        double vmc_support_force_raw[2];
        double vmc_support_force_filtered[2];
        int vmc_airborne[2];
        double contact_normal_force[2];
        double command_torque[6];
        double applied_torque[6];

        measure_wheel_clearances(m, d, map, wheel_clearance_lr);
        min_clearance = fmin(wheel_clearance_lr[0], wheel_clearance_lr[1]);
        max_clearance = fmax(wheel_clearance_lr[0], wheel_clearance_lr[1]);
        (void)max_clearance;
        measure_wheel_contact_normal_force(m,
                                           d,
                                           map,
                                           contact_normal_force);
        if (map->rotate_control_frame)
        {
            mjtNum rotated_quat[4];
            rotate_quat_into_controller_frame(&d->qpos[base_qpos + 3], rotated_quat);
            quat_to_euler(rotated_quat, &base_roll, &base_pitch, &base_yaw);
        }
        else
        {
            quat_to_euler(&d->qpos[base_qpos + 3], &base_roll, &base_pitch, &base_yaw);
        }
        base_rpy_rad[0] = base_roll;
        base_rpy_rad[1] = base_pitch;
        base_rpy_rad[2] = base_yaw;
        vmc_support_force_raw[0] =
            estimate_vmc_support_force(&left, base_accel_z_mps2);
        vmc_support_force_raw[1] =
            estimate_vmc_support_force(&right, base_accel_z_mps2);
        for (int leg = 0; leg < 2; ++leg)
        {
            vmc_support_force_filtered[leg] =
                update_support_force_observer(&g_support_force_observer[leg],
                                              vmc_support_force_raw[leg]);
            vmc_airborne[leg] =
                vmc_support_force_filtered[leg] < TAKE_OFF_FN_THRESHOLD;
        }
        const int contact_airborne =
            contact_normal_force[0] <= 0.1 &&
            contact_normal_force[1] <= 0.1;
        for (int actuator = 0; actuator < 6; ++actuator)
        {
            const int actuator_id = map->actuator[actuator];
            const double controller_sign =
                actuator < 4
                    ? (double)g_joint_ctrl_sign[actuator]
                    : (double)g_wheel_ctrl_sign[actuator - 4];
            command_torque[actuator] =
                controller_sign * d->ctrl[actuator_id];
            applied_torque[actuator] =
                controller_sign * d->actuator_force[actuator_id];
        }
        JumpTelemetry_Record(
            g_jump_telemetry,
            d->time,
            state.body_x,
            state.body_y,
            state.body_v,
            state.body_v_y,
            d->qpos[base_qpos + 2],
            min_clearance,
            wheel_clearance_lr,
            base_accel_z_mps2,
            base_rpy_rad,
            vmc_support_force_raw,
            vmc_support_force_filtered,
            vmc_airborne,
            contact_normal_force,
            contact_airborne,
            requested_torque,
            command_torque,
            applied_torque,
            &control_breakdown,
            g_drive_command.mode == DRIVE_JUMP || SimController_IsJumping(),
            chassis_move.jump_flag > chassis_move.jump_flag2
                ? chassis_move.jump_flag
                : chassis_move.jump_flag2,
            wheels_are_airborne(m, d, map));
    }
    update_airborne_metrics(m, d, map, m->opt.timestep);
    if (g_jump_attitude_active)
    {
        const double base_contact_force =
            measure_base_contact_normal_force(m, d, map);
        if (base_contact_force > g_jump_base_contact_peak_n)
        {
            g_jump_base_contact_peak_n = base_contact_force;
            g_jump_base_contact_peak_time = d->time;
        }
    }
    if (g_jump_attitude_active && !SimController_IsJumping() && !wheels_are_airborne(m, d, map))
    {
        g_jump_attitude_active = 0;
    }

    if (print_line)
    {
        const double virtual_left = measure_virtual_leg_length(m, d, map, 1);
        const double virtual_right = measure_virtual_leg_length(m, d, map, 0);
        const double wheel_left = measure_leg_length(m, d, map, 1);
        const double wheel_right = measure_leg_length(m, d, map, 0);
        double axis_length_unused = NAN;
        double axis_phi_left = NAN;
        double axis_phi_right = NAN;
        measure_leg_axis_geometry(m, d, map, 1, &axis_length_unused, &axis_phi_left);
        measure_leg_axis_geometry(m, d, map, 0, &axis_length_unused, &axis_phi_right);

        printf("t=%6.3f drive=%s steer=%s steer_in=% .1f jump=[%d %d] keys=[F:%d B:%d U:%d D:%d L:%d R:%d] vx_vis=% .3f vx_ctrl=% .3f xy_ref=[% .3f % .3f] pos_hold=% .2f planar=%d leg_ref=% .3f yaw_ref=% .3f yaw_rate_ref=% .3f pos=[% .3f % .3f % .3f] rpy=[% .3f % .3f % .3f] wheel_v=[% .3f % .3f] "
               "vmcL0=[% .6f % .6f] siteL=[% .3f % .3f] axisL=[% .6f % .6f] "
               "vmcPhi0=[% .6f % .6f] axisPhi0=[% .6f % .6f] kin=[%u %u] "
               "q=[% .6f % .6f | % .6f % .6f] tauQ=[% .2f % .2f | % .2f % .2f] "
               "phi1=[% .3f % .3f] phi4=[% .3f % .3f] "
               "u=[% .2f % .2f % .2f % .2f | % .2f % .2f]\n",
               d->time,
               drive_command_name(&g_drive_command),
               SteerStateMachine_PhaseName(g_steer_machine.phase),
               current_steer_input(d->time),
               chassis_move.jump_flag2,
               chassis_move.jump_flag,
               g_key_forward,
               g_key_backward,
               g_key_leg_up,
               g_key_leg_down,
               g_key_turn_left,
               g_key_turn_right,
               controller_to_visual_speed(g_drive_command.current_speed),
               g_drive_command.current_speed,
               g_drive_command.target_x,
               g_drive_command.target_y,
               g_drive_command.position_hold_blend,
               g_drive_command.planar_hold,
               g_drive_command.leg_set,
               g_drive_command.yaw_hold,
               g_drive_command.yaw_rate_ref,
               state.body_x,
               state.body_y,
               state.body_z,
               state.roll,
               state.pitch,
               state.yaw,
               state.wheel_vel[0],
               state.wheel_vel[1],
               left.L0,
               right.L0,
               virtual_left,
               virtual_right,
               wheel_left,
               wheel_right,
               left.phi0,
               right.phi0,
               axis_phi_left,
               axis_phi_right,
               (unsigned int)left.kinematics_valid,
               (unsigned int)right.kinematics_valid,
               left.q_front,
               left.q_rear,
               right.q_front,
               right.q_rear,
               left.raw_joint_torque[0],
               left.raw_joint_torque[1],
               right.raw_joint_torque[0],
               right.raw_joint_torque[1],
               left.phi1,
               right.phi1,
               left.phi4,
               right.phi4,
               output.joint_torque[0],
               output.joint_torque[1],
               output.joint_torque[2],
               output.joint_torque[3],
               output.wheel_torque[0],
               output.wheel_torque[1]);
        print_joint_axis_heights(m, d);
    }
}

static int run_headless(const mjModel *m,
                        mjData *d,
                        const ModelMap *map,
                        double sim_time,
                        int zero_control,
                        int zero_wheels,
                        int invert_right_joints,
                        int freeze_init,
                        int start_mode,
                        double standup_time)
{
    if (freeze_init)
    {
        printf("Initial state frozen at t=%6.3f\n", d->time);
        return 0;
    }

    int switched_to_safe = 0;
    int steps = (int)(sim_time / m->opt.timestep);
    for (int i = 0; i < steps; ++i)
    {
        step_controller(m,
                        d,
                        map,
                        i % 500 == 0,
                        zero_control,
                        zero_wheels,
                        invert_right_joints,
                        start_mode,
                        standup_time,
                        i % g_controller_period_steps == 0,
                        &switched_to_safe);
    }

    return 0;
}

static int run_viewer(mjModel *m,
                      mjData *d,
                      const ModelMap *map,
                      double sim_time,
                      int zero_control,
                      int zero_wheels,
                      int invert_right_joints,
                      int freeze_init,
                      int start_mode,
                      double standup_time)
{
    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW. If you are in WSL, check that WSLg/X server is available.\n");
        return 1;
    }

    GLFWwindow *window = glfwCreateWindow(1200, 900, "rm mujoco bridge", 0, 0);
    if (!window)
    {
        glfwTerminate();
        fprintf(stderr, "Failed to create GLFW window.\n");
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    g_model = m;
    g_data = d;
    mjv_defaultCamera(&g_camera);
    mjv_defaultOption(&g_option);
    mjv_defaultScene(&g_scene);
    mjr_defaultContext(&g_context);

    g_camera.distance = 2.0;
    g_camera.azimuth = 135.0;
    g_camera.elevation = -20.0;
    g_camera.lookat[0] = 0.0;
    g_camera.lookat[1] = 0.0;
    g_camera.lookat[2] = 0.25;

    mjv_makeScene(m, &g_scene, 2000);
    mjr_makeContext(m, &g_context, mjFONTSCALE_150);

    glfwSetKeyCallback(window, keyboard_callback);
    glfwSetCursorPosCallback(window, mouse_move_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);

    double last_wall = glfwGetTime();
    int print_tick = 0;
    int switched_to_safe = 0;

    while (!glfwWindowShouldClose(window) && d->time < sim_time)
    {
        double now = glfwGetTime();
        double elapsed = now - last_wall;
        last_wall = now;
        sync_keyboard_drive_command(window, elapsed);

        if (!g_paused && !freeze_init)
        {
            double target_time = d->time + elapsed;
            while (d->time < target_time && !glfwWindowShouldClose(window))
            {
                step_controller(m,
                                d,
                                map,
                                print_tick % 500 == 0,
                                zero_control,
                                zero_wheels,
                                invert_right_joints,
                                start_mode,
                                standup_time,
                                print_tick % g_controller_period_steps == 0,
                                &switched_to_safe);
                ++print_tick;
            }
        }

        mjv_updateScene(m, d, &g_option, 0, &g_camera, mjCAT_ALL, &g_scene);

        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
        mjr_render(viewport, &g_scene, &g_context);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    mjr_freeContext(&g_context);
    mjv_freeScene(&g_scene);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

int main(int argc, char **argv)
{
    const char *model_path = "sim/models/wheel_leg_urdf4_self_mesh_all.xml";
    int headless = 0;
    int zero_control = 0;
    int zero_wheels = 0;
    int invert_right_joints = 0;
    int freeze_init = 0;
    int debug_geometry = 0;
    int safe_mit_enabled = 1;
    int start_mode = CHASSIS_SAFE;
    double standup_time = 0.2;
    double sim_time = 100.0;
    double controller_period_ms = (double)CHASS_FSM_TIME;
    const char *init_key_name = "pos_debug_ground";
    const char *jump_telemetry_prefix = 0;
    const char *control_csv_path = 0;
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--headless") == 0)
        {
            headless = 1;
        }
        else if (strcmp(argv[i], "--zero-control") == 0)
        {
            zero_control = 1;
        }
        else if (strcmp(argv[i], "--zero-wheels") == 0)
        {
            zero_wheels = 1;
        }
        else if (strcmp(argv[i], "--safe-mit-off") == 0)
        {
            safe_mit_enabled = 0;
        }
        else if (strcmp(argv[i], "--safe-mit-on") == 0)
        {
            safe_mit_enabled = 1;
        }
        else if (strcmp(argv[i], "--control-period-ms") == 0 && i + 1 < argc)
        {
            controller_period_ms = atof(argv[++i]);
            if (!(controller_period_ms > 0.0))
            {
                fprintf(stderr, "--control-period-ms expects a positive value.\n");
                return 2;
            }
        }
        else if (strcmp(argv[i], "--control-csv") == 0)
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "--control-csv expects an output .csv path.\n");
                return 2;
            }
            control_csv_path = argv[++i];
        }
        else if (strcmp(argv[i], "--control-csv-period-ms") == 0 && i + 1 < argc)
        {
            g_control_csv_period_s = 0.001 * atof(argv[++i]);
            if (!(g_control_csv_period_s > 0.0))
            {
                fprintf(stderr, "--control-csv-period-ms expects a positive value.\n");
                return 2;
            }
        }
        else if (strcmp(argv[i], "--invert-right-joints") == 0)
        {
            invert_right_joints = 1;
        }
        else if (strcmp(argv[i], "--freeze-init") == 0)
        {
            freeze_init = 1;
        }
        else if (strcmp(argv[i], "--debug-geometry") == 0)
        {
            debug_geometry = 1;
        }
        else if (strcmp(argv[i], "--time") == 0 && i + 1 < argc)
        {
            sim_time = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--init-leg-length") == 0 && i + 1 < argc)
        {
            g_initial_leg_length = (float)atof(argv[++i]);
            if (g_initial_leg_length < MIN_LEG_LENGTH ||
                g_initial_leg_length > MAX_LEG_LENGTH)
            {
                fprintf(stderr,
                        "--init-leg-length expects %.3f..%.3f m.\n",
                        (double)MIN_LEG_LENGTH,
                        (double)MAX_LEG_LENGTH);
                return 2;
            }
        }
        else if (strcmp(argv[i], "--standup-time") == 0 && i + 1 < argc)
        {
            standup_time = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--forward-speed") == 0 && i + 1 < argc)
        {
            g_drive_command.forward_speed = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--auto-stop-time") == 0 && i + 1 < argc)
        {
            g_auto_stop_time = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--auto-steer-at") == 0 && i + 1 < argc)
        {
            g_auto_steer_time = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--auto-steer-duration") == 0 && i + 1 < argc)
        {
            g_auto_steer_duration = atof(argv[++i]);
            if (g_auto_steer_duration < 0.0)
            {
                g_auto_steer_duration = 0.0;
            }
        }
        else if (strcmp(argv[i], "--auto-steer-input") == 0 && i + 1 < argc)
        {
            g_auto_steer_input =
                (float)clamp_double(atof(argv[++i]), -1.0, 1.0);
        }
        else if (strcmp(argv[i], "--steer-leg-length") == 0 && i + 1 < argc)
        {
            g_steer_config.turn_leg_length = (float)atof(argv[++i]);
            if (g_steer_config.turn_leg_length < MIN_LEG_LENGTH ||
                g_steer_config.turn_leg_length > MAX_LEG_LENGTH)
            {
                fprintf(stderr,
                        "--steer-leg-length expects %.3f..%.3f m.\n",
                        (double)MIN_LEG_LENGTH,
                        (double)MAX_LEG_LENGTH);
                return 2;
            }
        }
        else if (strcmp(argv[i], "--steer-leg-rate") == 0 && i + 1 < argc)
        {
            g_steer_config.leg_rate = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--steer-yaw-rate") == 0 && i + 1 < argc)
        {
            g_steer_config.yaw_rate_max = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--steer-yaw-accel") == 0 && i + 1 < argc)
        {
            g_steer_config.yaw_accel_limit = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--steer-active-limit") == 0 && i + 1 < argc)
        {
            g_steer_config.active_time_limit = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--steer-planar-limit") == 0 && i + 1 < argc)
        {
            g_steer_config.active_planar_error_limit = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--steer-yaw-kp") == 0 && i + 1 < argc)
        {
            g_steer_yaw_kp = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--steer-pitch-target") == 0 && i + 1 < argc)
        {
            g_steer_pitch_target = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--steer-yaw-rate-kd") == 0 && i + 1 < argc)
        {
            g_steer_yaw_rate_kd = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--steer-yaw-torque-limit") == 0 && i + 1 < argc)
        {
            g_steer_yaw_torque_limit = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-at") == 0 && i + 1 < argc)
        {
            g_auto_jump_time = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-telemetry") == 0)
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr,
                        "--jump-telemetry expects an output path prefix.\n");
                return 2;
            }
            jump_telemetry_prefix = argv[++i];
        }
        else if (strcmp(argv[i], "--jump-thrust") == 0 && i + 1 < argc)
        {
            g_jump_thrust_ff = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-pitch-wheel-kp") == 0 && i + 1 < argc)
        {
            g_jump_pitch_wheel_kp = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-pitch-wheel-kd") == 0 && i + 1 < argc)
        {
            g_jump_pitch_wheel_kd = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-pitch-wheel-limit") == 0 && i + 1 < argc)
        {
            g_jump_pitch_wheel_limit = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-motor-wheel-limit") == 0 && i + 1 < argc)
        {
            g_jump_motor_wheel_limit = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-pitch-tp-kp") == 0 && i + 1 < argc)
        {
            g_jump_pitch_tp_kp = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-pitch-target") == 0 && i + 1 < argc)
        {
            g_jump_pitch_target = (float)atof(argv[++i]);
            g_jump_pitch_target_overridden = 1;
        }
        else if (strcmp(argv[i], "--jump-pitch-offset") == 0 && i + 1 < argc)
        {
            g_jump_pitch_offset = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-landing-pitch-target") == 0 && i + 1 < argc)
        {
            g_jump_landing_pitch_target = (float)atof(argv[++i]);
            g_jump_landing_pitch_target_overridden = 1;
        }
        else if (strcmp(argv[i], "--jump-recover-pitch-target") == 0 && i + 1 < argc)
        {
            g_jump_recover_pitch_target = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-landing-pitch-offset") == 0 && i + 1 < argc)
        {
            g_jump_landing_pitch_offset = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-pitch-tp-kd") == 0 && i + 1 < argc)
        {
            g_jump_pitch_tp_kd = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-pitch-tp-limit") == 0 && i + 1 < argc)
        {
            g_jump_pitch_tp_limit = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-skip-compression") == 0)
        {
            g_jump_compression_enabled = 0;
        }
        else if (strcmp(argv[i], "--jump-compress-target") == 0 && i + 1 < argc)
        {
            g_jump_compress_target = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-compress-rate") == 0 && i + 1 < argc)
        {
            g_jump_compress_rate = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-compress-support-scale") == 0 && i + 1 < argc)
        {
            g_jump_compress_support_scale = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-compress-tolerance") == 0 && i + 1 < argc)
        {
            g_jump_compress_tolerance = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-compress-hold") == 0 && i + 1 < argc)
        {
            g_jump_compress_hold_time = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-compress-timeout") == 0 && i + 1 < argc)
        {
            g_jump_compress_timeout = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-leg-swing-offset") == 0 && i + 1 < argc)
        {
            g_jump_leg_swing_offset = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-leg-swing-kp") == 0 && i + 1 < argc)
        {
            g_jump_leg_swing_kp = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-leg-swing-kd") == 0 && i + 1 < argc)
        {
            g_jump_leg_swing_kd = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-leg-swing-limit") == 0 && i + 1 < argc)
        {
            g_jump_leg_swing_limit = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-lqr-tp-weight") == 0 && i + 1 < argc)
        {
            g_jump_lqr_tp_weight = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-split-tp-weight") == 0 && i + 1 < argc)
        {
            g_jump_split_tp_weight = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-pitch-tp-weight") == 0 && i + 1 < argc)
        {
            g_jump_pitch_tp_weight = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-leg-swing-tp-weight") == 0 && i + 1 < argc)
        {
            g_jump_leg_swing_tp_weight = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-extend-l0") == 0 && i + 1 < argc)
        {
            g_jump_extend_l0 = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-extend-end-margin") == 0 && i + 1 < argc)
        {
            g_jump_extend_end_margin = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-extend-rate") == 0 && i + 1 < argc)
        {
            g_jump_extend_rate = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-preland-l0") == 0 && i + 1 < argc)
        {
            g_jump_preland_l0 = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-preland-clearance") == 0 && i + 1 < argc)
        {
            g_jump_preland_clearance = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-preland-rate") == 0 && i + 1 < argc)
        {
            g_jump_preland_rate = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-buffer-l0") == 0 && i + 1 < argc)
        {
            g_jump_buffer_l0 = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-buffer-rate") == 0 && i + 1 < argc)
        {
            g_jump_buffer_rate = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-buffer-support-scale") == 0 && i + 1 < argc)
        {
            g_jump_buffer_support_scale = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-buffer-pid-scale") == 0 && i + 1 < argc)
        {
            g_jump_buffer_pid_scale = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-preland-pid-scale") == 0 && i + 1 < argc)
        {
            g_jump_preland_pid_scale = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-landing-roll-kp") == 0 && i + 1 < argc)
        {
            g_jump_landing_roll_f0_kp = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-landing-roll-kd") == 0 && i + 1 < argc)
        {
            g_jump_landing_roll_f0_kd = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-landing-contact-kp") == 0 && i + 1 < argc)
        {
            g_jump_landing_contact_f0_kp = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-landing-balance-limit") == 0 && i + 1 < argc)
        {
            g_jump_landing_balance_f0_limit = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-landing-l0-kp") == 0 && i + 1 < argc)
        {
            g_jump_landing_roll_l0_kp = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-landing-l0-kd") == 0 && i + 1 < argc)
        {
            g_jump_landing_roll_l0_kd = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-landing-l0-limit") == 0 && i + 1 < argc)
        {
            g_jump_landing_balance_l0_limit = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-landing-clearance-kp") == 0 && i + 1 < argc)
        {
            g_jump_landing_clearance_l0_kp = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-landing-clearance-rate") == 0 && i + 1 < argc)
        {
            g_jump_landing_clearance_l0_rate = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-landing-clearance-limit") == 0 && i + 1 < argc)
        {
            g_jump_landing_clearance_l0_limit = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-landing-yaw-kp") == 0 && i + 1 < argc)
        {
            g_jump_landing_yaw_kp = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-takeoff-yaw-kp") == 0 && i + 1 < argc)
        {
            g_jump_takeoff_yaw_kp = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-takeoff-yaw-kd") == 0 && i + 1 < argc)
        {
            g_jump_takeoff_yaw_kd = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-takeoff-yaw-limit") == 0 && i + 1 < argc)
        {
            g_jump_takeoff_yaw_limit = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-landing-yaw-kd") == 0 && i + 1 < argc)
        {
            g_jump_landing_yaw_kd = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-landing-yaw-limit") == 0 && i + 1 < argc)
        {
            g_jump_landing_yaw_limit = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-landing-yaw-start-phase") == 0 && i + 1 < argc)
        {
            g_jump_landing_yaw_start_phase = atoi(argv[++i]);
            if (g_jump_landing_yaw_start_phase < 3)
            {
                g_jump_landing_yaw_start_phase = 3;
            }
            if (g_jump_landing_yaw_start_phase > 5)
            {
                g_jump_landing_yaw_start_phase = 5;
            }
        }
        else if (strcmp(argv[i], "--jump-wheel-recover-blend-time") == 0 && i + 1 < argc)
        {
            g_jump_wheel_recover_blend_time = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-tuck-kp") == 0 && i + 1 < argc)
        {
            g_jump_tuck_kp = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-tuck-kd") == 0 && i + 1 < argc)
        {
            g_jump_tuck_kd = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--jump-tuck-limit") == 0 && i + 1 < argc)
        {
            g_jump_tuck_torque_limit = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--drive") == 0 && i + 1 < argc)
        {
            const char *drive = argv[++i];
            if (strcmp(drive, "stand") == 0)
            {
                queue_drive_mode(&g_drive_command, DRIVE_STAND);
            }
            else if (strcmp(drive, "forward") == 0)
            {
                queue_drive_mode(&g_drive_command, DRIVE_FORWARD);
            }
            else if (strcmp(drive, "jump") == 0)
            {
                queue_drive_mode(&g_drive_command, DRIVE_STAND);
                g_auto_jump_time = 1.0;
            }
            else
            {
                fprintf(stderr, "--drive expects stand, forward, or jump.\n");
                return 2;
            }
        }
        else if (strcmp(argv[i], "--free-yaw") == 0)
        {
            g_drive_command.yaw_lock = 0;
        }
        else if (strcmp(argv[i], "--no-wheel-override") == 0)
        {
            g_use_wheel_balance_override = 0;
        }
        else if (strcmp(argv[i], "--wheel-override") == 0)
        {
            g_use_wheel_balance_override = 1;
        }
        else if ((strcmp(argv[i], "--override-pitch-kp") == 0 ||
                  strcmp(argv[i], "--stand-pitch-kp") == 0) &&
                 i + 1 < argc)
        {
            g_balance_pitch_kp = (float)atof(argv[++i]);
        }
        else if ((strcmp(argv[i], "--override-pitch-target") == 0 ||
                  strcmp(argv[i], "--stand-pitch-target") == 0) &&
                 i + 1 < argc)
        {
            g_balance_pitch_target = (float)atof(argv[++i]);
        }
        else if ((strcmp(argv[i], "--override-pitch-kd") == 0 ||
                  strcmp(argv[i], "--stand-pitch-kd") == 0) &&
                 i + 1 < argc)
        {
            g_balance_pitch_kd = (float)atof(argv[++i]);
        }
        else if ((strcmp(argv[i], "--override-pos-kp") == 0 ||
                  strcmp(argv[i], "--stand-pos-kp") == 0) &&
                 i + 1 < argc)
        {
            g_balance_pos_kp = (float)atof(argv[++i]);
        }
        else if ((strcmp(argv[i], "--override-vel-kd") == 0 ||
                  strcmp(argv[i], "--stand-vel-kd") == 0) &&
                 i + 1 < argc)
        {
            g_balance_vel_kd = (float)atof(argv[++i]);
        }
        else if ((strcmp(argv[i], "--override-pos-ramp-time") == 0 ||
                  strcmp(argv[i], "--stand-pos-ramp-time") == 0) &&
                 i + 1 < argc)
        {
            g_balance_pos_ramp_time = (float)atof(argv[++i]);
        }
        else if ((strcmp(argv[i], "--override-drive-kff") == 0 ||
                  strcmp(argv[i], "--stand-drive-kff") == 0) &&
                 i + 1 < argc)
        {
            g_balance_drive_kff = (float)atof(argv[++i]);
        }
        else if ((strcmp(argv[i], "--override-yaw-kp") == 0 ||
                  strcmp(argv[i], "--stand-yaw-kp") == 0) &&
                 i + 1 < argc)
        {
            g_balance_yaw_kp = (float)atof(argv[++i]);
        }
        else if ((strcmp(argv[i], "--override-yaw-kd") == 0 ||
                  strcmp(argv[i], "--stand-yaw-kd") == 0) &&
                 i + 1 < argc)
        {
            g_balance_yaw_kd = (float)atof(argv[++i]);
        }
        else if ((strcmp(argv[i], "--override-wheel-limit") == 0 ||
                  strcmp(argv[i], "--stand-wheel-limit") == 0) &&
                 i + 1 < argc)
        {
            g_balance_wheel_limit = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
        {
            const char *mode = argv[++i];
            if (strcmp(mode, "stand") == 0)
            {
                start_mode = CHASSIS_STAND_UP;
            }
            else if (strcmp(mode, "safe") == 0)
            {
                start_mode = CHASSIS_SAFE;
            }
            else if (strcmp(mode, "off") == 0)
            {
                start_mode = CHASSIS_OFF;
            }
            else
            {
                fprintf(stderr, "--mode expects stand, safe, or off.\n");
                return 2;
            }
        }
        else if (strcmp(argv[i], "--joint-signs") == 0 && i + 1 < argc)
        {
            const char *signs = argv[++i];
            if (strlen(signs) != 4)
            {
                fprintf(stderr, "--joint-signs expects four characters, for example +-+-.\n");
                return 2;
            }
            for (int j = 0; j < 4; ++j)
            {
                if (signs[j] == '+')
                {
                    g_joint_ctrl_sign[j] = 1.0;
                }
                else if (signs[j] == '-')
                {
                    g_joint_ctrl_sign[j] = -1.0;
                }
                else
                {
                    fprintf(stderr, "--joint-signs only accepts + or - characters.\n");
                    return 2;
                }
            }
        }


        else if (strcmp(argv[i], "--init-key") == 0 && i + 1 < argc)
        {
            init_key_name = argv[++i];
        }
        else if (strcmp(argv[i], "--hang-init") == 0)
        {
            init_key_name = "pos_debug_hang";
        }
        else if (strcmp(argv[i], "--ground-init") == 0)
        {
            init_key_name = "pos_debug_ground";
        }
        else
        {
            model_path = argv[i];
        }
    }

    char error[1024] = {0};
    model_path = resolve_default_model_path(model_path);

    mjModel *m = mj_loadXML(model_path, 0, error, sizeof(error));
    if (m == 0)
    {
        fprintf(stderr, "Failed to load MuJoCo model: %s\n%s\n", model_path, error);
        return 1;
    }

    const double requested_controller_period_s = 0.001 * controller_period_ms;
    g_controller_period_steps =
        (int)llround(requested_controller_period_s / m->opt.timestep);
    if (g_controller_period_steps < 1 ||
        fabs(g_controller_period_steps * m->opt.timestep -
             requested_controller_period_s) > 1.0e-9)
    {
        fprintf(stderr,
                "Control period %.6f ms is not an integer multiple of the MuJoCo timestep %.6f ms.\n",
                controller_period_ms,
                1000.0 * m->opt.timestep);
        mj_deleteModel(m);
        return 2;
    }
    g_controller_dt_s =
        (float)(g_controller_period_steps * m->opt.timestep);

    mjData *d = mj_makeData(m);
    if (d == 0)
    {
        fprintf(stderr, "Failed to allocate MuJoCo data.\n");
        mj_deleteModel(m);
        return 1;
    }

    ModelMap map;
    build_model_map(m, &map);
    /* The closed-chain export's visual front is opposite to the stable controller +x convention. */
    g_visual_forward_to_controller_sign = map.rotate_control_frame ? -1.0f : 1.0f;
    g_drive_command.forward_speed =
        visual_to_controller_speed(g_drive_command.forward_speed);
    for (int i = 0; i < 2; ++i)
    {
        const mjtNum axis_y = m->jnt_axis[3 * map.wheel[i].id + 1];
        g_wheel_ctrl_sign[i] = axis_y > 0.0 ? -1.0 : 1.0;
    }
    apply_model_initial_pose(m, d, &map, init_key_name);
    if (debug_geometry)
    {
        print_geometry_debug(m, d, &map, "init");
    }
    SimControllerState initial_state;
    read_state(m, d, &map, &initial_state);
    g_xml_initial_rpy[0] = initial_state.roll;
    g_xml_initial_rpy[1] = initial_state.pitch;
    g_xml_initial_rpy[2] = initial_state.yaw;
    if (!g_jump_pitch_target_overridden)
    {
        g_jump_pitch_target = initial_state.pitch + g_jump_pitch_offset;
    }
    if (!g_jump_landing_pitch_target_overridden)
    {
        g_jump_landing_pitch_target =
            initial_state.pitch + g_jump_landing_pitch_offset;
    }
    for (int i = 0; i < 4; ++i)
    {
        g_airborne_pose_target[i] = initial_state.joint_pos[i];
    }
    g_xml_initial_qpos = (mjtNum *)malloc(sizeof(mjtNum) * m->nq);
    if (g_xml_initial_qpos == 0)
    {
        fprintf(stderr, "Failed to allocate XML initial pose snapshot.\n");
        mj_deleteData(d);
        mj_deleteModel(m);
        return 1;
    }
    memcpy(g_xml_initial_qpos, d->qpos, sizeof(mjtNum) * m->nq);

    SimController_Init();
    SimController_SetSafeMitEnabled(safe_mit_enabled);
#if MUJOCO_VMC_KINEMATICS_MODE == MUJOCO_VMC_MODEL_CAD_ONE_CLOSURE_ID
    printf("VMC kinematics backend: CAD_ONE_CLOSURE (HTML 4.3/5.5), raw q=[front,rear]\n");
#else
    printf("VMC kinematics backend: LEGACY_IDEAL_FIVE_BAR\n");
#endif
    SteerStateMachine_Init(&g_steer_machine, &g_steer_config);
    SimController_SetAirbornePoseTarget(g_airborne_pose_target);
    SimController_SetAirbornePoseGains(g_jump_tuck_kp,
                                       g_jump_tuck_kd,
                                       g_jump_tuck_torque_limit);
    SimController_SetJumpThrust(g_jump_thrust_ff);
    SimController_SetJumpPitchTp(g_jump_pitch_target,
                                 g_jump_pitch_tp_kp,
                                 g_jump_pitch_tp_kd,
                                 g_jump_pitch_tp_limit);
    SimController_SetJumpCompression(g_jump_compression_enabled,
                                     g_jump_compress_target,
                                     g_jump_compress_rate,
                                     g_jump_compress_support_scale,
                                     g_jump_compress_tolerance,
                                     g_jump_compress_hold_time,
                                     g_jump_compress_timeout);
    SimController_SetJumpLegSwing(g_jump_leg_swing_offset,
                                  g_jump_leg_swing_kp,
                                  g_jump_leg_swing_kd,
                                  g_jump_leg_swing_limit);
    SimController_SetJumpTpWeights(g_jump_lqr_tp_weight,
                                   g_jump_split_tp_weight,
                                   g_jump_pitch_tp_weight,
                                   g_jump_leg_swing_tp_weight);
    SimController_SetJumpExtend(g_jump_extend_l0,
                                g_jump_extend_end_margin,
                                g_jump_extend_rate);
    SimController_SetJumpLandingLegLengths(g_jump_preland_l0,
                                           g_jump_buffer_l0);
    SimController_SetJumpPrelandClearance(g_jump_preland_clearance);
    SimController_SetJumpPrelandPidScale(g_jump_preland_pid_scale);
    SimController_SetJumpLandingDynamics(g_jump_preland_rate,
                                         g_jump_buffer_rate,
                                         g_jump_buffer_support_scale,
                                         g_jump_buffer_pid_scale);
    SimController_SetJumpLandingBalance(g_jump_landing_roll_f0_kp,
                                        g_jump_landing_roll_f0_kd,
                                        g_jump_landing_contact_f0_kp,
                                        g_jump_landing_balance_f0_limit);
    SimController_SetJumpLandingL0Balance(g_jump_landing_roll_l0_kp,
                                          g_jump_landing_roll_l0_kd,
                                          g_jump_landing_balance_l0_limit);
    SimController_SetJumpLandingClearanceBalance(
        g_jump_landing_clearance_l0_kp,
        g_jump_landing_clearance_l0_rate,
        g_jump_landing_clearance_l0_limit);
    SimController_SetJumpWheelRecoverBlendTime(g_jump_wheel_recover_blend_time);
    SimController_SetMode(start_mode);
    if (jump_telemetry_prefix != 0)
    {
        g_jump_telemetry =
            JumpTelemetry_Create(jump_telemetry_prefix,
                                 TAKE_OFF_FN_THRESHOLD);
        if (g_jump_telemetry == 0)
        {
            fprintf(stderr,
                    "Failed to initialize jump telemetry: %s\n",
                    jump_telemetry_prefix);
            free(g_xml_initial_qpos);
            g_xml_initial_qpos = 0;
            mj_deleteData(d);
            mj_deleteModel(m);
            return 1;
        }
        printf("Jump telemetry enabled: %s.[csv|svg]\n",
               jump_telemetry_prefix);
    }
    if (!open_control_csv(control_csv_path))
    {
        if (g_jump_telemetry != 0)
        {
            JumpTelemetry_Destroy(g_jump_telemetry);
            g_jump_telemetry = 0;
        }
        free(g_xml_initial_qpos);
        g_xml_initial_qpos = 0;
        mj_deleteData(d);
        mj_deleteModel(m);
        return 1;
    }
    if (control_csv_path != 0)
    {
        printf("Control CSV enabled: %s (period %.3f ms)\n",
               control_csv_path,
               1000.0 * g_control_csv_period_s);
    }
    g_drive_command.current_speed = 0.0f;
    g_drive_command.leg_set = g_initial_leg_length;
    g_drive_command.roll_set = INIT_ROLL;
    g_drive_command.position_hold_blend = start_mode == CHASSIS_STAND_UP ? 0.0f : 1.0f;
    g_drive_command.hold_position_pending = 1;
    g_drive_command.hold_yaw_pending = 1;
    printf("Drive mode: %s | visual_forward_speed=%.3f | controller_forward_speed=%.3f | visual_to_ctrl_sign=%.0f | yaw_lock=%d | wheel_sign=[%.0f %.0f] | wheel_override=%d\n",
           drive_command_name(&g_drive_command),
           controller_to_visual_speed(g_drive_command.forward_speed),
           g_drive_command.forward_speed,
           g_visual_forward_to_controller_sign,
           g_drive_command.yaw_lock,
           g_wheel_ctrl_sign[0],
           g_wheel_ctrl_sign[1],
           g_use_wheel_balance_override);
    printf("Controller schedule: physics_dt=%.3f ms control_dt=%.3f ms hold_steps=%d | SAFE_MIT=%s\n",
           1000.0 * m->opt.timestep,
           1000.0 * (double)g_controller_dt_s,
           g_controller_period_steps,
           safe_mit_enabled ? "on" : "off");
    printf("Wheel override gains: pitch_target=%.3f pitch_kp=%.3f pitch_kd=%.3f pos_kp=%.3f vel_kd=%.3f pos_ramp=%.3f drive_kff=%.3f yaw_kp=%.3f yaw_kd=%.3f limit=%.3f\n",
           g_balance_pitch_target,
           g_balance_pitch_kp,
           g_balance_pitch_kd,
           g_balance_pos_kp,
           g_balance_vel_kd,
           g_balance_pos_ramp_time,
           g_balance_drive_kff,
           g_balance_yaw_kp,
           g_balance_yaw_kd,
           g_balance_wheel_limit);
    printf("Steer override: pitch_target=%.3f yaw_kp=%.3f yaw_rate_kd=%.3f yaw_torque_limit=%.3f\n",
           g_steer_pitch_target,
           g_steer_yaw_kp,
           g_steer_yaw_rate_kd,
           g_steer_yaw_torque_limit);
    printf("Steer state machine: turn_l0=%.3f leg_rate=%.3f yaw_rate=%.3f yaw_accel=%.3f active_limit=%.3f planar_limit=%.3f input_deadband=%.3f brake_gyro=%.3f brake_v=%.3f brake_hold=%.3f recover_v=%.3f abort_rp=[%.3f %.3f]\n",
           g_steer_config.turn_leg_length,
           g_steer_config.leg_rate,
           g_steer_config.yaw_rate_max,
           g_steer_config.yaw_accel_limit,
           g_steer_config.active_time_limit,
           g_steer_config.active_planar_error_limit,
           g_steer_config.input_deadband,
           g_steer_config.brake_gyro_tolerance,
           g_steer_config.brake_linear_velocity_tolerance,
           g_steer_config.brake_hold_time,
           g_steer_config.recover_linear_velocity_tolerance,
           g_steer_config.abort_roll_limit,
           g_steer_config.abort_pitch_limit);
    if (g_auto_steer_time >= 0.0 && g_auto_steer_duration > 0.0)
    {
        printf("Auto steer: start=%.3f duration=%.3f input=%.3f\n",
               g_auto_steer_time,
               g_auto_steer_duration,
               g_auto_steer_input);
    }
    printf("Jump thrust feedforward: %.3f\n", g_jump_thrust_ff);
    printf("Motor torque limits: joint_normal=%.3f joint_jump=%.3f wheel_normal=%.3f wheel_jump=%.3f Nm\n",
           MAX_JOINT_TORQUE,
           MAX_JOINT_TORQUE_JUMP,
           LK_MAX_MF_TORQUE,
           fabsf(g_jump_motor_wheel_limit));
    printf("Jump pitch wheel hold: kp=%.3f kd=%.3f limit=%.3f\n",
           g_jump_pitch_wheel_kp,
           g_jump_pitch_wheel_kd,
           g_jump_pitch_wheel_limit);
    printf("Jump landing pitch target: %.3f\n",
           g_jump_landing_pitch_target);
    printf("Jump recover pitch target: %.3f\n",
           g_jump_recover_pitch_target);
    printf("Jump takeoff yaw damping: kp=%.3f kd=%.3f limit=%.3f\n",
           g_jump_takeoff_yaw_kp,
           g_jump_takeoff_yaw_kd,
           g_jump_takeoff_yaw_limit);
    printf("Jump pitch leg Tp hold: target=%.3f kp=%.3f kd=%.3f limit=%.3f\n",
           g_jump_pitch_target,
           g_jump_pitch_tp_kp,
           g_jump_pitch_tp_kd,
           g_jump_pitch_tp_limit);
    printf("Jump compression: enabled=%d target=%.3f rate=%.3f support=%.3f tolerance=%.3f hold=%.3f timeout=%.3f\n",
           g_jump_compression_enabled,
           g_jump_compress_target,
           g_jump_compress_rate,
           g_jump_compress_support_scale,
           g_jump_compress_tolerance,
           g_jump_compress_hold_time,
           g_jump_compress_timeout);
    printf("Jump leg swing: offset=%.3f kp=%.3f kd=%.3f limit=%.3f\n",
           g_jump_leg_swing_offset,
           g_jump_leg_swing_kp,
           g_jump_leg_swing_kd,
           g_jump_leg_swing_limit);
    printf("Jump Tp weights: lqr=%.3f split=%.3f pitch=%.3f leg_swing=%.3f\n",
           g_jump_lqr_tp_weight,
           g_jump_split_tp_weight,
           g_jump_pitch_tp_weight,
           g_jump_leg_swing_tp_weight);
    printf("Jump extend: target_l0=%.3f end_margin=%.3f rate=%.3f\n",
           g_jump_extend_l0,
           g_jump_extend_end_margin,
           g_jump_extend_rate);
    printf("Jump landing leg lengths: clearance=%.3f preland_l0=%.3f preland_rate=%.3f preland_pid_scale=%.3f buffer_l0=%.3f buffer_rate=%.3f buffer_support=%.3f buffer_pid=%.3f\n",
           g_jump_preland_clearance,
           g_jump_preland_l0,
           g_jump_preland_rate,
           g_jump_preland_pid_scale,
           g_jump_buffer_l0,
           g_jump_buffer_rate,
           g_jump_buffer_support_scale,
           g_jump_buffer_pid_scale);
    printf("Jump landing balance: roll_kp=%.3f roll_kd=%.3f contact_kp=%.5f limit=%.3f\n",
           g_jump_landing_roll_f0_kp,
           g_jump_landing_roll_f0_kd,
           g_jump_landing_contact_f0_kp,
           g_jump_landing_balance_f0_limit);
    printf("Jump landing L0 balance: roll_kp=%.3f roll_kd=%.3f limit=%.3f\n",
           g_jump_landing_roll_l0_kp,
           g_jump_landing_roll_l0_kd,
           g_jump_landing_balance_l0_limit);
    printf("Jump phase-4 clearance L0 balance: kp=%.3f rate=%.3f limit=%.3f\n",
           g_jump_landing_clearance_l0_kp,
           g_jump_landing_clearance_l0_rate,
           g_jump_landing_clearance_l0_limit);
    printf("Jump yaw wheel balance: start_phase=%d kp=%.3f kd=%.3f limit=%.3f\n",
           g_jump_landing_yaw_start_phase,
           g_jump_landing_yaw_kp,
           g_jump_landing_yaw_kd,
           g_jump_landing_yaw_limit);
    printf("Jump wheel recover blend time: %.3f\n",
           g_jump_wheel_recover_blend_time);
    printf("Jump XML-pose tuck: kp=%.3f kd=%.3f limit=%.3f target=[%.3f %.3f %.3f %.3f]\n",
           g_jump_tuck_kp,
           g_jump_tuck_kd,
           g_jump_tuck_torque_limit,
           g_airborne_pose_target[0],
           g_airborne_pose_target[1],
           g_airborne_pose_target[2],
           g_airborne_pose_target[3]);
    printf("XML initial base RPY target: [%.3f %.3f %.3f]\n",
           g_xml_initial_rpy[0],
           g_xml_initial_rpy[1],
           g_xml_initial_rpy[2]);
    if (start_mode == CHASSIS_STAND_UP && standup_time >= 0.0)
    {
        printf("Start in STAND_UP, auto switch to SAFE after %.3f s.\n", standup_time);
    }

    int result = headless ? run_headless(m, d, &map, sim_time, zero_control, zero_wheels, invert_right_joints, freeze_init, start_mode, standup_time) : run_viewer(m, d, &map, sim_time, zero_control, zero_wheels, invert_right_joints, freeze_init, start_mode, standup_time);
    if (g_body_z_range_valid)
    {
        printf("Body z range: min=%.3f max=%.3f rise=%.3f\n",
               g_body_z_min,
               g_body_z_max,
               g_body_z_max - g_body_z_min);
    }
    if (g_steer_metrics_valid)
    {
        printf("Steer summary: duration=%.3f yaw_delta=%.3f max_abs_roll=%.3f max_abs_pitch=%.3f max_abs_yaw_rate=%.3f max_planar_error=%.3f max_planar_speed=%.3f\n",
               g_steer_metrics_time,
               (float)wrap_pi((double)g_steer_yaw_end -
                              (double)g_steer_yaw_start),
               g_steer_abs_roll_max,
               g_steer_abs_pitch_max,
               g_steer_abs_yaw_rate_max,
               g_steer_planar_error_max,
               g_steer_planar_speed_max);
    }
    if (g_airborne_metrics_valid)
    {
        printf("Wheel clearance: min=%.3f max=%.3f | Airborne: total=%.3f max_span=%.3f max_abs_rpy=[%.3f %.3f %.3f]\n",
               g_wheel_clearance_min,
               g_wheel_clearance_max,
               g_airborne_time,
               g_airborne_max_duration,
               g_airborne_abs_roll_max,
               g_airborne_abs_pitch_max,
               g_airborne_abs_yaw_error_max);
        printf("Airborne final XML-base RPY error: [%.4f %.4f %.4f]\n",
               g_airborne_rpy_last_error[0],
               g_airborne_rpy_last_error[1],
               g_airborne_rpy_last_error[2]);
        if (g_airborne_joint_error_samples > 0)
        {
            printf("Airborne XML-pose joint error: rms=%.4f max_abs=[%.4f %.4f %.4f %.4f]\n",
                   sqrt(g_airborne_joint_error_sq_sum / (double)g_airborne_joint_error_samples),
                   g_airborne_joint_abs_max[0],
                   g_airborne_joint_abs_max[1],
                   g_airborne_joint_abs_max[2],
                   g_airborne_joint_abs_max[3]);
            printf("Airborne XML-pose convergence: best_rms=%.4f best=[%.4f %.4f %.4f %.4f] last=[%.4f %.4f %.4f %.4f]\n",
                   g_airborne_joint_best_rms,
                   g_airborne_joint_best_error[0],
                   g_airborne_joint_best_error[1],
                   g_airborne_joint_best_error[2],
                   g_airborne_joint_best_error[3],
                   g_airborne_joint_last_error[0],
                   g_airborne_joint_last_error[1],
                   g_airborne_joint_last_error[2],
                   g_airborne_joint_last_error[3]);
            printf("Airborne full-link XML shape: best_rms=%.4f last_rms=%.4f max_abs=%.4f\n",
                   g_airborne_full_pose_best_rms,
                   g_airborne_full_pose_last_rms,
                   g_airborne_full_pose_max_abs);
        }
    }
    if (g_jump_attitude_valid)
    {
        printf("Jump attitude: duration=%.3f max_abs_rp=[%.3f %.3f] pitch_range=[%.3f@%.3f %.3f@%.3f]\n",
               g_jump_attitude_time,
               g_jump_abs_roll_max,
               g_jump_abs_pitch_max,
               g_jump_pitch_min,
               g_jump_pitch_min_time,
               g_jump_pitch_max,
               g_jump_pitch_max_time);
        printf("Jump base-floor contact peak: %.3f N @ %.3f s\n",
               g_jump_base_contact_peak_n,
               g_jump_base_contact_peak_time);
    }
    if (g_jump_telemetry != 0)
    {
        if (!JumpTelemetry_Write(g_jump_telemetry))
        {
            result = result == 0 ? 1 : result;
        }
        JumpTelemetry_Destroy(g_jump_telemetry);
        g_jump_telemetry = 0;
    }
    if (g_control_csv != 0)
    {
        if (fclose(g_control_csv) != 0)
        {
            fprintf(stderr, "Failed to finalize control CSV.\n");
            result = result == 0 ? 1 : result;
        }
        g_control_csv = 0;
    }

    free(g_xml_initial_qpos);
    g_xml_initial_qpos = 0;
    mj_deleteData(d);
    mj_deleteModel(m);
    return result;
}
