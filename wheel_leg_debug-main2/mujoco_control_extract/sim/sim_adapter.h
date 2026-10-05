#ifndef SIM_ADAPTER_H
#define SIM_ADAPTER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    float joint_pos[4];       // controller motor coordinates, rad
    float joint_vel[4];       // rad/s
    float joint_torque_fdb[4]; // Nm

    float wheel_vel[2];       // rad/s

    float roll;               // rad
    float pitch;              // rad
    float yaw;                // rad, continuous if possible
    float gyro[3];            // rad/s, x/y/z
    float accel_body[3];      // m/s^2, optional
    float accel_world[3];     // m/s^2, optional

    float body_x;             // m
    float body_y;             // m
    float body_z;             // m
    float body_v;             // m/s
    float body_v_y;           // m/s
    float body_z_vel;         // m/s
} SimControllerState;

typedef struct
{
    float joint_torque[4];    // Nm, same order as DM_8009_Motor[0..3]
    float wheel_torque[2];    // Nm, same order as LK_9025_Motor[0..1]
} SimControllerOutput;

typedef struct
{
    float l0[2];
    float l0_set[2];
    float d_l0[2];
    float phi0[2];
    float d_phi0[2];
    float theta[2];
    float d_theta[2];
    float f0_total[2];
    float f0_gravity[2];
    float f0_leg_pid[2];
    float f0_jump_ff[2];
    float f0_balance[2];
    float tp_total[2];
    float tp_lqr[2];
    float tp_split[2];
    float tp_pitch[2];
    float tp_leg_swing[2];
    float joint_torque_f0[4];
    float joint_torque_tp[4];
    float joint_torque_vmc[4];
    float joint_torque_mit[4];
    float tp_weight[4]; // lqr, split, pitch, leg swing
} SimControllerControlBreakdown;

void SimController_Init(void);
void SimController_SetStandL0Pitch(float l0_pitch);
void SimController_SetSafeMitEnabled(int enabled);
void SimController_SetState(const SimControllerState *state);
void SimController_SetCommand(float v_set, float x_set, float leg_set, float roll_set, float yaw_set);
void SimController_SetDriveContext(int drive_forward, float position_hold_blend, int yaw_lock);
void SimController_SetJumpThrust(float thrust_ff);
void SimController_SetJumpPitchTp(float target, float kp, float kd, float limit);
void SimController_SetJumpCompression(int enabled,
                                      float target,
                                      float rate,
                                      float support_scale,
                                      float tolerance,
                                      float hold_time,
                                      float timeout);
void SimController_SetJumpLegSwing(float offset, float kp, float kd, float limit);
void SimController_SetJumpTpWeights(float lqr_weight,
                                    float split_weight,
                                    float pitch_weight,
                                    float leg_swing_weight);
void SimController_SetJumpExtend(float extend_l0,
                                 float end_margin,
                                 float extend_rate);
void SimController_SetJumpLandingLegLengths(float preland_l0, float buffer_l0);
void SimController_SetJumpPrelandClearance(float clearance);
void SimController_SetJumpPrelandPidScale(float pid_scale);
void SimController_SetJumpLandingDynamics(float preland_rate,
                                          float buffer_rate,
                                          float buffer_support_scale,
                                          float buffer_pid_scale);
void SimController_SetJumpLandingBalance(float roll_kp,
                                         float roll_kd,
                                         float contact_kp,
                                         float limit);
void SimController_SetJumpLandingL0Balance(float roll_kp,
                                           float roll_kd,
                                           float limit);
void SimController_SetJumpLandingClearanceBalance(float kp,
                                                  float rate,
                                                  float limit);
void SimController_SetJumpWheelRecoverBlendTime(float blend_time);
void SimController_SetAirbornePoseTarget(const float joint_pos[4]);
void SimController_SetAirbornePoseGains(float kp, float kd, float torque_limit);
void SimController_SetAirborne(int airborne);
void SimController_SetFlightObservation(int airborne,
                                        const float wheel_clearance_m[2],
                                        const float wheel_contact_normal_n[2]);
void SimController_SetMode(int mode);
int SimController_RequestJump(void);
int SimController_IsJumping(void);
float SimController_GetJumpWheelBalanceBlend(void);
void SimController_Step(float dt);
void SimController_GetOutput(SimControllerOutput *output);
void SimController_GetControlBreakdown(SimControllerControlBreakdown *breakdown);

#ifdef __cplusplus
}
#endif

#endif
