
// PID控制器结构体
typedef struct
{
    float kp, ki, kd;      // PID参数
    float integral;         // 积分项
    float prev_error;       // 上一次误差
    float output_limit;     // 输出限幅值
} PIDController;

// LESO观测器结构体
typedef struct
{
    float beta1, beta2;    // 观测器增益
    float b0;              // 系统增益估计值
    float z1, z2;          // 观测状态 [z1=系统状态, z2=总扰动]
    float dt;              // 采样时间
} LESO;

// 串级PID+LESO控制器
typedef struct
{
    PIDController outer;   // 外环PID (位置/角度)
    PIDController inner;   // 内环PID (速度/角速度)
    LESO leso;             // 扰动观测器
    float dt;              // 控制周期
} CascadePID_LESO;

// 初始化PID控制器
void PID_Init(PIDController *pid, float kp, float ki, float kd, float output_limit)
{
    pid->kp = kp;
    pid->ki = ki;
    pid->kd = kd;
    pid->integral = 0.0f;
    pid->prev_error = 0.0f;
    pid->output_limit = output_limit;
}

// PID计算（位置式）
float PID_Calculate(PIDController *pid, float setpoint, float measurement)
{
    // 计算误差
    float error = setpoint - measurement;
    
    // 积分项
    pid->integral += error * pid->ki;
    // 积分限幅
    if (pid->integral > pid->output_limit) pid->integral = pid->output_limit;
    if (pid->integral < -pid->output_limit) pid->integral = -pid->output_limit;
    
    // 微分项 (使用微分先行减少冲击)
    float derivative = (error - pid->prev_error);
    pid->prev_error = error;
    
    // PID输出
    float output = pid->kp * error + pid->integral + pid->kd * derivative;
    
    // 输出限幅
    if (output > pid->output_limit) output = pid->output_limit;
    if (output < -pid->output_limit) output = -pid->output_limit;
    
    return output;
}

// 初始化LESO
void LESO_Init(LESO *leso, float beta1, float beta2, float b0, float dt)
{
    leso->beta1 = beta1;
    leso->beta2 = beta2;
    leso->b0 = b0;
    leso->z1 = 0.0f;
    leso->z2 = 0.0f;
    leso->dt = dt;
}

// LESO更新和扰动估计
void LESO_Update(LESO *leso, float u, float y)
{
    // 计算观测误差
    float e = leso->z1 - y;
    
    // 更新状态观测 (欧拉积分)
    leso->z1 += leso->dt * (leso->z2 - leso->beta1 * e + leso->b0 * u);
    leso->z2 += leso->dt * (-leso->beta2 * e);
}

// 初始化串级控制器
void CascadePID_LESO_Init(CascadePID_LESO *ctrl, 
                         float outer_kp, float outer_ki, float outer_kd,
                         float inner_kp, float inner_ki, float inner_kd,
                         float leso_beta1, float leso_beta2, float leso_b0,
                         float output_limit, float dt)
{
    // 初始化PID控制器
    PID_Init(&ctrl->outer, outer_kp, outer_ki, outer_kd, output_limit);
    PID_Init(&ctrl->inner, inner_kp, inner_ki, inner_kd, output_limit);
    
    // 初始化LESO
    LESO_Init(&ctrl->leso, leso_beta1, leso_beta2, leso_b0, dt);
    ctrl->dt = dt;
}

// 串级PID+LESO控制计算
float CascadePID_LESO_Control(CascadePID_LESO *ctrl, float outer_setpoint, float outer_measurement, float inner_measurement)
{
    // 外环PID计算（生成速度指令）
    float inner_setpoint = PID_Calculate(&ctrl->outer, outer_setpoint, outer_measurement);
    
    // 内环PID计算（原始控制量）
    float u0 = PID_Calculate(&ctrl->inner, inner_setpoint, inner_measurement);
    
    // LESO扰动补偿
    float d = ctrl->leso.z2;  // 观测到的总扰动
    float u = (u0 - d) / ctrl->leso.b0;
    
    // 更新LESO（使用补偿后的控制量）
    LESO_Update(&ctrl->leso, u, inner_measurement);
    
    return u;
}

int main()
{
    // 控制器参数设置示例（需根据实际系统调整）
    CascadePID_LESO ctrl;
    CascadePID_LESO_Init(&ctrl,
                         /* 外环PID */ 1.5, 0.05, 0.2,
                         /* 内环PID */ 0.8, 0.01, 0.1,
                         /* LESO */ 100, 500, 1.2,
                         /* 输出限幅 */ 500,
                         /* 采样时间 */ 0.01);
    
    // 模拟控制循环
    float position_setpoint = 10.0f;  // 目标位置
    float position = 0.0f;            // 实际位置
    float velocity = 0.0f;             // 实际速度
    int i = 0;
    for(i = 0; i < 1000; i++) {
        // 执行控制计算（位置环+速度环+LESO）
        float u = CascadePID_LESO_Control(&ctrl, position_setpoint, position, velocity);
        
        // 简单系统模型模拟（二阶系统）
        float acceleration = u * 0.8 - velocity * 0.2;  // 包含阻尼项
        velocity += acceleration * ctrl.dt;
        position += velocity * ctrl.dt;
        
        // 添加扰动（测试LESO抗扰能力）
        if(i == 500) {
            velocity += 2.0f;  // 突加扰动
        }
        
        // 打印结果
        if(i % 50 == 0) {
            printf("Time=%04d ms | Setpoint=%.2f | Position=%.2f | Velocity=%.2f | Control=%.2f\n",
                   i*10, position_setpoint, position, velocity, u);
        }
    }
    
    return 0;
}
