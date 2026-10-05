#include <stdio.h>

// 定义LESO结构体
typedef struct {
    double z1;     // 状态1估计值（系统输出）
    double z2;     // 状态2估计值（系统状态导数）
    double z3;     // 扩张状态（总扰动估计）
    double beta1;  // 观测器增益1
    double beta2;  // 观测器增益2
    double beta3;  // 观测器增益3
    double b0;     // 控制增益估计值
    double dt;     // 采样时间
} LESO;

// 初始化LESO
void initLESO(LESO *eso, double beta1, double beta2, double beta3, double b0, double dt) {
    eso->z1 = 0.0;
    eso->z2 = 0.0;
    eso->z3 = 0.0;
    eso->beta1 = beta1;
    eso->beta2 = beta2;
    eso->beta3 = beta3;
    eso->b0 = b0;
    eso->dt = dt;
}

// 更新LESO状态（每次采样调用）
void updateLESO(LESO *eso, double y, double u) {
    // 1. 计算输出估计误差
    double e = eso->z1 - y;
    
    // 2. 状态更新（欧拉离散化）
    double dz1 = eso->z2 - eso->beta1 * e;
    double dz2 = eso->z3 - eso->beta2 * e + eso->b0 * u;
    double dz3 = -eso->beta3 * e;
    
    // 3. 积分更新状态
    eso->z1 += dz1 * eso->dt;
    eso->z2 += dz2 * eso->dt;
    eso->z3 += dz3 * eso->dt;
}

int main() {
    // 示例参数设置
    double omega_o = 50;      // 观测器带宽（rad/s）
    double b0_est = 1.25;     // 控制增益估计值
    double dt = 0.001;        // 采样时间（1ms）
    
    // 计算观测器增益（带宽参数化）
    double beta1 = 3.0 * omega_o;
    double beta2 = 3.0 * omega_o * omega_o;
    double beta3 = omega_o * omega_o * omega_o;
    
    // 初始化LESO
    LESO eso;
    initLESO(&eso, beta1, beta2, beta3, b0_est, dt);
    
    // 模拟运行（示例）
    double y_measure = 0.0;   // 实际系统输出（应由传感器获取）
    double u_input = 5.0;     // 控制输入
    
    for(int i = 0; i < 1000; i++) {
        updateLESO(&eso, y_measure, u_input);
        
        // 打印状态（实际应用中应移除）
        if(i % 100 == 0) {
            printf("t=%.3fs: z1=%.4f, z2=%.4f, z3=%.4f\n", 
                   i*dt, eso.z1, eso.z2, eso.z3);
        }
        
        // 这里应添加实际系统仿真代码
        // y_measure = ... （从实际系统获取）
    }
    
    return 0;
}