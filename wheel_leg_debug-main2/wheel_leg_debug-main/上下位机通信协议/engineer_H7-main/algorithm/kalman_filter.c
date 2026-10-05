#include "kalman_filter.h"
#include "string.h"

// x_minus=A*last_x_hat+B*last_u kalman filter计算先验值 A为状态转移矩阵
void kf_get_x_minus(double A[], double B[], double last_x_hat[], double last_u[], double x_minus[])
{
    //不检查矩阵和向量维度,user_mat运算库中维度错误直接返回
    double tempV1[2 + 12] = {0};
    double tempV2[2 + 12] = {0};
    // tempV1=A*last_x_hat
    user_mat_multiply(A, last_x_hat, tempV1);
    // tempV2=B*last_u
    user_mat_multiply(B, last_u, tempV2);
    // x_minus=tempV1+tempV2
    user_mat_add(tempV1, tempV2, x_minus);
}
// P_minus=A*last_P*A'+Q kalman filter计算先验值的协方差矩阵
void kf_get_P_minus(double A[], double last_P[], double Q[], double P_minus[])
{
    //不检查矩阵和向量维度,user_mat运算库中维度错误直接返回
    double tempM1[2 + 144] = {0};
    //此时A中是A'
    user_mat_transpose(A, A);
    // tempM1=last_P*A'
    user_mat_multiply(last_P, A, tempM1);
    //此时A中是A
    user_mat_transpose(A, A);
    // tempM1=A*last_P*A'
    user_mat_multiply(A, tempM1, tempM1);
    user_mat_add(tempM1, Q, P_minus);
}
// K=(P_minus*H')*inv(H*P_minus*H'+R) kalman filter计算kalman gain
void kf_get_K(double P_minus[], double H[], double R[], double K[])
{
    //不检查矩阵和向量维度,user_mat运算库中维度错误直接返回
    double tempM1[2 + 144] = {0};
    //此时H中是H'
    user_mat_transpose(H, H);
    // tempM1=P_minus*H'
    user_mat_multiply(P_minus, H, tempM1);
    //此时H中是H
    user_mat_transpose(H, H);
    // tempM1=H*P_minus*H'
    user_mat_multiply(H, tempM1, tempM1);
    // tempM1=H*P_minus*H'+R
    user_mat_add(tempM1, R, tempM1);
    // tempM1=inv(H*P_minus*H'+R)
    user_mat_inverse(tempM1, tempM1);
    //此时H中是H'
    user_mat_transpose(H, H);
    // temp1=H'*inv(H*P_minus*H'+R)
    user_mat_multiply(H, tempM1, tempM1);
    //此时H中是H
    user_mat_transpose(H, H);
    // K=P_minus*temp1
    user_mat_multiply(P_minus, tempM1, K);
}
// x_hat=x_minus+K*(z-H*x_minus) kalman filter计算后验值（融合值）得到最终修正结果x_hat
void kf_get_x_hat(double x_minus[], double K[], double z[], double H[], double x_hat[])
{
    //不检查矩阵和向量维度,user_mat运算库中维度错误直接返回
    double tempV1[2 + 12] = {0};
    // tempV1=H*x_minus
    user_mat_multiply(H, x_minus, tempV1);
    // tempV1=z-H*x_minus
    user_mat_subtract(z, tempV1, tempV1);
    // tempV1=K*(z-H*x_minus)
    user_mat_multiply(K, tempV1, tempV1);
    // x_hat=x_minus+tempV1
    user_mat_add(x_minus, tempV1, x_hat);
}
// P=P_minus-K*H*P_minus kalman filter计算后验值的协方差矩阵
void kf_get_P(double K[], double H[], double P_minus[], double P[])
{
    //不检查矩阵和向量维度,user_mat运算库中维度错误直接返回
    double tempM1[2 + 144] = {0};
    // tempM1=H*P_minus
    user_mat_multiply(H, P_minus, tempM1);
    // tempM1=K*H*P_minus
    user_mat_multiply(K, tempM1, tempM1);
    // P=P_minus-tempM1
    user_mat_subtract(P_minus, tempM1, P);
}
//一次kalman upgrade结束，当前x_hat和P复制到last_x_hat和last_P
void kf_copy(double last_x_hat[], double last_P[], double x_hat[], double P[])
{
    memcpy((void *)last_x_hat, (const void *)x_hat, sizeof(double) * (2 + (int)(x_hat[0]) * (int)(x_hat[1])));
    memcpy((void *)last_P, (const void *)P, sizeof(double) * (2 + (int)(P[0]) * (int)(P[1])));
}
// x_hat=x_minus+K*(z-h(x_minus)) extended kalman filter计算后验值（融合值）得到最终修正结果x_hat
void ekf_get_x_hat(double x_minus[], double K[], double z[], void (*h)(double x[], double res[]), double x_hat[])
{
    //不检查矩阵和向量维度,user_mat运算库中维度错误直接返回
    double tempV1[2 + 12] = {0};
    // tempV1=h(x_minus)
    h(x_minus, tempV1);
    // tempV1=z-h(x_minus)
    user_mat_subtract(z, tempV1, tempV1);
    // tempV1=K*(z-h(x_minus))
    user_mat_multiply(K, tempV1, tempV1);
    // x_hat=x_minus+tempV1
    user_mat_add(x_minus, tempV1, x_hat);
}
