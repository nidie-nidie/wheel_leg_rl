#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

#include "user_mat.h" // double数组表示矩阵，前两个数据分别存储矩阵的row和column,一维数组，按行排列
//矩阵运算最高维度为12
void kf_get_x_minus(double A[], double B[], double last_x_hat[], double last_u[], double x_minus[]);
void kf_get_P_minus(double A[], double last_P[], double Q[], double P_minus[]);
void kf_get_K(double P_minus[], double H[], double R[], double K[]);
void kf_get_x_hat(double x_minus[], double K[], double z[], double H[], double x_hat[]);
void kf_get_P(double K[], double H[], double P_minus[], double P[]);
void kf_copy(double last_x_hat[], double last_P[], double x_hat[], double P[]);
void ekf_get_x_hat(double x_minus[], double K[], double z[], void (*h)(double x[], double res[]), double x_hat[]);
#endif // KALMAN_FILTER_H_
