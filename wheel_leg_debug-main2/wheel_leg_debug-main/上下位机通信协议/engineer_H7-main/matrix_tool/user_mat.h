//个人比较喜欢的矩阵运算工具
// double数组前两个数据分别存储矩阵的row和column,一维数组，按行排列
//若为0维矩阵或向量，传入参数NULL_MAT,row或column为0,视作NULL_MAT
//支持数组（矩阵）空间连续操作,如user_mat_multiply(debug_M3, debug_M3, debug_M3);<=>debug_M3^2
//代价就是运行时对栈空间要求上升
//写代码的习惯：创建数组一定要设置为0或者赋值
//运算过程中占用栈空间较多(double数组比较大)建议多分配点栈空间
#ifndef USER_MAT_H
#define USER_MAT_H

typedef enum
{
    MAT_OK = 1,
    MAT_ERROR = 0
} MAT_STATES;

MAT_STATES user_mat_add(double M1[], double M2[], double resM[]);
MAT_STATES user_mat_subtract(double M1[], double M2[], double resM[]);
MAT_STATES user_mat_transpose(double M1[], double resM[]);
MAT_STATES user_mat_multiply(double M1[], double M2[], double resM[]);
MAT_STATES user_mat_inverse(double M1[], double resM[]);
MAT_STATES user_mat_num_multiply(double M1[], double num, double resM[]);

#endif // USER_MAT_H
