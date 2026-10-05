#include "user_mat.h"
#include "math.h"
#include "string.h"

#define offsetINDEX 2 //数组第2个数对应矩阵的第一个值
#define NULL_MAT 0    // 0矩阵

// M1+M2=resM
MAT_STATES user_mat_add(double M1[], double M2[], double resM[])
{
    //根据维度判断是否为0矩阵
    if ((int)M1[0] == 0 || (int)M1[1] == 0)
    {
        M1 = NULL_MAT;
    }
    //根据维度判断是否为0矩阵
    if ((int)M2[0] == 0 || (int)M2[1] == 0)
    {
        M2 = NULL_MAT;
    }
    if (M1 == NULL_MAT && M2 == NULL_MAT)
    {
        resM[0] = 0;
        resM[1] = 0; //设置该矩阵为NULL_MAT
        return MAT_OK;
    }
    if (M1 == NULL_MAT && M2 != NULL_MAT)
    {
        memcpy((void *)(resM), (const void *)M2, sizeof(double) * (M2[0] * M2[1] + offsetINDEX));
        return MAT_OK;
    }
    if (M1 != NULL_MAT && M2 == NULL_MAT)
    {
        memcpy((void *)(resM), (const void *)M1, sizeof(double) * (M1[0] * M1[1] + offsetINDEX));
        return MAT_OK;
    }
    //都为非0矩阵的情况下,将M1和M2数据搬运到局部空间
    double local_M1[2 + 144] = {0};
    double local_M2[2 + 144] = {0};
    memcpy((void *)(local_M1), (const void *)M1, sizeof(double) * ((int)M1[0] * (int)M1[1] + 2));
    memcpy((void *)(local_M2), (const void *)M2, sizeof(double) * ((int)M2[0] * (int)M2[1] + 2));

    int M1_row = M1[0];
    int M1_col = M1[1];
    int M2_row = M2[0];
    int M2_col = M2[1];
    if (M1_row != M2_row || M1_col != M2_col)
    {
        return MAT_ERROR;
    }
    for (int i = 0; i < M1_row; i++)
    {
        for (int j = 0; j < M1_col; j++)
        {
            resM[offsetINDEX + M1_col * i + j] = local_M1[offsetINDEX + M1_col * i + j] + local_M2[offsetINDEX + M1_col * i + j];
        }
    }
    resM[0] = M1_row;
    resM[1] = M1_col;
    return MAT_OK;
}
// M1-M2=resM
MAT_STATES user_mat_subtract(double M1[], double M2[], double resM[])
{
    //根据维度判断是否为0矩阵
    if ((int)M1[0] == 0 || (int)M1[1] == 0)
    {
        M1 = NULL_MAT;
    }
    //根据维度判断是否为0矩阵
    if ((int)M2[0] == 0 || (int)M2[1] == 0)
    {
        M2 = NULL_MAT;
    }
    if (M1 == NULL_MAT && M2 == NULL_MAT)
    {
        resM[0] = 0;
        resM[1] = 0; //设置该矩阵为NULL_MAT
        return MAT_OK;
    }
    if (M1 == NULL_MAT && M2 != NULL_MAT)
    {
        memcpy((void *)(resM), (const void *)M2, sizeof(double) * (M2[0] * M2[1] + offsetINDEX));
        //矩阵元素取反
        for (int i = 0; i < (int)M2[0]; i++)
        {
            for (int j = 0; j < (int)M2[1]; j++)
            {
                resM[offsetINDEX + i * (int)M2[1] + j] = -resM[offsetINDEX + i * (int)M2[1] + j];
            }
        }
        return MAT_OK;
    }
    if (M1 != NULL_MAT && M2 == NULL_MAT)
    {
        memcpy((void *)(resM), (const void *)M1, sizeof(double) * (M1[0] * M1[1] + offsetINDEX));
        return MAT_OK;
    }
    //都为非0矩阵的情况下,将M1和M2数据搬运到局部空间
    double local_M1[2 + 144] = {0};
    double local_M2[2 + 144] = {0};
    memcpy((void *)(local_M1), (const void *)M1, sizeof(double) * ((int)M1[0] * (int)M1[1] + 2));
    memcpy((void *)(local_M2), (const void *)M2, sizeof(double) * ((int)M2[0] * (int)M2[1] + 2));

    int M1_row = M1[0];
    int M1_col = M1[1];
    int M2_row = M2[0];
    int M2_col = M2[1];
    if (M1_row != M2_row || M1_col != M2_col)
    {
        return MAT_ERROR;
    }
    for (int i = 0; i < M1_row; i++)
    {
        for (int j = 0; j < M1_col; j++)
        {
            resM[offsetINDEX + M1_col * i + j] = local_M1[offsetINDEX + M1_col * i + j] - local_M2[offsetINDEX + M1_col * i + j];
        }
    }
    resM[0] = M1_row;
    resM[1] = M1_col;
    return MAT_OK;
}
// resM=M1'
MAT_STATES user_mat_transpose(double M1[], double resM[])
{
    //根据维度判断是否为0矩阵
    if ((int)M1[0] == 0 || (int)M1[1] == 0)
    {
        M1 = NULL_MAT;
    }
    if (M1 == NULL_MAT)
    {
        resM[0] = 0;
        resM[1] = 0; //设置该矩阵为NULL_MAT
        return MAT_OK;
    }
    //都为非0矩阵的情况下,将M1和M2数据搬运到局部空间
    double local_M1[2 + 144] = {0};
    memcpy((void *)(local_M1), (const void *)M1, sizeof(double) * ((int)M1[0] * (int)M1[1] + 2));

    int M1_row = M1[0];
    int M1_col = M1[1];
    for (int i = 0; i < M1_row; i++)
    {
        for (int j = 0; j < M1_col; j++)
        {
            resM[offsetINDEX + i + j * M1_row] = local_M1[M1_col * i + j + offsetINDEX];
        }
    }
    resM[0] = M1_col;
    resM[1] = M1_row;
    return MAT_OK;
}
// resM=M1*M2
MAT_STATES user_mat_multiply(double M1[], double M2[], double resM[])
{
    //根据维度判断是否为0矩阵
    if ((int)M1[0] == 0 || (int)M1[1] == 0)
    {
        M1 = NULL_MAT;
    }
    //根据维度判断是否为0矩阵
    if ((int)M2[0] == 0 || (int)M2[1] == 0)
    {
        M2 = NULL_MAT;
    }
    if (M1 == NULL_MAT || M2 == NULL_MAT)
    {
        resM[0] = 0;
        resM[1] = 0; //设置该矩阵为NULL_MAT
        return MAT_OK;
    }
    //都为非0矩阵的情况下,将M1和M2数据搬运到局部空间
    double local_M1[2 + 144] = {0};
    double local_M2[2 + 144] = {0};
    memcpy((void *)(local_M1), (const void *)M1, sizeof(double) * ((int)M1[0] * (int)M1[1] + 2));
    memcpy((void *)(local_M2), (const void *)M2, sizeof(double) * ((int)M2[0] * (int)M2[1] + 2));

    int M1_row = M1[0];
    int M1_col = M1[1];
    int M2_row = M2[0];
    int M2_col = M2[1];
    if (M1_col != M2_row)
    {
        return MAT_ERROR;
    }
    resM[0] = M1_row;
    resM[1] = M2_col;
    for (int i = 0; i < M1_row; i++)
    {
        for (int j = 0; j < M2_col; j++)
        {
            resM[offsetINDEX + j + i * M2_col] = 0;
            for (int nb = 0; nb < M1_col; nb++)
            {
                resM[offsetINDEX + j + i * M2_col] += local_M1[offsetINDEX + i * M1_col + nb] * local_M2[offsetINDEX + j + nb * M2_col];
            }
        }
    }
    return MAT_OK;
}
// resM=inv(M1)最大支持12维方阵求逆
MAT_STATES user_mat_inverse(double M1[], double resM[])
{
    //根据维度判断是否为0矩阵
    if ((int)M1[0] == 0 || (int)M1[1] == 0)
    {
        M1 = NULL_MAT;
    }
    if (M1 == NULL_MAT)
    {
        resM[0] = 0;
        resM[1] = 0; //设置该矩阵为NULL_MAT
        return MAT_OK;
    }
    int M1_row = M1[0];
    int M1_col = M1[1];
    if (M1_row != M1_col || M1_row > 12)
    {
        return MAT_ERROR;
    }
    double M_data[144] = {0};
    double s;
    double smax;
    double inversed_M_data[144] = {0};
    double cx_data[144] = {0}; //最大可求得矩阵逆的维度为12维
    int b;
    int b_n;
    int i;
    int j;
    int jA;
    int jj;
    int jp1j;
    int k;
    int ldap1;
    int mmj_tmp;
    int nnnnn;
    int u1;
    int yk;
    int ipiv_nb_data[12] = {0};
    int np_data[12] = {0}; //最大可求得矩阵逆的维度为12维
    for (int i = 0; i < M1_row; i++)
    {
        for (int j = 0; j < M1_col; j++)
        {
            M_data[i + j * M1_row] = M1[M1_col * i + j + offsetINDEX];
        }
    }
    if ((M1_row == 0) || (M1_col == 0))
    {
        yk = M1_row * M1_col;
        for (i = 0; i < yk; i++)
        {
            inversed_M_data[i] = M_data[i];
        }
    }
    else
    {
        nnnnn = M1_row;
        yk = M1_row * M1_col;
        for (i = 0; i < yk; i++)
        {
            inversed_M_data[i] = 0.0;
        }
        yk = M1_row * M1_col;
        for (i = 0; i < yk; i++)
        {
            cx_data[i] = M_data[i];
        }
        b_n = M1_row;
        ipiv_nb_data[0] = 1;
        yk = 1;
        for (k = 2; k <= b_n; k++)
        {
            yk++;
            ipiv_nb_data[k - 1] = yk;
        }
        ldap1 = M1_row;
        yk = M1_row - 1;
        u1 = M1_row;
        if (yk <= u1)
        {
            u1 = yk;
        }
        for (j = 0; j < u1; j++)
        {
            mmj_tmp = nnnnn - j;
            b = j * (nnnnn + 1);
            jj = j * (ldap1 + 1);
            jp1j = b + 2;
            if (mmj_tmp < 1)
            {
                yk = -1;
            }
            else
            {
                yk = 0;
                if (mmj_tmp > 1)
                {
                    smax = fabs(cx_data[jj]);
                    for (k = 2; k <= mmj_tmp; k++)
                    {
                        s = fabs(cx_data[(b + k) - 1]);
                        if (s > smax)
                        {
                            yk = k - 1;
                            smax = s;
                        }
                    }
                }
            }
            if (cx_data[jj + yk] != 0.0)
            {
                if (yk != 0)
                {
                    b_n = j + yk;
                    ipiv_nb_data[j] = b_n + 1;
                    for (k = 0; k < nnnnn; k++)
                    {
                        yk = k * nnnnn;
                        jA = j + yk;
                        smax = cx_data[jA];
                        i = b_n + yk;
                        cx_data[jA] = cx_data[i];
                        cx_data[i] = smax;
                    }
                }
                i = jj + mmj_tmp;
                for (jA = jp1j; jA <= i; jA++)
                {
                    cx_data[jA - 1] /= cx_data[jj];
                }
            }
            b_n = b + nnnnn;
            jA = jj + ldap1;
            for (jp1j = 0; jp1j <= mmj_tmp - 2; jp1j++)
            {
                yk = b_n + jp1j * nnnnn;
                smax = cx_data[yk];
                if (cx_data[yk] != 0.0)
                {
                    i = jA + 2;
                    b = mmj_tmp + jA;
                    for (yk = i; yk <= b; yk++)
                    {
                        cx_data[yk - 1] += cx_data[((jj + yk) - jA) - 1] * -smax;
                    }
                }
                jA += nnnnn;
            }
        }
        b_n = M1_row;
        np_data[0] = 1;
        yk = 1;
        for (k = 2; k <= b_n; k++)
        {
            yk++;
            np_data[k - 1] = yk;
        }
        i = M1_row;
        for (k = 0; k < i; k++)
        {
            b = ipiv_nb_data[k];
            if (b > k + 1)
            {
                yk = np_data[b - 1];
                np_data[b - 1] = np_data[k];
                np_data[k] = yk;
            }
        }
        for (k = 0; k < nnnnn; k++)
        {
            i = np_data[k];
            inversed_M_data[k + (int)M1_row * (i - 1)] = 1.0;
            for (j = k + 1; j <= nnnnn; j++)
            {
                if (inversed_M_data[(j + (int)M1_row * (i - 1)) - 1] != 0.0)
                {
                    b = j + 1;
                    for (jA = b; jA <= nnnnn; jA++)
                    {
                        inversed_M_data[(jA + (int)M1_row * (i - 1)) - 1] -=
                            inversed_M_data[(j + (int)M1_row * (i - 1)) - 1] *
                            cx_data[(jA + (int)M1_row * (j - 1)) - 1];
                    }
                }
            }
        }
        for (j = 0; j < nnnnn; j++)
        {
            yk = nnnnn * j - 1;
            for (k = nnnnn; k >= 1; k--)
            {
                b_n = nnnnn * (k - 1) - 1;
                i = k + yk;
                if (inversed_M_data[i] != 0.0)
                {
                    inversed_M_data[i] /= cx_data[k + b_n];
                    for (jA = 0; jA <= k - 2; jA++)
                    {
                        b = (jA + yk) + 1;
                        inversed_M_data[b] -= inversed_M_data[i] * cx_data[(jA + b_n) + 1];
                    }
                }
            }
        }
        for (int i = 0; i < M1_row; i++)
        {
            for (int j = 0; j < M1_row; j++)
            {
                resM[M1_col * i + j + offsetINDEX] = inversed_M_data[i + j * M1_row];
            }
        }
        resM[0] = M1_row;
        resM[1] = M1_col;
    }
    return MAT_OK;
}

MAT_STATES user_mat_num_multiply(double M1[], double num, double resM[])
{
    //根据维度判断是否为0矩阵
    if ((int)M1[0] == 0 || (int)M1[1] == 0)
    {
        M1 = NULL_MAT;
    }
    //都为非0矩阵的情况下,将M1和M2数据搬运到局部空间
    double local_M1[2 + 144] = {0};
    memcpy((void *)(local_M1), (const void *)M1, sizeof(double) * ((int)M1[0] * (int)M1[1] + 2));
    int M1_row = M1[0];
    int M1_col = M1[1];
    for (int i = 0; i < M1_row; i++)
    {
        for (int j = 0; j < M1_col; j++)
        {
            resM[M1_col * i + j + offsetINDEX] = local_M1[M1_col * i + j + offsetINDEX] * num;
        }
    }
    resM[0] = M1_row;
    resM[1] = M1_col;
    return MAT_OK;
}
