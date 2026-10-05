#include "scara_kinematics.h"

// matlab拟合scara作用域边缘曲线 4组多项式系数
//右手系
// static double p1[11]    = {-1.10065178273691e-22, 1.06645332267733e-19, -1.98070247542431e-17, -9.01209362520754e-15, 2.96610079693952e-12, 1.75520102235951e-10, -1.21357691731590e-07, 3.85247448810884e-06, 9.82819302033759e-05, -0.0491176786968019, 437.137342610030};
// static double p2[8]     = {3.84663645304406e-10, 5.70553992838450e-07, 0.000361679303699002, 0.127026860974384, 26.6969694166810, 3357.73670397313, 234016.557026143, 6972319.01824987};
// static double p3[8]     = {4.23575783078938e-14, 1.11891283783721e-11, 5.48266493973248e-10, -1.07478937940491e-07, -7.78557552183604e-06, -0.00266221190038819, 0.0115628796358165, 185.765483072071};
// static double p4[11]    = {-4.96722851403753e-21, 1.14090410334196e-17, -1.13935681434100e-14, 6.48680841833184e-12, -2.31926447897326e-09, 5.40178413458652e-07, -8.21146036034213e-05, 0.00790450262928784, -0.448628796177971, 13.4112239873496, 0};
// static float x_limit[2] = {-250.0f, 440.0f};
/*                 / \ x_axis
                    |
                    |
                    |
                    |
                    |
<--------------------
y_axis
*/

// matlab拟合scara作用域边缘曲线 4组多项式系数
//左手系 边界曲线是x=f(y)

static double p1_left[12]    = {-9.19783820154168e-26, -8.43412855479107e-23, 1.68308078383591e-19, -5.53784999153763e-17, -7.32176988987681e-15, 5.22038285883451e-12, -1.33739379602848e-10, -1.58438016933300e-07, 1.10139207872326e-05, -0.000134235298985242, -0.0447874371527445, 437.074377011056};
static double p2_left[8]     = {-5.83471804600852e-14, 1.04234403581324e-10, -7.72639220455988e-08, 3.06768322353393e-05, -0.00701654097258198, 0.918449748231459, -62.9447674491854, 1889.65231899421};
static double p3_left[8]     = {2.63110665475006e-14, 2.90568291485181e-12, -5.33933793010990e-10, -7.87966873042083e-08, 3.43665441748701e-06, -0.00241616306396199, -0.00446805569412984, 184.852565240003};
static double p4_left[6]     = {-2.42025565077405e-06, -0.00235099169942876, -0.911645157853126, -176.380787311356, -17026.5122739973, -656014.080150478};
static float y_limit_left[2] = {-228.0f, 400.0f};

static initial_state init_state;
static initial_state current_state; //在push函数中使用

/**
 * @brief two dimensional scara
 * @param handcoor 1:right hand 2:left hand
 * @param angles rad -pi~pi
 */
void scara_inverse_kinematics(float x, float y, float L1, float L2, uint8_t handcoor, float angles[2])
{
    if (pow(x, 2) + pow(y, 2) > pow(L1 + L2, 2)) {
        x = (L1 + L2) * cos(atan2(y, x));
        y = (L1 + L2) * sin(atan2(y, x));
    }
    float cos_beta   = (pow(x, 2) + pow(y, 2) - pow(L1, 2) - pow(L2, 2)) / (2 * L1 * L2);
    float sin_beta   = 0.0f;
    float temp       = 1 - pow(cos_beta, 2);
    float calc_error = 0.1f;
    if (temp < 0) {
        if (temp > -calc_error) {
            temp = 0;
        } else {
            return;
        }
    }
    //避免解出奇怪值
    if (cos_beta > 1 || cos_beta < -1) { return; }
    if (handcoor == 1) {
        sin_beta = sqrt(temp);
    } else if (handcoor == 2) {
        sin_beta = -sqrt(temp);
    } else {
    }
    angles[1] = atan2(sin_beta, cos_beta);
    angles[0] = atan2(y, x) - atan2(L2 * sin_beta, L1 + L2 * cos_beta);
}

/**
 * @brief 算多项式结果
 * @param p 多项式系数
 * @param  terms 多项式项数
 */
double polyval_calc(double p[], double x, uint8_t terms)
{
    double y = 0;
    for (int i = 0; i < terms; i++) {
        y += pow(x, terms - i - 1) * p[i];
    }
    return y;
}

float debug[100];
/**
 * @brief 二分法将点投到边界
 */
static void ProjectOnCurve(float x, float y, float res_xy[2], double polyval)
{
    if (y / x > 1000 || x / y > 1000) {
        res_xy[0] = polyval;
        res_xy[1] = y;
        debug[1]  = 666;
        return;
    }
    double temp_coordinate_max[2];
    double temp_coordinate_min[2];
    double temp_coordinate_middle[2];
    double ErrorOfPro;

    temp_coordinate_max[0]    = y;
    temp_coordinate_max[1]    = x;
    temp_coordinate_min[0]    = polyval * y / x;
    temp_coordinate_min[1]    = polyval;
    temp_coordinate_middle[0] = 0.5 * (temp_coordinate_max[0] + temp_coordinate_min[0]);
    ErrorOfPro                = polyval_calc(p1_left, temp_coordinate_middle[0], 12) - temp_coordinate_middle[0] * x / y;
    // while (fabs(ErrorOfPro)>10)//迭代次数过多导致程序有问题
    for (int i = 0; i < 3; i++) {
        if (ErrorOfPro > 0) {
            temp_coordinate_min[0] = temp_coordinate_middle[0];
        }
        if (ErrorOfPro < 0) {
            temp_coordinate_max[0] = temp_coordinate_middle[0];
        }
        temp_coordinate_middle[0] = 0.5 * (temp_coordinate_max[0] + temp_coordinate_min[0]);
        ErrorOfPro                = polyval_calc(p1_left, temp_coordinate_middle[0], 12) - temp_coordinate_middle[0] * x / y;
    }
    res_xy[0] = temp_coordinate_middle[1];
    res_xy[1] = temp_coordinate_middle[0];
}

/**
 * @brief 越界检查并转化到最短边界点
 * @param res_xy 输出的结果xy坐标，第一个为x，第二个为y
 */
void check_boundary_scara_lefthand(float x, float y, float res_xy[2])
{
    //限位
    if (y < y_limit_left[0]) {
        y = y_limit_left[0];
    }
    if (y > y_limit_left[1]) {
        y = y_limit_left[1];
    }
    //算基本值

    double polyval_p1 = polyval_calc(p1_left, y, 12) - 20;
    double polyval_p2 = polyval_calc(p2_left, y, 8);
    double polyval_p3 = polyval_calc(p3_left, y, 8);
    double polyval_p4 = polyval_calc(p4_left, y, 6);

    if (x > polyval_p1) {
        // ProjectOnCurve(x,y,res_xy,polyval_p1);
        res_xy[1] = y;
        res_xy[0] = polyval_p1;
    } else {
        if (y >= -y_limit_left[0] && y < -170 && x < polyval_p4) { x = polyval_p4; }
        if (y >= -170 && y < 100 && x < polyval_p3) { x = polyval_p3; }
        if (y >= 100 && y <= y_limit_left[1] && x < polyval_p2) { x = polyval_p2; }
        res_xy[0] = x;
        res_xy[1] = y;
    }
}

/**
 * @brief 越界检查并转化到最短边界点
 * @param res_xy 输出的结果xy坐标，第一个为x，第二个为y
 */
void check_boundary_scara(float x, float y, float res_xy[2])
{
    // //限位
    // if (x < x_limit[0]) {
    //     x = x_limit[0];
    // }
    // if (x > x_limit[1]) {
    //     x = x_limit[1];
    // }
    // //算基本值

    // double polyval_p1 = polyval_calc(p1, x, 11);
    // double polyval_p2 = polyval_calc(p2, x, 8);
    // double polyval_p3 = polyval_calc(p3, x, 8);
    // double polyval_p4 = polyval_calc(p4, x, 11);

    // if (y > polyval_p1) {
    //     y = polyval_p1;
    // } else {
    //     if (x > -250 && x < -177 && y < polyval_p2) { y = polyval_p2; }
    //     if (x >= -177 && x < 85 && y < polyval_p3) { y = polyval_p3; }
    //     if (x >= 85 && x <= 440 && y < polyval_p4) { y = polyval_p4; }
    // }
    // res_xy[0] = x;
    // res_xy[1] = y;
}

/**
 * @brief scara运动学正解算
 * @param angle1大臂角度
 * @param angle2小臂角度
 * @param res_xy 第一个为x坐标，第二个为y坐标
 */
void scara_forward_kinematics(float angle1, float angle2, float L1, float L2, float res_xy[2])
{
    float x, y;
    float cos_angle1  = cos(angle1);
    float cos_angle12 = cos(angle1 + angle2);
    float sin_angle1  = sin(angle1);
    float sin_angle12 = sin(angle1 + angle2);
    if (cos_angle12 < -1 || cos_angle12 > 1 || cos_angle1 > 1 || cos_angle1 < -1 || sin_angle12 < -1 || sin_angle12 > 1 || sin_angle1 > 1 || sin_angle1 < -1) { return; }
    x         = L1 * cos_angle1 + L2 * cos_angle12;
    y         = L1 * sin_angle1 + L2 * sin_angle12;
    res_xy[0] = x;
    res_xy[1] = y;
}

/**
 * @brief 进行推矿操作的时候初始化一次，记录机械臂主要是末端执行器位姿信息
 */
void GetCurrentState(float angle1, float angle2, float angle3, float angle4, float z, float roll_angle)
{
    current_state.init_angle1 = angle1 + 0.f;
    current_state.init_angle2 = angle2 - 0.f;
    current_state.init_yaw    = angle3 - 0.;
    current_state.init_pitch  = angle4 - 0.;
    current_state.init_Z      = z;
    current_state.init_roll   = roll_angle;
}
/**
 * @brief 推矿操作，按下一个键进入
 * @param length 推出的长度
 */
void PushToCube(float result[6], float length)
{
    float xy[2];
    uint8_t handcoor;
    float res_angle[2];
    float OperatorVector[3]; // x,y,z
    float Euler_Yaw   = current_state.init_angle1 + current_state.init_angle2 + current_state.init_yaw;
    float Euler_Pitch = current_state.init_pitch;
    OperatorVector[0] = cos(Euler_Yaw) * cos(Euler_Pitch) * length;
    OperatorVector[1] = sin(Euler_Yaw) * cos(Euler_Pitch) * length;
    OperatorVector[2] = sin(Euler_Pitch) * length;
    scara_forward_kinematics(current_state.init_angle1, current_state.init_angle2, ARMLENGHT1, ARMLENGHT2, xy);
    xy[0] += OperatorVector[0];
    xy[1] += OperatorVector[1];
    if (current_state.init_angle2 < 0) { handcoor = 2; } //机械臂呈左手形状
    else {
        handcoor = 1;
    } //机械臂呈右手形状
    scara_inverse_kinematics(xy[0], xy[1], ARMLENGHT1, ARMLENGHT2, handcoor, res_angle);
    result[0] = res_angle[0] - 0.f;
    result[1] = res_angle[1] + 0.f;
    result[2] = current_state.init_yaw - (res_angle[0] - current_state.init_angle1 + res_angle[1] - current_state.init_angle2);
    result[3] = current_state.init_pitch;
    result[4] = current_state.init_roll;
    result[6] = current_state.init_Z + OperatorVector[2];
}

/**
 * @brief 微调模式下得到角度
 * @param angle4 目前机械臂的编码值
 * @param z 目前机械臂高度
 * @param roll_angle 目前机械臂角度
 * @param delta_... 自定义发过来的微调数据
 * @param result 6个数，0大臂编码值，1小臂编码值，2yaw编码值，3pitch编码值，4roll角度，5z高度
 */
void GC_get_target_angles_slightly(slightly_controll_data data_pack, float result[6])
{
    float xy[2];
    uint8_t handcoor;
    float res_angle[2];
    scara_forward_kinematics(init_state.init_angle1, init_state.init_angle2, ARMLENGHT1, ARMLENGHT2, xy);
    xy[0] += data_pack.delta_x;
    xy[1] += data_pack.delta_y;
    if (init_state.init_angle2 < 0) { handcoor = 2; } //机械臂呈左手形状
    else {
        handcoor = 1;
    } //机械臂呈右手形状
    scara_inverse_kinematics(xy[0], xy[1], ARMLENGHT1, ARMLENGHT2, handcoor, res_angle);
    result[0] = res_angle[0] - 0.f;
    result[1] = res_angle[1] + 0.f;
    result[2] = init_state.init_yaw + data_pack.delta_yaw;
    result[3] = init_state.init_pitch + data_pack.delta_pitch;
    result[4] = init_state.init_roll + data_pack.delta_roll;
    result[5] = init_state.init_Z + data_pack.delta_z;
}

void StateInit(float angle1, float angle2, float angle3, float angle4, float z, float roll_angle)
{
    init_state.init_angle1 = angle1 + 0.f;
    init_state.init_angle2 = angle2 - 0.f;
    init_state.init_yaw    = angle3;
    init_state.init_pitch  = angle4;
    init_state.init_Z      = z;
    init_state.init_roll   = roll_angle;
}

void RecordMode(uint8_t HeadByte)
{
}


#define LimitMax(input, max)   \
    {                          \
        if (input > max)       \
        {                      \
            input = max;       \
        }                      \
        else if (input < -max) \
        {                      \
            input = -max;      \
        }                      \
    }



