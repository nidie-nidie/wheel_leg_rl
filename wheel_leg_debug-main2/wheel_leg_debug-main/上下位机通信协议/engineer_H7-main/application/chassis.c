#include "chassis.h"

//四个电机实例
M3508motor_t motor_lf,motor_lb,motor_rf,motor_rb;

//每个电机的速度参考值
double vt_lf,vt_lb,vt_rf,vt_rb;

//轮子速度PID结构体
pid_type_def wheel_speed_pid_lf,wheel_speed_pid_lb,wheel_speed_pid_rf,wheel_speed_pid_rb;

double lf_pid_output,lb_pid_output,rf_pid_output,rb_pid_output;
//走直线PID结构体
pid_type_def straight_line_pid;

//hfdcan1的接收数组
extern uint8_t rx_data1[8];

double vx,vy,vw;//解算速度
float target_vx,target_vy;//目标速度等
//麦轮解算
static void Magnum_calculate(double chassis_vx,double chassis_vy, double chassis_wz,int reverse_flag);
static void splitAndStoreSignals(int16_t signal1, int16_t signal2, int16_t signal3, int16_t signal4, uint8_t *outputArray) ;

extern RC_ctrl_t rc_ctrl;//遥控器数据

extern Chassis_CMD_data_t Chassis_CMD_data;

extern IMU_DATA imu_data;

extern uint16_t rec_id1;

uint8_t chassis_can_send[8]={0};//

double initial_yaw = 0; // 记录进入模式时的yaw角度
static int first_enter_X = 1; // 用于标记是否是第一次进入x轴走直线
static int first_enter_Y = 1; // 用于标记是否是第一次进入y轴走直线

double yaw_error;//每次相对目标方向的偏差

static void send_wheel_can_data(void);
static void slow_start(void);

void CHASSIS_INIT(void)
{
    M3508_init(&motor_lf, 1, 3591/187);
    M3508_init(&motor_lb, 4, 3591/187);
    M3508_init(&motor_rf, 3, 3591/187);
    M3508_init(&motor_rb, 2, 3591/187);

    PID_init(&wheel_speed_pid_lf, PID_POSITION,  39, 0.01, 0.01, 10000, 15000);//轮子自身速度的PID

    PID_init(&wheel_speed_pid_lb, PID_POSITION,  39, 0.01, 0.01, 10000, 15000);//轮子自身速度的PID

    PID_init(&wheel_speed_pid_rf, PID_POSITION,  39, 0.01, 0.01, 10000, 15000);//轮子自身速度的PID

    PID_init(&wheel_speed_pid_rb, PID_POSITION,  39, 0.01, 0.01, 10000, 15000);//轮子自身速度的PID

    PID_init(&straight_line_pid, PID_POSITION,  120, 0.001, 0.25, 10000, 10);//轮子走直线PID 95 0.2 100


}

void CHASSIS_TASK(void)
{
    if(rec_id1==motor_lf.id)
    M3508_fbkdata(&motor_lf,rx_data1);
    if(rec_id1==motor_lb.id)
    M3508_fbkdata(&motor_lb,rx_data1);
    if(rec_id1==motor_rf.id)
    M3508_fbkdata(&motor_rf,rx_data1);
    if(rec_id1==motor_rb.id)
    M3508_fbkdata(&motor_rb,rx_data1);
		
		
		

    vx=Chassis_CMD_data.vx;
    vy=Chassis_CMD_data.vy;
    vw=Chassis_CMD_data.vw;
		
		
		if((fabs(vw)<5)&&(vx==0||vy==0))//无自旋且是单向平移（有一个方向的是0）
		{
			if(vy==0&&vx!=0)Chassis_CMD_data.Chassis_straight_mode =  STRAIGHT_X_ON;
			if(vx==0&&vy!=0)Chassis_CMD_data.Chassis_straight_mode =  STRAIGHT_Y_ON;
			if(vy==0&&vx==0)Chassis_CMD_data.Chassis_straight_mode =  STRAIGHT_OFF;
			
		}
		else//有自旋或多向平移
		{
			Chassis_CMD_data.Chassis_straight_mode =  STRAIGHT_OFF;
		}
			
		

    if(Chassis_CMD_data.Chassis_straight_mode == STRAIGHT_X_ON)
    {
        // 每次进入模式时记录初始yaw角度
        if (first_enter_X)
        {
            initial_yaw = imu_data.yawTotal; // 记录初始偏航角
            first_enter_X = 0;
			      PID_clear(&straight_line_pid);
        }
    
        // 计算当前偏航角相对初始偏航角的误差
        yaw_error = imu_data.yawTotal - initial_yaw;
    
        vw = PID_calc(&straight_line_pid, yaw_error, 0); // 通过偏航角修正
        vy = 0;
    }
    else
    {
        first_enter_X = 1; // 退出模式时重置标志，确保下次进入时重新记录yaw角度
    }

    if(Chassis_CMD_data.Chassis_straight_mode ==  STRAIGHT_Y_ON)
    {
        // 每次进入模式时记录初始yaw角度
        if (first_enter_Y)
        {
            initial_yaw = imu_data.yawTotal; // 记录初始偏航角
            first_enter_Y = 0;
		      	PID_clear(&straight_line_pid);

        }
    
        // 计算当前偏航角相对初始偏航角的误差
        yaw_error = imu_data.yawTotal - initial_yaw;
    
        vw = PID_calc(&straight_line_pid, yaw_error, 0); // 通过偏航角修正
        vx = 0;
    }
    else
    {
        first_enter_Y = 1; // 退出模式时重置标志，确保下次进入时重新记录yaw角度
    }
		
		if(Chassis_CMD_data.Chassis_straight_mode ==  STRAIGHT_Y_ON)
		{}//不进行操作
		

//    slow_start();

//    Magnum_calculate(target_vx,target_vy,vw,1);//最终结果的解算
			Magnum_calculate(vx,vy,vw,1);//最终结果的解算
		
		
    send_wheel_can_data();

}

//麦克纳姆轮解算
//向前为vy 向右为vx

static void Magnum_calculate(double chassis_vx,double chassis_vy, double chassis_wz,int reverse_flag)
{
    vt_lf = reverse_flag*(-chassis_vx - chassis_vy - chassis_wz  );
    vt_rf = reverse_flag*(-chassis_vx + chassis_vy - chassis_wz );
    vt_lb = reverse_flag*(chassis_vx - chassis_vy - chassis_wz  );
    vt_rb = reverse_flag*(chassis_vx + chassis_vy - chassis_wz );
}

//拆分信号
static void splitAndStoreSignals(int16_t signal1, int16_t signal2, int16_t signal3, int16_t signal4, uint8_t *outputArray) {
    // 拆分每个信号的高八位和低八位
    outputArray[0] = (signal1 >> 8) & 0xFF;  // signal1的高八位
    outputArray[1] = signal1 & 0xFF;        // signal1的低八位
    outputArray[2] = (signal2 >> 8) & 0xFF;  // signal2的高八位
    outputArray[3] = signal2 & 0xFF;        // signal2的低八位
    outputArray[4] = (signal3 >> 8) & 0xFF;  // signal3的高八位
    outputArray[5] = signal3 & 0xFF;        // signal3的低八位
    outputArray[6] = (signal4 >> 8) & 0xFF;  // signal4的高八位
    outputArray[7] = signal4 & 0xFF;        // signal4的低八位
}


static void send_wheel_can_data(void)
{


    lf_pid_output = PID_calc(&wheel_speed_pid_lf, motor_lf.para.vel_fbk, vt_lf);
    lb_pid_output = PID_calc(&wheel_speed_pid_lb, motor_lb.para.vel_fbk, vt_lb);
    rf_pid_output = PID_calc(&wheel_speed_pid_rf, motor_rf.para.vel_fbk, vt_rf);
    rb_pid_output = PID_calc(&wheel_speed_pid_rb, motor_rb.para.vel_fbk, vt_rb);

    splitAndStoreSignals(lf_pid_output,rb_pid_output, rf_pid_output,lb_pid_output,chassis_can_send);

    //发送信息
    fdcanx_send_data(&hfdcan1,0x200,chassis_can_send, 8);
}


static void slow_start(void)
{
    
	if(vx<=150&&vx>5)
	{target_vx=vx;}
	else if(vx>150)
	{
		target_vx=target_vx+0.1f;
	}
	else if
	(vx>=-150&&vx<-5)
	{
		target_vx=vx;
	}
	else if(vx<-150)
	{
		target_vx=target_vx-0.1f;
	}
	else
	{
		target_vx=0;
	}
			
	if(target_vx>=380)
	{target_vx=380;}
	
	if(target_vx<=-380)
	{
		target_vx=-380;
	}
	
	
	if(vy<=150&&vy>5)
	{target_vy=vy;}
	else if(vy>150)
	{
		target_vy=target_vy+0.1f;
	}
	else if
	(vy>=-150&&vy<-5)
	{
		target_vx=vy;
	}
	else if(vy<-150)
	{
		target_vy=target_vy-0.1f;
	}
	else
	{
		target_vy=0;
	}
			
	if(target_vy>=380)
	{target_vy=380;}
	
	if(target_vy<=-380)
	{
		target_vy=-380;
	}
		

}

