#include "robot_cmd.h"


// 当前使用的指针（总指针）
KeyComboCounter_t *v_counter = &rc_v_counter;
KeyComboCounter_t *b_counter = &rc_b_counter;
KeyComboCounter_t *g_counter = &rc_g_counter;
KeyComboCounter_t *x_counter = &rc_x_counter;
KeyComboCounter_t *z_counter = &rc_z_counter;
KeyComboCounter_t *c_counter = &rc_c_counter;
KeyComboCounter_t *r_counter = &rc_r_counter;
KeyComboCounter_t *f_counter = &rc_f_counter;





void chassis_vision_ctrl_adjust(Chassis_CMD_data_t *chassis_cmd,minipc_t *minipc);

static unsigned int debug_count = 0;


extern custom_cmd_t custom_cmd;//自定义控制器指令

float Task_T2;

int ref_test=0;
uint8_t uart5_rx_buff[18]={0};

extern RC_ctrl_t rc_ctrl;//遥控器数据
Chassis_CMD_data_t Chassis_CMD_data;
ARM_CMD_data_t ARM_CMD_data;

float RC_SPEED_RATIO=0.6;

void lift_cmd(ARM_CMD_data_t *arm_cmd, int mode,double ref);//上升模式控制
//roll轴模式控制
void roll_cmd(ARM_CMD_data_t *arm_cmd, int mode,double ref);
//视觉控制底盘
void chassis_vision_ctrl(Chassis_CMD_data_t *chassis_cmd,minipc_t *minipc,int speed);
//视觉控制机械臂
void arm_vision_ctrl(ARM_CMD_data_t *arm_cmd,minipc_t *minipc);
//键鼠控制机械臂
void key_ctrl_arm(void);


//键鼠控底盘
void key_ctrl_chassis(void);
//自定义控制器控制 机械臂
void costum_ctrl_arm(void);

//底盘自动模式
void chassis_auto(void);
//图传角度
void video_angle_ctrl(void);
//吸盘控制
void sucker_ctrl(void);

//自动模式跳转中心
void auto_mode_select_centre(void);

//发送所有消息
void normally_set_all_data(void);
//真正的手动模式
void real_handle_mode(void);
//自动模式的预先动作
void get_ready_auto(void);

//正常解算
void normally_tranverse(void);

//yaw轴控制
void yaw_absolute_ctrl(void);

//锁定机械臂
void lock_arm(void);

//自动取银矿
void auto_get_silver(void);

//自动推矿
void push_block(void);

void auto_fetch_block(void);

void auto_put_block(void);//自动收矿
void limit_all_angle_lift(void);//设置限幅

//将电机值返回给minipc
void minipc_send(minipc_t* pc);
	
//传送带手动控制
void trans_single_control(void);
//解算数组
float res_scara_angle[2]; //第一个为大臂，第二个为小臂
static float rc_mode_xy[2]             = {0, 0}; // 目标x,y坐标
static float rc_mode_xy_after_check[2] = {0, 0};//被检查过的

float target_forward = 0; //目标前进速度开根号

uint32_t count_for_modeShift = 0;    //取金矿模式切换倒计时
uint8_t auto_operate_mode    = 0;    // 0为手动模式1为自动取东西模式2为自动放东西模式3为自动取银矿
uint32_t count_for_drop      = 0;    //等待矿石掉落计时
uint32_t count_for_lift      = 0;    //自动取矿石抬升倒计时
uint32_t count_for_suck      = 0;    //等待吸住矿石
uint32_t step_process_backup = 0;    //收矿时机械臂步进式前进标志位
uint32_t count_for_push      = 0;    //推矿计数
float target_height          = 0;    //推矿时目标高度
uint8_t chassis_auto_flag    = 0;    //底盘自动控制标志位
float silver_mode_step       = -1.f; //自动银矿模式步骤


uint32_t count_for_step0     = 0;
uint32_t count_for_step1     = 0;
uint32_t count_for_step1_5   = 0;
uint32_t count_for_step2     = 0;
uint32_t count_for_step2_5   = 0;
uint32_t count_for_step3     = 0;
uint32_t count_for_step4     = 0;
uint32_t count_for_step4_5   = 0;
uint32_t count_for_step5     = 0;
uint32_t count_for_step5_5   = 0;
uint32_t count_for_step6     = 0;
uint32_t count_for_step7     = 0;
uint32_t count_for_step8     = 0;
uint32_t count_for_step9     = 0;

uint32_t use_custom_flag =0;

uint32_t use_absolute_flag=1;//绝对模式控制

float target_angle1     = 0;
float target_angle2     = 0;
float target_angle3     = 0;
float target_angle4     = 0;

volatile float target_lift_speed = 0;
float target_roll       = 0;



uint8_t if_solve_flag      = 1; //大臂小臂是否过解算标志位
float backback_step        = 0; //自动收矿模式步骤
uint8_t backget_step       = 0; //自动从矿仓取矿步骤
uint32_t backback_count0_5 = 0;

//yaw轴绝对角度
static float yaw_absolute= 0;

//正常底盘控制
void normally_chassis_control(void);


void arm_vision_ctrl_adjust(ARM_CMD_data_t *arm_cmd,minipc_t *minipc);//调整后的视觉控制


void video_offline_protect(void);//图传离线保护

pid_type_def y_vision_ctrl_pid,x_vision_ctrl_pid;
void ROBOT_CMD_INIT(void)
{
    HAL_UARTEx_ReceiveToIdle_IT(&huart5,uart5_rx_buff, sizeof(uart5_rx_buff));
	
	  PID_init(&y_vision_ctrl_pid,PID_POSITION,1,0,0,1000,100);
	  PID_init(&x_vision_ctrl_pid,PID_POSITION,1,0,0,1000,100);
}

//使用图传链路还是遥控呢
void	remote_cmd_choose(void);

//roll和loft直接由此控制
void ROBOT_CMD_TASK(void)
{
	ROBOT_STATE=NORMAL;
	
	referee_rc_decode(&video_cmd);//图传遥控

	memcpy(&custom_cmd, &video_cmd.custom_robot_data, 19);//专门针对自定义控制器解码
	
	remote_cmd_choose();
	
	//0点设置
	if(ctrl_flag&&g_flag)
	{
		lift_motor.not_first=0;//重新初始化
		roll.not_first=0;
		roll_real=0;
		temp_roll=0;
		ARM_CMD_data.roll_angle=0;
	}
	Height_Calculation();
	//吸盘控制
	sucker_ctrl();
	video_angle_ctrl();//图传角度控制
	
	if_solve_flag=1;//默认需要解算
	use_custom_flag=0;//默认不用自定义控制器
	
	if(auto_operate_mode==0)//手动模式
	{
		
		//自动模式跳转区间
		auto_mode_select_centre();
		
		//判断遥控器拨杆,如果在下面就用自定义控制
		if((rc_ctrl.rc.s[0]==2)&&rc_cm.state==DEVICE_OK)
		{
			costum_ctrl_arm();
			ROBOT_STATE=CUSTOM;
		}
		else
    {
		//如果在其他地方就用键鼠或拨杆控制
		//几个电机//鼠标遥控
		//roll //鼠标//遥控器
		//抬升 //鼠标//遥控器
		  real_handle_mode();
		//模式预备
		//低头到银矿距离
     
        //取银矿模式预备

        //取金矿模式预备

        //兑矿预备（一般指4级矿）
        //口朝左

        //口朝右
	    //准备好喽！！！！！！！！！！
		  get_ready_auto();
		 //取金矿模式锁机械臂
        lock_arm();
		 //推矿模式
	    push_block();		 	
	    }	
	}
	/******************************自动收矿模式**************************************/
		if(auto_operate_mode==2)
		{
			auto_put_block();
		
		}
	/*************************************自动从矿仓取矿模式****************************************/
		if(auto_operate_mode==1)
		{
			auto_fetch_block();
		}	
	/*************************************自动取银矿模式****************************************/
			if(auto_operate_mode==3)
		{
			auto_get_silver();
		}
		
		
		//离线检查
		video_offline_protect();//很好的保护
		
		
		
			/***********************底盘部分***********************/
	//底盘是否使能
	   if (chassis_auto_flag == 0) {
				//判断是否有w轴			
				//没有欧米伽就是纯横移
			 normally_chassis_control();
			 
			 
    }
//		 
		
		
		
		
		
		/******数值变换区*************是否解算 不解算就由其他方法控制 用于区分直接控制还是解算控制 是每个小步骤中的小心机************************/
	  if (if_solve_flag==1) {
      normally_tranverse();
    }
    else if (if_solve_flag==0) {
			;//什么也不做
    }
	  
		if(use_custom_flag==0)
		{
    // yaw轴旋转量转换到关节电机角度 这个是一定会做的
			if(use_absolute_flag==1){
        yaw_absolute_ctrl();//得到target_angle3
				}

		}
		
		
		
		
	/***********************视觉*************************/
	if(v_counter->ctrl_press_count%2==1)
	{
		chassis_vision_ctrl_adjust(&Chassis_CMD_data,&minipc);
		arm_vision_ctrl_adjust(&ARM_CMD_data, &minipc);
		minipc_send(&minipc);	
	}
	
	
//	limit_all_angle_lift();//限幅
		if(target_lift_speed>10&&height>=580)
		{target_lift_speed=-10;}
		
		VAL_LIMIT(target_angle1, MAXARM_MIN, MAXARM_MAX);
	
    VAL_LIMIT(target_angle2, MINARM_MIN, MINARM_MAX);
	
    if (pitch_motor.para.pos > -0.1f) {
			  VAL_LIMIT(target_angle3, FINE_MIN2, FINE_MAX2);//为何
    } else {
        VAL_LIMIT(target_angle3, FINE_MIN, FINE_MAX);
    }
		
    VAL_LIMIT(target_angle4, PITCH_MIN, PITCH_MAX);
	

	
	  normally_set_all_data();//一些误差在此进行校正

}

//roll轴模式控制
void roll_cmd(ARM_CMD_data_t *arm_cmd, int mode,double ref)
{
	    switch (mode)
    {
    case ROLL_KEEP_MODE:
        arm_cmd->roll_mode=ROLL_KEEP_MODE;
        break;
    case ROLL_ANGLE_MODE:
        arm_cmd->roll_mode=ROLL_ANGLE_MODE;
        arm_cmd->roll_angle=ref;
        break;
    
    default:
        break;
    }
	
}

//上升模式控制
void lift_cmd(ARM_CMD_data_t *arm_cmd, int mode,double ref)//都是正为上
{

    switch (mode)
    {
    case LIFT_KEEP_MODE:
        arm_cmd->lift_mode=LIFT_KEEP_MODE;
        break;
    case LIFT_HEIGHT_MODE:
        arm_cmd->lift_mode=LIFT_HEIGHT_MODE;
        arm_cmd->lift_height=ref;
        break;
    case LIFT_SPEED_MODE:
        arm_cmd->lift_mode=LIFT_SPEED_MODE;
        arm_cmd->lift_speed=ref;
        
        break;
    
    default:
        break;
    }
	
}



/**
 * @brief 视觉控制
 *
 */
static void VisionContorl(void)
{
    
}


/**
 * @brief 图传链路以及自定义控制器的模式和控制量设置
 *
 */

static void VideoControlSet(void)
{
}



//模式预备 好几个if
//取金矿 推矿模式
//自动放东西 取东西

//自动取银矿 放银矿

//0为手动模式1为自动取东西模式2为自动放东西模式3为自动取银矿




//通过直觉控制底盘速度 但是你可能用不到了
void chassis_vision_ctrl(Chassis_CMD_data_t *chassis_cmd,minipc_t *minipc,int speed)
{
//	switch(minipc->minipc2mcu.chassis_direction)
//	{
//		case FOR_BACK://前后模式
//			chassis_cmd->Chassis_straight_mode=STRAIGHT_Y_ON;
//		  chassis_cmd->vy=minipc->minipc2mcu.chassis_ctrl;
//		  
//		  
//			
//			break;
//		case LEFT_RIGHT://左右模式
//			chassis_cmd->Chassis_straight_mode=STRAIGHT_X_ON;
//		  chassis_cmd->vx=minipc->minipc2mcu.chassis_ctrl;
//			
//			break;
//		default:
//			
//			break;
//	}
}

void arm_vision_ctrl(ARM_CMD_data_t *arm_cmd,minipc_t *minipc)//视觉控制 需要控制模式和真正的角度
{
   arm_cmd->maximal_arm_angle=minipc->minipc2mcu.max_angle_ctrl;
   arm_cmd->minimal_arm_angle=minipc->minipc2mcu.min_angle_ctrl;
	 arm_cmd->finesse_angle=minipc->minipc2mcu.finesse_angle_ctrl;
	 arm_cmd->pitch_arm_angle=minipc->minipc2mcu.pitch_angle_ctrl;
	
	//roll轴配置
	if(minipc->minipc2mcu.roll_mode==ANGLE){
		arm_cmd->roll_mode=ROLL_ANGLE_MODE;
	 arm_cmd->roll_angle=minipc->minipc2mcu.roll_angle_ctrl;}//绝对式
	if(minipc->minipc2mcu.roll_mode==SPEED){
	 arm_cmd->roll_mode=ROLL_ANGLE_MODE;
	 arm_cmd->roll_angle+=minipc->minipc2mcu.roll_angle_ctrl;}//增量式
	
		if(minipc->minipc2mcu.roll_mode==KEEP){
	 arm_cmd->roll_mode=ROLL_KEEP_MODE;
	  }//保持 可能要配合增量式使用
	
	//z轴配置
	if(minipc->minipc2mcu.z_mode==ANGLE){
		arm_cmd->lift_mode=LIFT_HEIGHT_MODE;
	 arm_cmd->lift_height=minipc->minipc2mcu.z_ctrl;}//绝对式
	if(minipc->minipc2mcu.z_mode==SPEED){
	 arm_cmd->lift_mode=LIFT_HEIGHT_MODE;
	 arm_cmd->lift_height+=minipc->minipc2mcu.z_ctrl;}//增量式 
	 
	if(minipc->minipc2mcu.z_mode==KEEP){
	 arm_cmd->lift_mode=LIFT_KEEP_MODE;
	  }//增量式 
	
	//没想好速度模式在视觉方面要怎么用
	
}

//将电机值返回给minipc
void minipc_send(minipc_t* pc)
{
	pc->mcu2minipc.max_realangle=(float)max_motor.para.pos;
	pc->mcu2minipc.min_realangle=(float)min_motor.para.pos;
	pc->mcu2minipc.finesse_realangle=(float)finesse_motor.para.pos;
	pc->mcu2minipc.pitch_realangle=(float)pitch_motor.para.pos;
	pc->mcu2minipc.roll_realangle=(float)roll_real;
	pc->mcu2minipc.z_realheight=(float)height;
	
	minipc_upgrade(pc);//发送函数
	CDC_Transmit_HS(pc->mcu2minipc_buf,sizeof(pc->mcu2minipc));
}




//自定义控制器控制 机械臂
void costum_ctrl_arm(void)
{
	target_angle1=custom_cmd.maximal_arm_target;
	target_angle2=custom_cmd.minimal_arm_target;
	target_angle3=custom_cmd.finesse_target;
	target_angle4=custom_cmd.pitch_arm_target;
	
	//还没加上roll
	if(custom_cmd.roll_cmd_direction==1)
	{
		 ARM_CMD_data.roll_mode = ROLL_ANGLE_MODE;//角度模式 增量控制
		 ARM_CMD_data.roll_angle = 1 + roll_real;//增量式控制
	}
	if(custom_cmd.roll_cmd_direction==2)
	{
		 ARM_CMD_data.roll_mode = ROLL_ANGLE_MODE;//角度模式 增量控制
		 ARM_CMD_data.roll_angle = -1 + roll_real;//增量式控制
	}
  if(custom_cmd.roll_cmd_direction==0)
	{
		 ARM_CMD_data.roll_mode = ROLL_KEEP_MODE;//角度模式 增量控制

	}
	

	
	if(custom_cmd.lift_cmd==UP)
	{
		target_lift_speed=400;
	}
	
		if(custom_cmd.lift_cmd==DOWN)
	{
		target_lift_speed=-400;
	}
		if(custom_cmd.lift_cmd==LIFT_KEEP)
	{
		target_lift_speed=0;
	}
	
	
		if(custom_cmd.sucker_cmd==ON)
	{
     ARM_CMD_data.sucker_mode = SUCKER_ON;
	}
	
		if(custom_cmd.sucker_cmd==OFF)
	{
     ARM_CMD_data.sucker_mode = SUCKER_OFF;
	}
	
	if_solve_flag=0;//默认需要解算
  	
	use_custom_flag=1;//使用自定义标志位
	
}

//底盘自动模式
void chassis_auto(void)
{
	
}
	
//图传角度
void video_angle_ctrl(void)
{
	  if (v_counter->single_press_count % 2 == 1) {
        ARM_CMD_data.video_angle = PITCH_120;
    } else {
        ARM_CMD_data.video_angle = PITCH_90;
    }
}
//吸盘控制
void sucker_ctrl(void)
{
	    if (r_counter->single_press_count % 2 == 1) //|| switch_is_up(rc_data[TEMP].rc.switch_right)) {
    {    
        ARM_CMD_data.sucker_mode = SUCKER_ON;
    } else {
        ARM_CMD_data.sucker_mode = SUCKER_OFF;
    }
	
	
}


void auto_mode_select_centre(void)
{
	        /***************自动模式通道****************/
        if (b_counter->shift_press_count % 2 == 1) {
//            ARM_CMD_data.auto_mode=AUTO_OUTPUT ; //开启自动放东西模式
					auto_operate_mode = 2; //开启自动放东西模式
            yaw_absolute      = 0;
        } else {
            count_for_drop = 0;
        }
        if (g_counter->shift_press_count % 2 == 1) {
//            ARM_CMD_data.auto_mode=AUTO_OUTPUT;//自动从矿仓取
					auto_operate_mode = 1; //开启自动取东西模式
            yaw_absolute      = 0;
        }
        if (g_counter->shift_press_count % 2 == 0 && b_counter->shift_press_count % 2 == 0) {
            //trans停下来
					ARM_CMD_data.trans_mode=STOP;
        }
        //自动银矿模式启动
        if (x_counter->shift_press_count % 2 == 1) {
//            ARM_CMD_data.auto_mode=GET_SILVER;
					auto_operate_mode = 3;
            yaw_absolute      = 0;
        }
}



void real_handle_mode(void)//真手动模式
{
	 /******************手动部分********************/
        // xy命令获取
        //鼠标控
        if ((!shift_flag)&&(!ctrl_flag) && (!(press_left && press_right)) ){
            rc_mode_xy[1] -= rc_ctrl.mouse.x / 50;
            rc_mode_xy[0] -= rc_ctrl.mouse.y / 30;
        }
				
        //遥控器控
        if (rc_ctrl.rc.ch[0] > 100) {//上下
            rc_mode_xy[0] += (rc_ctrl.rc.ch[3] ) / 240;
        }
        if (rc_ctrl.rc.ch[0] < -100) {
            rc_mode_xy[0] += (rc_ctrl.rc.ch[3] ) / 240;
        }
        if (rc_ctrl.rc.ch[1] > 100) {
            rc_mode_xy[1] -= (rc_ctrl.rc.ch[1]) / 240;
        }
        if (rc_ctrl.rc.ch[1] < -100) {
            rc_mode_xy[1] -= (rc_ctrl.rc.ch[1]) / 240;
        }
				
				
        //末端三轴
        // yaw pitch,按住ctrl鼠标控制shift？
				//ctrl+两个按键下摁
        if ((!shift_flag)&&(!(press_left && press_right))&& (ctrl_flag)) {
            if (rc_ctrl.mouse.x > 2) {
                yaw_absolute -= 0.0001 * rc_ctrl.mouse.x;
            }
            if (rc_ctrl.mouse.x < -2) {
                yaw_absolute -= 0.0001 * rc_ctrl.mouse.x;
            }
            if (rc_ctrl.mouse.y > 2) {
                target_angle4 -= 0.0002 * rc_ctrl.mouse.y;
            }
            if (rc_ctrl.mouse.y < -2) {
                target_angle4 -= 0.0002 * rc_ctrl.mouse.y;
            }
        }




        // roll??????????????????????????????????????????????????
        if (q_flag || e_flag) {
            ARM_CMD_data.roll_mode = ROLL_ANGLE_MODE;//角度模式 增量控制
					  ARM_CMD_data.roll_angle = ((1.f *e_flag) - (1.f * q_flag)) + roll_real;//增量式控制
        } else {
            ARM_CMD_data.roll_mode = ROLL_KEEP_MODE;//保持
        }
        
				
	
			
        //遥控器拨杆抬
		if(!(press_left || press_right))
		{
				
			
        if (rc_ctrl.rc.s[1]==2) {//左拨杆向下
            target_lift_speed = -300;
        }
        if (rc_ctrl.rc.s[1]==1) {//左拨杆向上
            target_lift_speed = 300;
        }
				if (rc_ctrl.rc.s[1]==3) {//左拨杆中间
            target_lift_speed = 0;
        }
			}
				
		//抬升
        //鼠标抬
			if(!ctrl_flag){
        if (press_left && (!press_right)) {
            target_lift_speed =target_lift_speed+10.f;
        } else if (press_right && (!press_left)) {
            target_lift_speed=target_lift_speed-10.f;
        } else {
            target_lift_speed =0;
        }
			}
				
        if (target_lift_speed > 400) {
            target_lift_speed = 400;
        }
        if (target_lift_speed < -400) {
            target_lift_speed = -400;
        }
				
				

				//控制传送带
				trans_single_control();
				
}

//发送所有消息
void normally_set_all_data(void)
{
	  ARM_CMD_data.maximal_arm_angle = -0.27f + target_angle1;//1.05
    ARM_CMD_data.minimal_arm_angle = -0.009f + target_angle2;//-2.28
    ARM_CMD_data.finesse_angle     = 0.042f + target_angle3;//target_angle3;//向右-6.8~2.74向左  //最右-3.1最左6.08
    ARM_CMD_data.pitch_arm_angle   = -0.094f + target_angle4;//向下-1.4~0.78向上   //roll-180,+180
	  ARM_CMD_data.lift_speed   =  target_lift_speed;
}



//模式预备
void get_ready_auto(void)
{
	        //低头到银矿距离
        if (x_flag) {
            rc_mode_xy[0] = 250;
            target_angle4 = -PI / 2;
            yaw_absolute  = 0;
        }
        //取银矿模式预备
        if (ctrl_flag & x_flag) {
            rc_mode_xy[0]     = 250;
            rc_mode_xy[1]     = 200;
            target_angle4     = -PI / 2;
            yaw_absolute      = 0;
            lift_height_cmd(100, &target_lift_speed);
        }
        //取金矿模式预备
        if (c_flag) {
            rc_mode_xy[1] = 0;
            target_angle4 = 0;
            yaw_absolute  = 0;
        }
        //兑矿预备（一般指4级矿）
        //口朝左
        if (ctrl_flag && q_flag) {
            rc_mode_xy[0] = 250;
            rc_mode_xy[1] = 200;
            yaw_absolute  = -PI / 4;
            target_angle4 = -PI / 4;
        }
        //口朝右
        if (ctrl_flag && e_flag) {
            rc_mode_xy[0] = 330;
            rc_mode_xy[1] = 0;
            yaw_absolute  = PI / 4;
            target_angle4 = -PI / 4;
        }
				


}

//正常解算
void normally_tranverse(void)
{
	        /*****************************xy解算部分**********************************/
        // x限位
        if (rc_mode_xy[0] > ARMLENGHT1 + ARMLENGHT2) {
            rc_mode_xy[0] = ARMLENGHT1 + ARMLENGHT2;
        }
        if (rc_mode_xy[0] < 0) {
            rc_mode_xy[0] = 0;
        }
        // y限位
        if (rc_mode_xy[1] > 400) {
            rc_mode_xy[1] = 400;
        }
        if (rc_mode_xy[1] < -228) {
            rc_mode_xy[1] = -228;
        }
        check_boundary_scara_lefthand(rc_mode_xy[0], rc_mode_xy[1], rc_mode_xy_after_check);
        rc_mode_xy[0] = rc_mode_xy_after_check[0];
        rc_mode_xy[1] = rc_mode_xy_after_check[1];
        scara_inverse_kinematics(rc_mode_xy_after_check[0], rc_mode_xy_after_check[1], ARMLENGHT1, ARMLENGHT2, 2, res_scara_angle);
        target_angle1 = res_scara_angle[0];
        target_angle2 = res_scara_angle[1];
}

void yaw_absolute_ctrl(void)
{

//	  if (min_motor.para.pos > 0.2) {
//        target_angle3 = yaw_absolute - max_motor.para.pos - min_motor.para.pos;//?????????
//    } else {
//        target_angle3 = yaw_absolute - max_motor.para.pos - min_motor.para.pos;
//    }
	target_angle3 = yaw_absolute - max_motor.para.pos - min_motor.para.pos;
//		
//		pos_speed_ctrl(&hfdcan3,  1,  target_angle3+0.1, 1.5);//finesse
}

void lock_arm(void)
{
		/**********************取金矿模式锁机械臂***********************************/
        if (z_counter->shift_press_count % 3 == 0) {//真正决定会不会进入此模式
            count_for_modeShift= 0;
            c_counter->ctrl_press_count = 0; //伸直机械臂命令清零
        }
        if (z_counter->shift_press_count % 3 == 1) {
            lift_height_cmd(80, &target_lift_speed);
            rc_mode_xy[0]       = 330;
            rc_mode_xy[1]       = 0;
            target_angle4       = -0.15;
            yaw_absolute        = 0;
            count_for_modeShift = 0;
            //伸直机械臂
            if (c_counter->ctrl_press_count % 2 == 1) {
                target_angle1 = 0;
                target_angle2 = 0;
                yaw_absolute  = 0;
                target_angle4 = -0.15;
                if_solve_flag = 0; //机械臂不解算
            }
        }
        if (z_counter->shift_press_count % 3 == 2) { //这个模式不会长时间持续
            lift_height_cmd(120, &target_lift_speed);
            rc_mode_xy[0]     = 330;
            rc_mode_xy[1]     = 0;
            target_angle4     = 0;
            yaw_absolute      = 0;
            count_for_modeShift++;
            if (count_for_modeShift > 300 && (c_counter->ctrl_press_count % 2 != 1)) {
                z_counter->shift_press_count = 0;
            }
            if_solve_flag = 0; //机械臂不解算
        }
}


void auto_get_silver(void)
{
	 //开吸盘
        ARM_CMD_data.sucker_mode = SUCKER_ON;
        //其他模式清零
        z_counter->shift_press_count = 0;
        //识别自动模式是否关闭,并清空倒计时的自动标志位
        if (x_counter->shift_press_count % 2 == 0) {
            auto_operate_mode = 0;
            count_for_drop    = 0;
            silver_mode_step  = -1.f;
            count_for_step0   = 0;
            count_for_step1   = 0;
            count_for_step1_5 = 0;
            count_for_step2   = 0;
            count_for_step2_5 = 0;
            count_for_step3   = 0;
            count_for_step4   = 0;
            count_for_step4_5 = 0;
            count_for_step5   = 0;
            count_for_step5_5 = 0;
            count_for_step6   = 0;
            count_for_step7   = 0;
            count_for_step8   = 0;
            count_for_step9   = 0;
            backback_count0_5 = 0;
        }
        if (silver_mode_step == 0) {
            rc_mode_xy[0]     = 250;
            rc_mode_xy[1]     = 200;
            yaw_absolute      = 0;
            target_angle4     = -PI / 2;
            lift_height_cmd(395, &target_lift_speed); 
            count_for_step0++;
            if (count_for_step0 > 150) {
                silver_mode_step = 1;
            }
        }
        if (silver_mode_step == -1.f) {
            silver_mode_step = 0;
        }
        if (silver_mode_step == 1) {
            rc_mode_xy[0]     = 250;
            rc_mode_xy[1]     = 200;
            yaw_absolute      = 0;
            target_angle4     = -PI / 2;
            lift_height_cmd(590, &target_lift_speed); 
            count_for_step1++;
            if (count_for_step1 > 150) {
                silver_mode_step = 1.5f;
            }
        }
        if (silver_mode_step == 1.5f) {
            rc_mode_xy[0] -= 1;
            rc_mode_xy[1]     = 200;
            target_angle4     = -PI / 2;
            lift_height_cmd(590, &target_lift_speed);
            if (rc_mode_xy[0] <= 220) {
                rc_mode_xy[0] = 220;
            }
            count_for_step1_5++;
            if (count_for_step1_5 > 100) {
                silver_mode_step = 2;
            }
        }
        if (silver_mode_step == 2) {
            ARM_CMD_data.trans_mode = GET;
            target_angle4               = -PI / 2;
            if (backback_step == 0) {
                lift_height_cmd(590, &target_lift_speed); 
                rc_mode_xy[0]     = 220;
                rc_mode_xy[1] += 4;
                if (rc_mode_xy[1] >= 400) {
                    backback_step = 0.5f;
                }
                if_solve_flag = 1; //机械臂解算
            }
            if (backback_step == 0.5f) {
                lift_height_cmd(590, &target_lift_speed); 
                target_angle1     = PI / 2;
                target_angle2     = 0.41921f;
							
							  use_absolute_flag=0;//不要绝对控制
							
							  target_angle3            =-0.5;
							
							
                backback_count0_5++;
                if (backback_count0_5 > 170) {
                    backback_step = 1;
                }
                if_solve_flag = 0; //机械臂不解算
            }
            if (backback_step == 1) {
                lift_height_cmd(590, &target_lift_speed);
                target_angle1     = 0.90956f;
                target_angle2     = 2.41921f;
                if (fabs(min_motor.para.pos - 2.41921f) < 0.15) {
                    backback_step = 2;
                }
                if_solve_flag = 0; //机械臂不解算
            }
            if (backback_step == 2) {
                lift_height_cmd(590, &target_lift_speed); 
                target_angle1     = 1.42956f;
                target_angle2     = 2.15921f;
                if (fabs(min_motor.para.pos- 2.15921f) < 0.15) {
                    backback_step = 3;
                }
                if_solve_flag = 0; //机械臂不解算
            }
            if (backback_step == 3) {
                ARM_CMD_data.sucker_mode = SUCKER_OFF;
                lift_height_cmd(350, &target_lift_speed); 
                target_angle1            = 1.42956f;
                target_angle2            = 2.15921f;
							
							  
                count_for_drop++;
                if (count_for_drop > 300) {
                    backback_step = 4;
                }
                if_solve_flag = 0; //机械臂不解算
            }
            if (backback_step == 4) {
                lift_height_cmd(590, &target_lift_speed); 
                rc_mode_xy[0]     = 250;
                rc_mode_xy[1]     = 200;
							
							  use_absolute_flag=1;//不要绝对控制
                float temp_xy[2]  = {0, 0};
                scara_forward_kinematics(max_motor.para.pos, min_motor.para.pos, ARMLENGHT1, ARMLENGHT2, temp_xy);
                if ((fabs(rc_mode_xy[0] - temp_xy[0]) < 40) && (fabs(rc_mode_xy[1] - temp_xy[1]) < 40)) {
                    backback_step            = 0; //步进标志位清零
                    ARM_CMD_data.sucker_mode = SUCKER_ON;
                    silver_mode_step         = 2.5f;
                    count_for_drop           = 0;
                    backback_count0_5        = 0;
                }
                if_solve_flag = 1; //机械臂解算
            }
        }
        if (silver_mode_step == 2.5f) {
            rc_mode_xy[0] = 250;
            rc_mode_xy[1] -= 5;
            yaw_absolute      = 0;
            target_angle4     = -PI / 2;
            lift_height_cmd(590, &target_lift_speed);
            if (rc_mode_xy[1] <= (200 - 270)) {
                rc_mode_xy[1] = (200 - 270);
            }
            count_for_step2_5++;
            if (count_for_step2_5 > 200) {
                silver_mode_step = 3;
            }
        }
        if (silver_mode_step == 3) {
            rc_mode_xy[0]     = 250;
            rc_mode_xy[1]     = (200 - 270);
            yaw_absolute      = 0;
            target_angle4     = -PI / 2;
            lift_height_cmd(0, &target_lift_speed); 
            count_for_step3++;
            if (count_for_step3 > 700) {
                silver_mode_step = 4;
            }
        }
        if (silver_mode_step == 4) {
            rc_mode_xy[0]     = 250;
            rc_mode_xy[1]     = (200 - 270);
            yaw_absolute      = 0;
            target_angle4     = -PI / 2;
            lift_height_cmd(590, &target_lift_speed); 
            count_for_step4++;
            if (count_for_step4 > 150) {
                silver_mode_step = 4.5f;
            }
        }
        if (silver_mode_step == 4.5f) {
            rc_mode_xy[0] -= 1;
            rc_mode_xy[1]     = (200 - 270);
            target_angle4     = -PI / 2;
            lift_height_cmd(590, &target_lift_speed); 
            if (rc_mode_xy[0] <= 220) {
                rc_mode_xy[0] = 220;
            }
            count_for_step4_5++;
            if (count_for_step4_5 > 100) {
                silver_mode_step = 5;
            }
        }
        if (silver_mode_step == 5) {
            ARM_CMD_data.trans_mode = GET;
            target_angle4               = -PI / 2;
            if (backback_step == 0) {
                lift_height_cmd(590, &target_lift_speed); 
                rc_mode_xy[0]     = 220;
                rc_mode_xy[1] += 4;
                if (rc_mode_xy[1] >= 400) {
                    backback_step = 0.5f;
                }
                if_solve_flag = 1; //机械臂解算
            }
            if (backback_step == 0.5f) {
                lift_height_cmd(600, &target_lift_speed); 
                target_angle1     = PI / 2;
                target_angle2     = 0.41921f;
							
								use_absolute_flag=0;//不要绝对控制
							
							  target_angle3            =-0.5;
                backback_count0_5++;
                if (backback_count0_5 > 170) {
                    backback_step = 1;
                }
                if_solve_flag = 0; //机械臂不解算
            }
            if (backback_step == 1) {
                lift_height_cmd(590, &target_lift_speed); 
                target_angle1     = 0.90956f;
                target_angle2     = 2.41921f;
                if (fabs(min_motor.para.pos - 2.41921f) < 0.15) {
                    backback_step = 2;
                }
                if_solve_flag = 0; //机械臂不解算
            }
            if (backback_step == 2) {
                lift_height_cmd(590, &target_lift_speed); 
                target_angle1     = 1.42956f;
                target_angle2     = 2.15921f;
                if (fabs(min_motor.para.pos - 2.15921f) < 0.15) {
                    backback_step = 3;
                }
                if_solve_flag = 0; //机械臂不解算
            }
            if (backback_step == 3) {
                ARM_CMD_data.sucker_mode = SUCKER_OFF;
                lift_height_cmd(350, &target_lift_speed);
                target_angle1            = 1.42956f;
                target_angle2            = 2.15921f;
                count_for_drop++;
                if (count_for_drop > 300) {
                    backback_step = 4;
                }
                if_solve_flag = 0; //机械臂不解算
            }
            if (backback_step == 4) {
                lift_height_cmd(590, &target_lift_speed); 
                rc_mode_xy[0]     = 250;
                rc_mode_xy[1]     = 200;
							  use_absolute_flag=1;//不要绝对控制
                float temp_xy[2]  = {0, 0};
                scara_forward_kinematics(max_motor.para.pos, min_motor.para.pos, ARMLENGHT1, ARMLENGHT2, temp_xy);
                if ((fabs(rc_mode_xy[0] - temp_xy[0]) < 40) && (fabs(rc_mode_xy[1] - temp_xy[1]) < 40)) {
                    backback_step            = 0; //步进标志位清零
                    ARM_CMD_data.sucker_mode = SUCKER_ON;
                    silver_mode_step         = 5.5f;
                    backback_count0_5        = 0;
                }
                if_solve_flag = 1; //机械臂解算
            }
        }
        if (silver_mode_step == 5.5f) {
            rc_mode_xy[0] = 250;
            rc_mode_xy[1] -= 5;
            yaw_absolute      = 0;
            target_angle4     = -PI / 2;
            lift_height_cmd(590, &target_lift_speed); 
            if (rc_mode_xy[1] <= 0) {
                rc_mode_xy[1] = 0;
            }
            count_for_step5_5++;
            if (count_for_step5_5 > 200) {
                silver_mode_step = 6;
            }
        }
        if (silver_mode_step == 6) {
            rc_mode_xy[0]     = 250;
            rc_mode_xy[1]     = 0;
            yaw_absolute      = 0;
            target_angle4     = -PI / 2;
            lift_height_cmd(590, &target_lift_speed); 
            count_for_step6++;
            if (count_for_step6 > 150) {
                //回到手动控制模式
                x_counter->shift_press_count = 0;//肯定
                ARM_CMD_data.sucker_mode                             = SUCKER_ON;
                r_counter->single_press_count            = 1; //设置手动模式吸盘开
            }
        }
}



void push_block(void)
{
	       /*********************推矿模式********************************/
        if (f_flag) {
            chassis_auto_flag = 1; //开启底盘自动模式
            if (count_for_push == 0) {
                target_height = height;
            }
            //球坐标->笛卡尔坐标
            float rho    = 1; //默认长度
            float theta  = yaw_absolute;
            float phi    =pitch_motor.para.pos - 0.1f;
            float push_x = rho * cos(phi) * cos(theta);
            float push_y = rho * cos(phi) * sin(theta);
            float push_z = rho * sin(phi);
            //用于调节各方向基准速度一致
            float x_base      = 0.5f;
            float y_base      = 0.5f;
            float z_base_down = 0.16f;
            float z_base_up   = 0.59f;
            float chassis_x   = 72;
            float chassis_y   = 102;

            float xy_record[2]; //记录
            xy_record[0] = rc_mode_xy[0];
            xy_record[1] = rc_mode_xy[1];
            rc_mode_xy[0] += x_base * push_x;
            rc_mode_xy[1] += y_base * push_y;
            float temp_xy_check[2];
            temp_xy_check[0] = rc_mode_xy[0];
            temp_xy_check[1] = rc_mode_xy[1];

            // x限位
            if (rc_mode_xy[0] > ARMLENGHT1 + ARMLENGHT2) {
                rc_mode_xy[0] = ARMLENGHT1 + ARMLENGHT2;
            }
            if (rc_mode_xy[0] < 0) {
                rc_mode_xy[0] = 0;
            }
            // y限位
            if (rc_mode_xy[1] > 400) {
                rc_mode_xy[1] = 400;
            }
            if (rc_mode_xy[1] < -228) {
                rc_mode_xy[1] = -228;
            }
            check_boundary_scara_lefthand(rc_mode_xy[0], rc_mode_xy[1], rc_mode_xy_after_check);
            rc_mode_xy[0] = rc_mode_xy_after_check[0];
            rc_mode_xy[1] = rc_mode_xy_after_check[1];
            //如果碰到限位
            if (rc_mode_xy[0] != temp_xy_check[0] || rc_mode_xy[1] != temp_xy_check[1]) {
                rc_mode_xy[0]       = xy_record[0];
                rc_mode_xy[1]       = xy_record[1];
                Chassis_CMD_data.vx = -chassis_y * push_y;///////////////////////////////////////////////////////////////////////////////////////
                Chassis_CMD_data.vy = chassis_x * push_x;
            }
            if (push_z >= 0) {
                target_height += z_base_up * push_z;
            } else {
                target_height += z_base_down * push_z;
            }
            if (target_height < 0) { target_height = 0; }
            lift_height_cmd(target_height,&target_lift_speed);
            count_for_push++;
        } else {
            count_for_push    = 0;
            chassis_auto_flag = 0; //关闭底盘自动模式
        }
}
	
void auto_fetch_block(void)
{      
   	//其他模式清零
        z_counter->shift_press_count = 0;
        //识别自动模式是否关闭,并清空倒计时的自动标志位
        if (g_counter->shift_press_count % 2 == 0) {
            auto_operate_mode = 0;
            count_for_lift    = 0;
            count_for_suck    = 0;
        }
        ARM_CMD_data.trans_mode = OUTPUT;
        target_angle4               = -PI / 2 - 0.14;
        ARM_CMD_data.sucker_mode    = SUCKER_ON;
        if (height < 230) //在危险高度以下
        {
            target_lift_speed = 500;
        } else { //可以开始收回
            if (backget_step == 0) {
							  lift_height_cmd(450, &target_lift_speed); 
                rc_mode_xy[0]     = 250;
                rc_mode_xy[1] += 4;
                if (rc_mode_xy[1] >= 400) {
                    backget_step = 1;
                }
                if_solve_flag = 1; //机械臂解算
            }
            if (backget_step == 1) {
							  lift_height_cmd(400, &target_lift_speed); 
                target_angle1     = 1.42956f;
                target_angle2     = 2.11921f;
                if (fabs(min_motor.para.pos - 2.11921f) < 0.15) {
                    backget_step = 2;
                }
                if_solve_flag = 0; //机械臂不解算
            }
            if (backget_step == 2) {
							  lift_height_cmd(260, &target_lift_speed); 
                
                target_angle1     = 1.42956f;
                target_angle2     = 2.11921f;
                count_for_suck++;
                if (count_for_suck > 400) {
                    backget_step = 3;
                }
                if_solve_flag = 0; //机械臂不解算
            }
            //吸上矿之后抬升
            if (backget_step == 3) {
							  lift_height_cmd(550, &target_lift_speed); 
                
                target_angle1     = 1.42956f;
                target_angle2     = 2.11921f;
                count_for_lift++;
                if (count_for_lift > 200) {
                    backget_step = 4;
                }
                ARM_CMD_data.trans_mode = STOP;
                if_solve_flag               = 0; //机械臂不解算
            }
            if (backget_step == 4) {
                lift_height_cmd(550, &target_lift_speed); 
                target_angle1     = 1.02956f;
                target_angle2     = 2.11921f;
                count_for_lift++;
                if (count_for_lift > 300) {
                    backget_step = 5;
                }
                if_solve_flag = 0; //机械臂不解算
            }
            if (backget_step == 5) {
                lift_height_cmd(550, &target_lift_speed); 
                rc_mode_xy[0]     = 250;
                rc_mode_xy[1]     = 0;
                float temp_xy[2]  = {0, 0};
                scara_forward_kinematics(max_motor.para.pos, min_motor.para.pos, ARMLENGHT1, ARMLENGHT2, temp_xy);
                if ((fabs(rc_mode_xy[0] - temp_xy[0]) < 40) && (fabs(rc_mode_xy[1] - temp_xy[1]) < 40)) {
                    //回到手动控制模式
                    g_counter->shift_press_count = 0;
                    ARM_CMD_data.sucker_mode= SUCKER_ON;
                    r_counter->single_press_count= 1; //设置手动模式吸盘开
                    backget_step = 0; //标志位清零
                }
                if_solve_flag = 1; //机械臂解算
            }
        }
}
	

void auto_put_block(void)//自动放东西
{
//其他模式清零
        z_counter->shift_press_count = 0;
        //识别自动模式是否关闭,并清空倒计时的自动标志位
        if (b_counter->shift_press_count % 2 == 0) {
            auto_operate_mode = 0;
            count_for_drop    = 0;
        }
        ARM_CMD_data.trans_mode = GET;
//        target_angle4               = -PI / 2;
				target_angle4               = -PI / 2 - 0.08;
        if (height < 250) //在危险高度以下
        {
            target_lift_speed = 500;
        } else { //可以开始收回
            if (backback_step == 0) {
							lift_height_cmd(550, &target_lift_speed); 
                
                rc_mode_xy[0]     = 250;
                rc_mode_xy[1] += 4;
                if (rc_mode_xy[1] >= 400) {
                    backback_step = 1;
                }
                if_solve_flag = 1; //机械臂解算
            }
            if (backback_step == 1) {
							lift_height_cmd(550, &target_lift_speed); 
                
                target_angle1     = 0.90956f;
                target_angle2     = 2.41921f;
							
							  use_absolute_flag=0;//不要绝对控制
							
							  target_angle3            =-0.5;
							
                if (fabs(min_motor.para.pos- 2.41921f) < 0.15) {
                    backback_step = 2;
                }
                if_solve_flag = 0; //机械臂不解算
            }
            if (backback_step == 2) {
							lift_height_cmd(550, &target_lift_speed); 
                target_angle1     = 1.42956f;
                target_angle2     = 2.15921f;

							
                if (fabs(min_motor.para.pos - 2.15921f) < 0.15) {
                    backback_step = 3;
                }
                if_solve_flag = 0; //机械臂不解算
            }
            if (backback_step == 3) {
                ARM_CMD_data.sucker_mode = SUCKER_OFF;
                lift_height_cmd(375, &target_lift_speed); 
                target_angle1            = 1.42956f;
                target_angle2            = 2.15921f;
							  
							
                count_for_drop++;
                if (count_for_drop > 300) {
                    backback_step = 4;
                }
                if_solve_flag = 0; //机械臂不解算
            }
            if (backback_step == 4) {
                lift_height_cmd(590, &target_lift_speed); 
                rc_mode_xy[0]     = 250;
                rc_mode_xy[1]     = 0;
							
							  use_absolute_flag=1;
							
                float temp_xy[2]  = {0, 0};
                scara_forward_kinematics(max_motor.para.pos, min_motor.para.pos, ARMLENGHT1, ARMLENGHT2, temp_xy);
                if ((fabs(rc_mode_xy[0] - temp_xy[0]) < 40) && (fabs(rc_mode_xy[1] - temp_xy[1]) < 40)) {//坐标差小于一定程度
                    //回到手动控制模式
                    b_counter->shift_press_count = 0;
                    backback_step                                        = 0; //步进标志位清零
                    ARM_CMD_data.sucker_mode                             = SUCKER_ON;
                }
                if_solve_flag = 1; //机械臂解算
            }
        }
	
}
//正常控制底盘
void normally_chassis_control(void)
{
	 //底盘非自动模式
    if (chassis_auto_flag == 0) {
        float speed_scale = 0.14f;
			
			
        //平移缓启动
        if ((d_flag && (!a_flag)) ) {
            Chassis_CMD_data.vx += 10 * speed_scale;
        }
        if ((a_flag && (!d_flag)) ) {
            Chassis_CMD_data.vx -= 10 * speed_scale;
        }
        if ((!a_flag) && (!d_flag)) {
            Chassis_CMD_data.vx = 0;
        }
        if (Chassis_CMD_data.vx > 2000 * speed_scale) {
            Chassis_CMD_data.vx = 2000 * speed_scale;
        }
        if (Chassis_CMD_data.vx < -2000 * speed_scale) {
            Chassis_CMD_data.vx = -2000 * speed_scale;
        }
				
        //前进缓启动
        if ((w_flag && (!s_flag))) {
            target_forward += 6 * speed_scale;
        }
        if ((s_flag && (!w_flag))) {
            target_forward -= 6 * speed_scale;
        }
        if ((!s_flag) && (!w_flag))  {
            target_forward = 0;
        }
        Chassis_CMD_data.vy = target_forward * target_forward * target_forward;//多项式吗 有意思
				
				
				
        if (target_forward > 44) {
            target_forward = 44;
        }
        if (target_forward < -44) {
            target_forward = -44;
        }
				
        if (Chassis_CMD_data.vy > 2000 * speed_scale) {
            Chassis_CMD_data.vy = 2000 * speed_scale;
        }
        if (Chassis_CMD_data.vy < -2000 * speed_scale) {
            Chassis_CMD_data.vy = -2000 * speed_scale;
        }
				
				
        if (press_right && press_left) {
            Chassis_CMD_data.vw = -(float)rc_ctrl.mouse.x * 1 * (target_forward / 13 + 1);
        } else  {
            Chassis_CMD_data.vw = 0;
        } 
				//打算改成自动化的那种
//        switch(f_counter.shift_press_count%3){
//					case 0:
//						Chassis_CMD_data.Chassis_straight_mode=STRAIGHT_OFF;
//					break;
//					case 1:
//						Chassis_CMD_data.Chassis_straight_mode=STRAIGHT_Y_ON;
//					break;
//					case 2:
//						Chassis_CMD_data.Chassis_straight_mode=STRAIGHT_X_ON; 
//					break;
//					default:
//					break;
//				}
    }
}

void chassis_vision_ctrl_adjust(Chassis_CMD_data_t *chassis_cmd,minipc_t *minipc)
{
	chassis_cmd->vx=PID_calc(&x_vision_ctrl_pid,minipc->minipc2mcu.x_erro,0);
  chassis_cmd->vy=PID_calc(&y_vision_ctrl_pid,minipc->minipc2mcu.y_erro,0);
}

void arm_vision_ctrl_adjust(ARM_CMD_data_t *arm_cmd,minipc_t *minipc)//调整后的视觉控制
{
	 target_angle1=minipc->minipc2mcu.max_angle_ctrl;
   target_angle2=minipc->minipc2mcu.min_angle_ctrl;
	 target_angle3=minipc->minipc2mcu.finesse_angle_ctrl;
	 target_angle4=minipc->minipc2mcu.pitch_angle_ctrl;
	
	//z轴配置
	if(minipc->minipc2mcu.z_mode==ANGLE){
//	  if(minipc->minipc2mcu.z_ctrl>=400)
//		{minipc->minipc2mcu.z_ctrl=400;}
//    if(minipc->minipc2mcu.z_ctrl<=-400)
//		{minipc->minipc2mcu.z_ctrl=-400;}
		
	  lift_height_cmd(minipc->minipc2mcu.z_ctrl, &target_lift_speed);
	}//绝对式
	
	
	if(minipc->minipc2mcu.z_mode==SPEED){
		
//		float temp_target_height=height+minipc->minipc2mcu.z_ctrl;
//		lift_height_cmd(temp_target_height, &target_lift_speed);
		lift_height_cmd(minipc->minipc2mcu.z_ctrl, &target_lift_speed);
		
	}//增量式 
	
	
	
	
	
	
	
	//roll轴配置
	if(minipc->minipc2mcu.roll_mode==ANGLE){
		arm_cmd->roll_mode=ROLL_ANGLE_MODE;
	  arm_cmd->roll_angle=minipc->minipc2mcu.roll_angle_ctrl;
	}//绝对式
	
	if(minipc->minipc2mcu.roll_mode==SPEED){
	  arm_cmd->roll_mode=ROLL_ANGLE_MODE;
		
		if(minipc->minipc2mcu.roll_angle_ctrl>=1)
		{minipc->minipc2mcu.roll_angle_ctrl=1;}
		if(minipc->minipc2mcu.roll_angle_ctrl<=-1)
		{minipc->minipc2mcu.roll_angle_ctrl=-1;}
		
	  arm_cmd->roll_angle+=minipc->minipc2mcu.roll_angle_ctrl;
	}//增量式
	
	if(minipc->minipc2mcu.roll_mode==KEEP){
	 arm_cmd->roll_mode=ROLL_KEEP_MODE;
	}//保持 可能要配合增量式使用
	
	
	
	
	 
	
//	 float temp_xy[2]  = {0, 0};
//   scara_forward_kinematics(ARM_CMD_data.maximal_arm_angle, ARM_CMD_data.minimal_arm_angle, ARMLENGHT1, ARMLENGHT2, temp_xy);
//	 rc_mode_xy[0]     = temp_xy[0];
//   rc_mode_xy[1]     = temp_xy[1];
//	 
	 
//	//roll轴配置
//	if(minipc->minipc2mcu.roll_mode==ANGLE){
//		arm_cmd->roll_mode=ROLL_ANGLE_MODE;
//	 arm_cmd->roll_angle=minipc->minipc2mcu.roll_angle_ctrl;}//绝对式
//	if(minipc->minipc2mcu.roll_mode==SPEED){
//	 arm_cmd->roll_mode=ROLL_ANGLE_MODE;
//	 arm_cmd->roll_angle+=minipc->minipc2mcu.roll_angle_ctrl;}//增量式
//	
//		if(minipc->minipc2mcu.roll_mode==KEEP){
//	 arm_cmd->roll_mode=ROLL_KEEP_MODE;
//	  }//保持 可能要配合增量式使用
//		
//	
//	//z轴配置
//	if(minipc->minipc2mcu.z_mode==ANGLE){
//	lift_height_cmd(minipc->minipc2mcu.z_ctrl, &target_lift_speed);
//	}//绝对式
//	
//	if(minipc->minipc2mcu.z_mode==SPEED){
//		float temp_target_height=height+minipc->minipc2mcu.z_ctrl;
//		lift_height_cmd(minipc->minipc2mcu.z_ctrl, &target_lift_speed);
//		
//	}//增量式 
//	 
//	if(minipc->minipc2mcu.z_mode==KEEP){
//	 target_lift_speed=0;
//	 }//增量式 
	
//  没想好速度模式在视觉方面要怎么用
	
}

void limit_all_angle_lift(void)//设置限幅
{
		VAL_LIMIT(target_angle1, MAXARM_MIN, MAXARM_MAX);
	
    VAL_LIMIT(target_angle2, MINARM_MIN, MINARM_MAX);
	
//    if (pitch_motor.para.pos > -0.1f) {
//			VAL_LIMIT(target_angle3, FINE_MIN2, FINE_MAX2);//为何
//    } else {
//        VAL_LIMIT(target_angle3, FINE_MIN, FINE_MAX);
//    }
		
    VAL_LIMIT(target_angle4, PITCH_MIN, PITCH_MAX);
		
		if(target_lift_speed>10&&height>=MAX_HEIGHT)
		{target_lift_speed=-10;}
//		if(target_lift_speed<-10&&height<=20)//高度限幅 这个没必要
//		{target_lift_speed=10;}
		
		
	static float temp_angle1;
	static float temp_angle2;
	static float temp_angle3;
	static float temp_angle4;


	
	if(max_motor.para.pos>1.42&&(target_angle1>temp_angle1))
	{
		target_angle1=temp_angle1;
	}
		if(max_motor.para.pos<-0.3&&(target_angle1<temp_angle1))
	{
		target_angle1=temp_angle1;
	}
	
		if(finesse_motor.para.pos>1.71&&(target_angle3>temp_angle3))
	{
		target_angle3=temp_angle3;
	}
		if(finesse_motor.para.pos<-1.11&&(target_angle3<temp_angle3))
	{
		target_angle3=temp_angle3;
	}
	

		if(pitch_motor.para.pos<-1.64&&(target_angle4<temp_angle4))
	{
		target_angle4=temp_angle4;
	}
	
	
	temp_angle1=target_angle1;
	temp_angle2=target_angle2;
	temp_angle3=target_angle3;
	temp_angle4=target_angle4;
		
}

void	remote_cmd_choose(void)
{



	if((video_cm.state==DEVICE_OK)&&(rc_cm.state==DEVICE_NOT_CONNECTED))
	{//遥控离线 使用图传链路
			
	    	rc_ctrl.mouse.x = video_cmd.remote_control.mouse_x;                             //!< Mouse X axis
        rc_ctrl.mouse.y = video_cmd.remote_control.mouse_y;                             //!< Mouse Y axis
        rc_ctrl.mouse.z = video_cmd.remote_control.mouse_z;                             //!< Mouse Z axis
			
	      memcpy(&rf_v_counter, v_counter, sizeof(KeyComboCounter_t));
        memcpy(&rf_b_counter, b_counter, sizeof(KeyComboCounter_t));
        memcpy(&rf_g_counter, g_counter, sizeof(KeyComboCounter_t));
        memcpy(&rf_x_counter, x_counter, sizeof(KeyComboCounter_t));
        memcpy(&rf_z_counter, z_counter, sizeof(KeyComboCounter_t));
        memcpy(&rf_c_counter, c_counter, sizeof(KeyComboCounter_t));
        memcpy(&rf_r_counter, r_counter, sizeof(KeyComboCounter_t));
        memcpy(&rf_f_counter, f_counter, sizeof(KeyComboCounter_t));

        v_counter = &rf_v_counter;
        b_counter = &rf_b_counter;
        g_counter = &rf_g_counter;
        x_counter = &rf_x_counter;
        z_counter = &rf_z_counter;
        c_counter = &rf_c_counter;
        r_counter = &rf_r_counter;
        f_counter = &rf_f_counter;
				// 将所有 rc_ 打头的按键标志清零
        rc_w_flag = 0;
        rc_s_flag = 0;
        rc_a_flag = 0;
        rc_d_flag = 0;
        rc_q_flag = 0;
        rc_e_flag = 0;
        rc_shift_flag = 0;
        rc_ctrl_flag = 0;

        rc_press_left = 0;
        rc_press_right = 0;

        rc_r_flag = 0;
        rc_f_flag = 0;
        rc_g_flag = 0;
        rc_z_flag = 0;
        rc_x_flag = 0;
        rc_c_flag = 0;
        rc_v_flag = 0;
        rc_b_flag = 0;



			
			
	}
		
		if(((video_cm.state==DEVICE_NOT_CONNECTED)&&(rc_cm.state==DEVICE_OK))
			||((video_cm.state==DEVICE_OK)&&(rc_cm.state==DEVICE_OK)))//同时可用 或者 图传离线 使用遥控链路
	{

		    memcpy(&rc_v_counter, v_counter, sizeof(KeyComboCounter_t));
        memcpy(&rc_b_counter, b_counter, sizeof(KeyComboCounter_t));
        memcpy(&rc_g_counter, g_counter, sizeof(KeyComboCounter_t));
        memcpy(&rc_x_counter, x_counter, sizeof(KeyComboCounter_t));
        memcpy(&rc_z_counter, z_counter, sizeof(KeyComboCounter_t));
        memcpy(&rc_c_counter, c_counter, sizeof(KeyComboCounter_t));
        memcpy(&rc_r_counter, r_counter, sizeof(KeyComboCounter_t));
        memcpy(&rc_f_counter, f_counter, sizeof(KeyComboCounter_t));

        v_counter = &rc_v_counter;
        b_counter = &rc_b_counter;
        g_counter = &rc_g_counter;
        x_counter = &rc_x_counter;
        z_counter = &rc_z_counter;
        c_counter = &rc_c_counter;
        r_counter = &rc_r_counter;
        f_counter = &rc_f_counter;
        // 将所有 rf_ 打头的按键标志清零
        rf_w_flag = 0;
        rf_s_flag = 0;
        rf_a_flag = 0;
        rf_d_flag = 0;
        rf_q_flag = 0;
        rf_e_flag = 0;
        rf_shift_flag = 0;
        rf_ctrl_flag = 0;

        rf_press_left = 0;
        rf_press_right = 0;

        rf_r_flag = 0;
        rf_f_flag = 0;
        rf_g_flag = 0;
        rf_z_flag = 0;
        rf_x_flag = 0;
        rf_c_flag = 0;
        rf_v_flag = 0;
        rf_b_flag = 0;

		
	}
		
	

        w_flag     = rc_w_flag     || rf_w_flag;
        s_flag     = rc_s_flag     || rf_s_flag;
        a_flag     = rc_a_flag     || rf_a_flag;
        d_flag     = rc_d_flag     || rf_d_flag;
        q_flag     = rc_q_flag     || rf_q_flag;
        e_flag     = rc_e_flag     || rf_e_flag;
        shift_flag = rc_shift_flag || rf_shift_flag;
        ctrl_flag  = rc_ctrl_flag  || rf_ctrl_flag;

        press_left =	rc_press_left||rf_press_left;
        press_right =	rc_press_right||rf_press_right;	

        r_flag     = rc_r_flag     || rf_r_flag;
        f_flag     = rc_f_flag     || rf_f_flag;
        g_flag     = rc_g_flag     || rf_g_flag;
        z_flag     = rc_z_flag     || rf_z_flag;
        x_flag     = rc_x_flag     || rf_x_flag;
        c_flag     = rc_c_flag     || rf_c_flag;
        v_flag     = rc_v_flag     || rf_v_flag;
        b_flag     = rc_b_flag     || rf_b_flag;

}

void video_offline_protect(void)//图传离线保护
{
	if(video_cm.state==DEVICE_NOT_CONNECTED&&rc_cm.state==DEVICE_NOT_CONNECTED)
	{
		Chassis_CMD_data.vy=0;
		Chassis_CMD_data.vx=0;
		Chassis_CMD_data.vw=0;	
		ARM_CMD_data.sucker_mode    = SUCKER_ON;
		
    		
	}
}
//传送带手动控制
void trans_single_control(void)
{
	if(z_counter->single_press_count%3==0)
	{
		ARM_CMD_data.trans_mode=STOP;
	}
	if(z_counter->single_press_count%3==1)
	{
		ARM_CMD_data.trans_mode=GET;
	}
	if(z_counter->single_press_count%3==2)
	{
		ARM_CMD_data.trans_mode=OUTPUT;
	}
		
}
//如果在算法库中直接return后就可以跳过错误的值而不执行