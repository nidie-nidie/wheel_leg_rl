#include "Music_Task.h"
#include "cmsis_os.h"

#include "bsp_buzzer.h"
// #include "data_exchange.h"
#include "KFIFO.h"
#include "Motor.h"
#include "Config.h"
#include "music.h"
#include "music_calibrate.h"
#include "music_canon.h"
#include "music_castle_in_the_sky.h"
#include "music_deja_vu.h"
#include "music_error.h"
#include "music_gong_xi_fa_cai.h"
#include "music_hao_yun_lai.h"
#include "music_meow.h"
#include "music_motor_offline.h"
#include "music_rc_offline.h"
#include "music_referee.h"
#include "music_see_you_again.h"
#include "music_start.h"
#include "music_typedef.h"
#include "music_unity.h"
#include "music_you.h"
#include "music_crychic.h"
#include "music_vbat.h"
// #include "referee.h"
#include "PS2_Task.h"
#include "Remote_Control.h"
#include "stm32h7xx_hal.h"

#if INCLUDE_uxTaskGetStackHighWaterMark
uint32_t music_high_water;
#endif

#define ABNORMAL_WARNING_INTERVAL 3000 // ms

#define STEP_INIT 1
#define STEP_NORMAL 2

#define PLAY_LIST_FIFO_BUF_LENGTH 10

#define is_play_cali()                                                                 \
    (is_play == CALI_BEGIN || is_play == CALI_MIDDLE_TIME || is_play == CALI_GIMBAL || \
     is_play == CALI_IMU || is_play == CALI_CHASSIS)

#define SET_MUSIC_TO_PLAY(MUSIC_INDEX) \
    if (is_play < MUSIC_INDEX)         \
    {                                  \
        is_play = MUSIC_INDEX;         \
        play_id = 0;                   \
    }

// Enum Declarations，根据优先级排列，越后面优先级越高
typedef enum
{
    PLAY_NONE = 0,
    POWER_UP,
    CALI_BEGIN,
    CALI_MIDDLE_TIME,
    CALI_GIMBAL,
    CALI_IMU,
    CALI_CHASSIS,
    PLAY_MOTOR_OFFLINE,
    PLAY_RC_OFFLINE,
    REFEREE_OFFLINE,
    VBAT_LOW,
} Playing_e;

typedef enum
{
    start = 0,
    motor_offline,
    rc_offline,
    canon,
    castle_in_the_sky,
    deja_vu,
    error,
    gong_xi_fa_cai,
    hao_yun_lai,
    meow,
    referee,
    see_you_again,
    unity,
    you,
    crychic,
    vbat_low,
} MusicIndex_e;

// Variable Declarations
static uint8_t music_step = STEP_INIT;

Playing_e is_play = POWER_UP;
kfifo *play_list_fifo = NULL;

static MusicInfo_s MUSICS[20];
static uint32_t now = 0;
static uint32_t start_time = 0;
static uint32_t play_id = 0; // 0为保留音符，不使用

static uint32_t task_count = 0;
static uint32_t last_abnormal_warning_time = 0;
// static uint32_t last_task_time = 0;
// static uint32_t task_duration = 0;

/**
 * @brief 播放音乐
 * @param  none
 * @return 结束1 未结束0
 */
bool PlayMusic(MusicInfo_s *music_info, float volume)
{
    now = HAL_GetTick();
    bool end = false;

    if (play_id == 0)
    {
        start_time = now;
        play_id++;
        buzzer_note(music_info->notes[play_id].note, volume);
    }
    if (now - start_time >= music_info->notes[play_id].end)
    {
        play_id++;
        if (play_id > music_info->last_note_id)
        {
            end = true;
            play_id = 0;
            buzzer_note(0, 0.1);
        }
        else
        {
            buzzer_note(music_info->notes[play_id].note, volume);
        }
    }

    return end;
}

/*******************************************************************************/
/* Main Functions                                                              */
/*     Music_Task                                                              */
/*     MusicInit                                                               */
/*     MusicPlay                                                               */
/*******************************************************************************/

static void MusicInit(void);
static void MusicPlay(void);

void Music_Task(void)
{
    // 初始化音乐
    MusicInit();

    while (1)
    {
        task_count++;
        MusicPlay();
        // 系统延时
        vTaskDelay(MUSIC_TASK_TIME_MS);

#if INCLUDE_uxTaskGetStackHighWaterMark
        music_high_water = uxTaskGetStackHighWaterMark(NULL);
#endif
    }
}

static void MusicInit(void)
{
    play_list_fifo = kfifo_alloc(PLAY_LIST_FIFO_BUF_LENGTH);

    music_step = STEP_INIT;

    // clang-format off
    MUSICS[start]             = MusicStartInit();
    MUSICS[motor_offline]     = MusicMotorOfflineInit();
    MUSICS[rc_offline]        = MusicRcOfflineInit();
    // MUSICS[you]               = MusicYouInit();
    // MUSICS[unity]             = MusicUnityInit();
    // MUSICS[canon]             = MusicCanonInit();
    // MUSICS[castle_in_the_sky] = MusicCastleInTheSkyInit();
    // MUSICS[see_you_again]     = MusicSeeYouAgainInit();
    MUSICS[hao_yun_lai]       = MusicHaoYunLaiInit();
    MUSICS[gong_xi_fa_cai]    = MusicGongXiFaCaiInit();
    MUSICS[crychic]           = MusicCrychicInit();
    MUSICS[vbat_low]             = MusicVBatLowInit();
    // MUSICS[deja_vu]           = MusicDejaVuInit();
    // clang-format on
}

static void MusicPlay(void)
{
    if (music_step == STEP_INIT)
    { // 开机
        if (PlayMusic(&MUSICS[start], 0.5f))
        {
            music_step = STEP_NORMAL;
            is_play = PLAY_NONE;
        }
    }
    else
    { // 正常状态
        if (HAL_GetTick() - last_abnormal_warning_time > ABNORMAL_WARNING_INTERVAL)
        {
            last_abnormal_warning_time = HAL_GetTick();

            if ((ENABLE_ALARM_RC_OFFLINE && GetRcOffline()) || (ENABLE_ALARM_PS2_OFFLINE && ps2_lost))
            { // 检测遥控器是否离线
                is_play = PLAY_RC_OFFLINE;
                __kfifo_put(play_list_fifo, (unsigned char *)&is_play, sizeof(is_play));
            }
            if (ENABLE_ALARM_MOTOR_OFFLINE && ScanOfflineMotor())
            { // 检测是否存在离线电机
                is_play = PLAY_MOTOR_OFFLINE;
                __kfifo_put(play_list_fifo, (unsigned char *)&is_play, sizeof(is_play));
            }
            // if (ENABLE_CHECK_REFEREE_OFFLINE && GetRefereeOffline())
            // { // 检测裁判系统是否离线
            //     __kfifo_put(&play_list_fifo, (unsigned char *)&REFEREE_OFFLINE, sizeof(REFEREE_OFFLINE));
            // }
            if (ENABLE_ALARM_VBAT_LOW && vbus_low_warning)
            { // 检测电池电压过低
                is_play = VBAT_LOW;
                __kfifo_put(play_list_fifo, (unsigned char *)&is_play, sizeof(is_play));
            }
        }

        // 播放列表内容
        if (kfifo_len(play_list_fifo) > 0 && is_play == PLAY_NONE)
        {
            __kfifo_get(play_list_fifo, (unsigned char *)&is_play, sizeof(is_play));
            play_id = 0;
        }

        // 根据is_play播放对应音乐
        switch (is_play)
        {
        case POWER_UP:
        {
            if (PlayMusic(&MUSICS[start], 0.5f))
                is_play = PLAY_NONE;
        }
        break;

        case PLAY_MOTOR_OFFLINE:
        {
            if (PlayMusic(&MUSICS[motor_offline], 0.5f))
                is_play = PLAY_NONE;
        }
        break;

        case PLAY_RC_OFFLINE:
        {
            if (PlayMusic(&MUSICS[rc_offline], 0.5f))
                is_play = PLAY_NONE;
        }
        break;

        case REFEREE_OFFLINE:
        {
            if (PlayMusic(&MUSICS[error], 0.5f))
                is_play = PLAY_NONE;
        }
        break;

        case VBAT_LOW:
        {
            if (PlayMusic(&MUSICS[vbat_low], 0.5f))
                is_play = PLAY_NONE;
        }
        break;

        default:
        {
            buzzer_off();
        }
        break;
        }
    }
}
