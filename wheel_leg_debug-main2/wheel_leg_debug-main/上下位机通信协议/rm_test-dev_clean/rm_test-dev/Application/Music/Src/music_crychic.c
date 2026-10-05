#include "music_crychic.h"

#include "bsp_buzzer.h"
#include "music.h"
#include "stm32h7xx_hal.h"

// https://www.bilibili.com/opus/1025244505008242694

// 定义中音
#define B1 262
#define B2 296
#define B3 330
#define B4 349
#define B5 392
#define B5p 415
#define B6 440
#define B7 494

// 定义高音
#define C1 523
#define C2 587
#define C2p 622
#define C3 659
#define C4 698
#define C4p 741
#define C5 784
#define C6 880
#define C7 988

// 定义高二度
#define D1 1047
#define D2 1175
#define D3 1319
#define D4 1397
#define D5 1568
#define D6 1760
#define D7 1976

// 定义低音与休止符
#define A5 196
#define A6 220
#define A7 247

// 定义节拍 6/8拍
#define ONE_EIGHTH 250
#define T_16 (ONE_EIGHTH / 2)              // 十六分音符 (=), 1/16
#define T_8 (ONE_EIGHTH)                   // 八分音符 (_), 1/8
#define T_8D (ONE_EIGHTH + ONE_EIGHTH / 2) // 附点八分音符 (_. ), 3/16
#define T_4 (ONE_EIGHTH * 2)               // 四分音符 (无下划线), 1/4
#define T_4D (ONE_EIGHTH * 3)              // 附点四分音符 (. ), 3/8
// clang-format on

#define NOTE_NUM 450
static Note Notes[NOTE_NUM]; // Array of notes
static MusicInfo_s MUSIC_INFO;

/*-------------------- User functions --------------------*/

MusicInfo_s MusicCrychicInit(void)
{
    MUSIC_INFO.notes = Notes;

    // ================= 前奏 =================
    // 3^ 2^_ 1^ 2^_ |
    WRITE_NOTE(C3, T_4);
    WRITE_NOTE(C2, T_8);
    WRITE_NOTE(C1, T_4);
    WRITE_NOTE(C2, T_8);

    // 3^ . 4^= 3^= 2^ . |
    WRITE_NOTE(C3, T_4);
    WRITE_NOTE(C4, T_16);
    WRITE_NOTE(C3, T_16);
    WRITE_NOTE(C2, T_4D);

    // 3^ 2^_ 1^ 2^_ |
    WRITE_NOTE(C3, T_4);
    WRITE_NOTE(C2, T_8);
    WRITE_NOTE(C1, T_4);
    WRITE_NOTE(C2, T_8);

    // 3^ . 4^= 3^= 2^ . |
    WRITE_NOTE(C3, T_4);
    WRITE_NOTE(C4, T_16);
    WRITE_NOTE(C3, T_16);
    WRITE_NOTE(C2, T_4D);

    // 3^ 2^_ 1^ 2^_ |
    WRITE_NOTE(C3, T_4);
    WRITE_NOTE(C2, T_8);
    WRITE_NOTE(C1, T_4);
    WRITE_NOTE(C2, T_8);

    // 3^ . 4^= 3^= 2^ . |
    WRITE_NOTE(C3, T_4);
    WRITE_NOTE(C4, T_16);
    WRITE_NOTE(C3, T_16);
    WRITE_NOTE(C2, T_4D);

    // 3^ 2^_ 1^ 2^_ |
    WRITE_NOTE(C3, T_4);
    WRITE_NOTE(C2, T_8);
    WRITE_NOTE(C1, T_4);
    WRITE_NOTE(C2, T_8);

    // 3^_. 4^= 3^_ 2^ 1= 2= |
    WRITE_NOTE(C3, T_8D);
    WRITE_NOTE(C4, T_16);
    WRITE_NOTE(C3, T_8);
    WRITE_NOTE(C2, T_4);
    WRITE_NOTE(B1, T_16);
    WRITE_NOTE(B2, T_16);

    // ================= 主歌 1 =================
    // 3_ 3_ 2_ 4_ 3_ 2_ |
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B2, T_8);
    WRITE_NOTE(B4, T_8);
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B2, T_8);

    // 2_ 2_ 1= 1= 4_ 3_ 2_ |
    WRITE_NOTE(B2, T_8);
    WRITE_NOTE(B2, T_8);
    WRITE_NOTE(B1, T_16);
    WRITE_NOTE(B1, T_16);
    WRITE_NOTE(B4, T_8);
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B2, T_8);

    // 2 1= 2= 3 . |
    WRITE_NOTE(B2, T_4);
    WRITE_NOTE(B1, T_16);
    WRITE_NOTE(B2, T_16);
    WRITE_NOTE(B3, T_4D);

    // 0 . 3_ 5_ 1^= 7= |
    SLEEP_NOTE(T_4D);
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B5, T_8);
    WRITE_NOTE(C1, T_16);
    WRITE_NOTE(B7, T_16);

    // 7 1^_ 7 1^_ |
    WRITE_NOTE(B7, T_4);
    WRITE_NOTE(C1, T_8);
    WRITE_NOTE(B7, T_4);
    WRITE_NOTE(C1, T_8);

    // 7= 6= 5 5_ 2_ 4_ |
    WRITE_NOTE(B7, T_16);
    WRITE_NOTE(B6, T_16);
    WRITE_NOTE(B5, T_4);
    WRITE_NOTE(B5, T_8);
    WRITE_NOTE(B2, T_8);
    WRITE_NOTE(B4, T_8);

    // 4 3_ 3 5._ |
    WRITE_NOTE(B4, T_4);
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B3, T_4);
    WRITE_NOTE(A5, T_8);

    // 4_ 3_ 2_ 3 5_ |
    WRITE_NOTE(B4, T_8);
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B2, T_8);
    WRITE_NOTE(B3, T_4);
    WRITE_NOTE(B5, T_8);

    // 1 . 0 1_ |
    WRITE_NOTE(B1, T_4D);
    SLEEP_NOTE(T_4);
    WRITE_NOTE(B1, T_8);

    // 2_ 1_. 1= 1_ 5_ 1_ |
    WRITE_NOTE(B2, T_8);
    WRITE_NOTE(B1, T_8D);
    WRITE_NOTE(B1, T_16);
    WRITE_NOTE(B1, T_8);
    WRITE_NOTE(B5, T_8);
    WRITE_NOTE(B1, T_8);

    // 4 3_ 2_ 1_ 1_ |
    WRITE_NOTE(B4, T_4);
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B2, T_8);
    WRITE_NOTE(B1, T_8);
    WRITE_NOTE(B1, T_8);

    // 1 . 0 1= 2= |
    WRITE_NOTE(B1, T_4D);
    SLEEP_NOTE(T_4);
    WRITE_NOTE(B1, T_16);
    WRITE_NOTE(B2, T_16);

    // ================= 主歌 2 =================
    // 3_ 3_ 2_ 4_ 3_ 2_ |
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B2, T_8);
    WRITE_NOTE(B4, T_8);
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B2, T_8);

    // 2_ 2_ 1_ 4_ 3_ 2_ |
    WRITE_NOTE(B2, T_8);
    WRITE_NOTE(B2, T_8);
    WRITE_NOTE(B1, T_8);
    WRITE_NOTE(B4, T_8);
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B2, T_8);

    // 2 1= 2= 3 . |
    WRITE_NOTE(B2, T_4);
    WRITE_NOTE(B1, T_16);
    WRITE_NOTE(B2, T_16);
    WRITE_NOTE(B3, T_4D);

    // 0 . 3_ 5_ 1^_ |
    SLEEP_NOTE(T_4D);
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B5, T_8);
    WRITE_NOTE(C1, T_8);

    // 7 1^_ 7 1^_ |
    WRITE_NOTE(B7, T_4);
    WRITE_NOTE(C1, T_8);
    WRITE_NOTE(B7, T_4);
    WRITE_NOTE(C1, T_8);

    // 7= 6= 5 5_ 2_ 4_ |
    WRITE_NOTE(B7, T_16);
    WRITE_NOTE(B6, T_16);
    WRITE_NOTE(B5, T_4);
    WRITE_NOTE(B5, T_8);
    WRITE_NOTE(B2, T_8);
    WRITE_NOTE(B4, T_8);

    // 4_ 3_ 3_ 3 5._ |
    WRITE_NOTE(B4, T_8);
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B3, T_4);
    WRITE_NOTE(A5, T_8);

    // 4_ 3_ 2_ 3 5_ |
    WRITE_NOTE(B4, T_8);
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B2, T_8);
    WRITE_NOTE(B3, T_4);
    WRITE_NOTE(B5, T_8);

    // 1 . 0 1= 1= |
    WRITE_NOTE(B1, T_4D);
    SLEEP_NOTE(T_4);
    WRITE_NOTE(B1, T_16);
    WRITE_NOTE(B1, T_16);

    // 2_ 1 1_ 5_ 1_ |
    WRITE_NOTE(B2, T_8);
    WRITE_NOTE(B1, T_4);
    WRITE_NOTE(B1, T_8);
    WRITE_NOTE(B5, T_8);
    WRITE_NOTE(B1, T_8);

    // 4_ 4= 4= 3= 2= 2_ 1_ 7._ |
    WRITE_NOTE(B4, T_8);
    WRITE_NOTE(B4, T_16);
    WRITE_NOTE(B4, T_16);
    WRITE_NOTE(B3, T_16);
    WRITE_NOTE(B2, T_16);
    WRITE_NOTE(B2, T_8);
    WRITE_NOTE(B1, T_8);
    WRITE_NOTE(A7, T_8);

    // 1 . 0 . |
    WRITE_NOTE(B1, T_4D);
    SLEEP_NOTE(T_4D);

    // ================= 导歌 (Pre-Chorus) =================
    // 6_ 5_ 5_ 5_ 4_ 4_ |
    WRITE_NOTE(B6, T_8);
    WRITE_NOTE(B5, T_8);
    WRITE_NOTE(B5, T_8);
    WRITE_NOTE(B5, T_8);
    WRITE_NOTE(B4, T_8);
    WRITE_NOTE(B4, T_8);

    // 3_ 2_ 2_ 2 5_ |
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B2, T_8);
    WRITE_NOTE(B2, T_8);
    WRITE_NOTE(B2, T_4);
    WRITE_NOTE(B5, T_8);

    // 5_ 4= 4= 4_ 4_ 3_ 2_ |
    WRITE_NOTE(B5, T_8);
    WRITE_NOTE(B4, T_16);
    WRITE_NOTE(B4, T_16);
    WRITE_NOTE(B4, T_8);
    WRITE_NOTE(B4, T_8);
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B2, T_8);

    // 2 1= 7.= 1 . |
    WRITE_NOTE(B2, T_4);
    WRITE_NOTE(B1, T_16);
    WRITE_NOTE(A7, T_16);
    WRITE_NOTE(B1, T_4D);

    // 6_ 5_ 5_ 5_ 4_ 4_ |
    WRITE_NOTE(B6, T_8);
    WRITE_NOTE(B5, T_8);
    WRITE_NOTE(B5, T_8);
    WRITE_NOTE(B5, T_8);
    WRITE_NOTE(B4, T_8);
    WRITE_NOTE(B4, T_8);

    // 3_ 2_ 2_ 2 3_ |
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B2, T_8);
    WRITE_NOTE(B2, T_8);
    WRITE_NOTE(B2, T_4);
    WRITE_NOTE(B3, T_8);

    // 4_ 3= 3= 3= 3= 3_ 2_ 3_ |
    WRITE_NOTE(B4, T_8);
    WRITE_NOTE(B3, T_16);
    WRITE_NOTE(B3, T_16);
    WRITE_NOTE(B3, T_16);
    WRITE_NOTE(B3, T_16);
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B2, T_8);
    WRITE_NOTE(B3, T_8);

    // 2^ 1^_ 1^ 1^_ |
    WRITE_NOTE(C2, T_4);
    WRITE_NOTE(C1, T_8);
    WRITE_NOTE(C1, T_4);
    WRITE_NOTE(C1, T_8);

    // 7 6_ 6 . |
    WRITE_NOTE(B7, T_4);
    WRITE_NOTE(B6, T_8);
    WRITE_NOTE(B6, T_4D);

    // 0 6_ 6_ 5_ 4= 4= |
    SLEEP_NOTE(T_8);
    WRITE_NOTE(B6, T_8);
    WRITE_NOTE(B6, T_8);
    WRITE_NOTE(B5, T_8);
    WRITE_NOTE(B4, T_16);
    WRITE_NOTE(B4, T_16);

    // 4 . 3= 4= 5 |
    WRITE_NOTE(B4, T_4D);
    WRITE_NOTE(B3, T_16);
    WRITE_NOTE(B4, T_16);
    WRITE_NOTE(B5, T_4);

    // 5 . 0 . |
    WRITE_NOTE(B5, T_4D);
    SLEEP_NOTE(T_4D);

    // ================= 副歌前奏 (Chorus 1) =================
    // 3= 2= 3= 2= 3= 4= 5 4= 5= |
    WRITE_NOTE(B3, T_16);
    WRITE_NOTE(B2, T_16);
    WRITE_NOTE(B3, T_16);
    WRITE_NOTE(B2, T_16);
    WRITE_NOTE(B3, T_16);
    WRITE_NOTE(B4, T_16);
    WRITE_NOTE(B5, T_4);
    WRITE_NOTE(B4, T_16);
    WRITE_NOTE(B5, T_16);

    // 6 6= 7= 1^ 2^= 1^= |
    WRITE_NOTE(B6, T_4);
    WRITE_NOTE(B6, T_16);
    WRITE_NOTE(B7, T_16);
    WRITE_NOTE(C1, T_4);
    WRITE_NOTE(C2, T_16);
    WRITE_NOTE(C1, T_16);

    // 5 5_ 5_ 4_ 4_ |
    WRITE_NOTE(B5, T_4);
    WRITE_NOTE(B5, T_8);
    WRITE_NOTE(B5, T_8);
    WRITE_NOTE(B4, T_8);
    WRITE_NOTE(B4, T_8);

    // 3 3= 4= 5 . |
    WRITE_NOTE(B3, T_4);
    WRITE_NOTE(B3, T_16);
    WRITE_NOTE(B4, T_16);
    WRITE_NOTE(B5, T_4);

    // 3= 2= 3= 2= 3= 4= 5 4= 5= |
    WRITE_NOTE(B3, T_16);
    WRITE_NOTE(B2, T_16);
    WRITE_NOTE(B3, T_16);
    WRITE_NOTE(B2, T_16);
    WRITE_NOTE(B3, T_16);
    WRITE_NOTE(B4, T_16);
    WRITE_NOTE(B5, T_4);
    WRITE_NOTE(B4, T_16);
    WRITE_NOTE(B5, T_16);

    // 6 #5= 6= 7 0= #5= |
    WRITE_NOTE(B6, T_4);
    WRITE_NOTE(B5p, T_16);
    WRITE_NOTE(B6, T_16);
    WRITE_NOTE(B7, T_4);
    SLEEP_NOTE(T_16);
    WRITE_NOTE(B5p, T_16);

    // 3^_ 3^_ 0= 6= 4^_ 3^_ 2^_ |
    WRITE_NOTE(C3, T_8);
    WRITE_NOTE(C3, T_8);
    SLEEP_NOTE(T_16);
    WRITE_NOTE(B6, T_16);
    WRITE_NOTE(C4, T_8);
    WRITE_NOTE(C3, T_8);
    WRITE_NOTE(C2, T_8);

    // 2^_. 1^= 1^= 7= 1^ 5_ 1^_ |
    WRITE_NOTE(C2, T_8D);
    WRITE_NOTE(C1, T_16);
    WRITE_NOTE(C1, T_16);
    WRITE_NOTE(B7, T_16);
    WRITE_NOTE(C1, T_4);
    WRITE_NOTE(B5, T_8);
    WRITE_NOTE(C1, T_8);

    // 2^_ 1^_ 1^_ 1^ 5_ |
    WRITE_NOTE(C2, T_8);
    WRITE_NOTE(C1, T_8);
    WRITE_NOTE(C1, T_8);
    WRITE_NOTE(C1, T_4);
    WRITE_NOTE(B5, T_8);

    // 2^_ 1^_ 1^_ 1^ 5= 1^= |
    WRITE_NOTE(C2, T_8);
    WRITE_NOTE(C1, T_8);
    WRITE_NOTE(C1, T_8);
    WRITE_NOTE(C1, T_4);
    WRITE_NOTE(B5, T_16);
    WRITE_NOTE(C1, T_16);

    // 2^_ 1^_ 1^_ 1^ 5= 1^= |
    WRITE_NOTE(C2, T_8);
    WRITE_NOTE(C1, T_8);
    WRITE_NOTE(C1, T_8);
    WRITE_NOTE(C1, T_4);
    WRITE_NOTE(B5, T_16);
    WRITE_NOTE(C1, T_16);

    // 2^_. 3^= 2^_ 1^ 1^_ |
    WRITE_NOTE(C2, T_8D);
    WRITE_NOTE(C3, T_16);
    WRITE_NOTE(C2, T_8);
    WRITE_NOTE(C1, T_4);
    WRITE_NOTE(C1, T_8);

    // 7_ 6_ 6_ 6 5_ |
    WRITE_NOTE(B7, T_8);
    WRITE_NOTE(B6, T_8);
    WRITE_NOTE(B6, T_8);
    WRITE_NOTE(B6, T_4);
    WRITE_NOTE(B5, T_8);

    // 5 4_ 4_ 3_ 2_ |
    WRITE_NOTE(B5, T_4);
    WRITE_NOTE(B4, T_8);
    WRITE_NOTE(B4, T_8);
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B2, T_8);

    // 3 . 0 . |
    WRITE_NOTE(B3, T_4D);
    SLEEP_NOTE(T_4D);

    // 3_ 4_ 3_ 4_ 3_ 2= 1= |
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B4, T_8);
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B4, T_8);
    WRITE_NOTE(B3, T_8);
    WRITE_NOTE(B2, T_16);
    WRITE_NOTE(B1, T_16);

    // 1 . 1 . |
    WRITE_NOTE(B1, T_4D);
    WRITE_NOTE(B1, T_4D);

    // 3^_. 4^= 3^_ 2^ 1= 2= |
    WRITE_NOTE(C3, T_8D);
    WRITE_NOTE(C4, T_16);
    WRITE_NOTE(C3, T_8);
    WRITE_NOTE(C2, T_4);
    // WRITE_NOTE(B1, T_16);
    // WRITE_NOTE(B2, T_16);

    SLEEP_NOTE(1000);

    return MUSIC_INFO;
}
