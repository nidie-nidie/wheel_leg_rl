#include "music_you.h"

#include "bsp_buzzer.h"
#include "music.h"
#include "stm32h7xx_hal.h"

// clang-format off
#define note_A   220
#define note_3A  110  
#define note_5A  440  
#define note_sA  233  //233.082
#define note_B   247  //246.942
#define note_3B  123  //123.471
#define note_5B  494  //493.883
#define note_C   262  //261.626
#define note_5C  523  //523.251
#define note_sC  277  //277.183
#define note_D   294  //293.665
#define note_sD  311  //311.127
#define note_5D  587  //587.33
#define note_3sD 156  //155.563
#define note_E   330  //329.629
#define note_3E  165  //164.814
#define note_F   349  //349.228
#define note_3F  175  //174.614
#define note_sF  370  //369.994
#define note_3sF 185  //184.997
#define note_G   392  //391.995
#define note_sG  415  //415.305
#define note_3G  196  //195.998
#define note_5sG 831  //830.609
// clang-format on

#define NOTE_NUM 570
static Note Notes[NOTE_NUM]; // Array of notes
static MusicInfo_s MUSIC_INFO;

/*-------------------- User functions --------------------*/

MusicInfo_s MusicYouInit(void)
{
    MUSIC_INFO.notes = Notes;

    Notes[0].end = 0;

    float t = 40;

    WRITE_NOTE(note_5B, 200); // 前奏
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5B, 400);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5B, 400);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5D, 400);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5C, 200);
    WRITE_NOTE(note_5C, 200);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5B, 200);
    WRITE_NOTE(note_5C, 200);

    WRITE_NOTE(note_5B, 200);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5B, 400);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5B, 400);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5D, 400);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5C, 200);
    WRITE_NOTE(note_5C, 200);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5B, 200);
    WRITE_NOTE(note_5C, 200);

    WRITE_NOTE(note_5B, 200);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5B, 400);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5B, 400);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5D, 400);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5C, 200);
    WRITE_NOTE(note_5C, 200);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5B, 200);
    WRITE_NOTE(note_5C, 200);

    WRITE_NOTE(note_5B, 200);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5B, 400);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5B, 400);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5D, 400);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5C, 200);
    WRITE_NOTE(note_5C, 200);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_D, 400);

    WRITE_NOTE(note_E, 1200);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_G, 800);

    WRITE_NOTE(note_5C, 800);
    WRITE_NOTE(note_5B, 800);
    WRITE_NOTE(note_E, 800);
    WRITE_NOTE(note_D, 400);

    WRITE_NOTE(note_E, 1200);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_5C, 1600);

    WRITE_NOTE(note_5B, 400);
    WRITE_NOTE(note_5D, 800);
    WRITE_NOTE(note_E, 2000);

    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_5B, 800);

    WRITE_NOTE(note_5C, 800);
    WRITE_NOTE(note_5B, 800);
    WRITE_NOTE(note_E, 800);
    WRITE_NOTE(note_D, 400);

    WRITE_NOTE(note_E, 1200);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_5B, 800);

    WRITE_NOTE(note_5C, 800);
    WRITE_NOTE(note_5D, 2000);

    WRITE_NOTE(0, 800); // 我一直追寻着你
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_D, 200);
    WRITE_NOTE(note_E, 600);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_G, 400);

    WRITE_NOTE(0, 400); // 你好像不远也不近
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_D, 200);
    WRITE_NOTE(note_E, 600);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_G, 800);
    WRITE_NOTE(note_C, 1200);

    WRITE_NOTE(0, 400); // 却总保持着距离
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_D, 200);
    WRITE_NOTE(note_E, 600);
    //	WRITE_NOTE( note_D , 2 );
    WRITE_NOTE(note_D, 400);
    //	WRITE_NOTE( note_C , 1 );
    WRITE_NOTE(note_C, 400);
    WRITE_NOTE(note_D, 400);

    WRITE_NOTE(0, 400);
    WRITE_NOTE(note_E, 400); // 我一直幻想着你
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_D, 200);
    WRITE_NOTE(note_E, 600);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_G, 400);

    WRITE_NOTE(0, 400); // 在我身边在我怀里
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_D, 200);
    WRITE_NOTE(note_E, 600);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_5C, 800);
    WRITE_NOTE(note_C, 1200);

    WRITE_NOTE(0, 400); // 让我欢笑让我哭泣
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_D, 200);
    WRITE_NOTE(note_E, 600);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_B, 400);
    WRITE_NOTE(note_A, 200);
    WRITE_NOTE(note_3G, 1000);

    WRITE_NOTE(0, 200); // 你是我灵魂的旋律
    WRITE_NOTE(note_3G, 200);
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_3G, 200);
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_3G, 200);
    WRITE_NOTE(note_G, 800);
    WRITE_NOTE(note_E, 600);
    WRITE_NOTE(note_D, 200);
    WRITE_NOTE(note_C, 400);
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_C, 800);

    WRITE_NOTE(0, 200);
    WRITE_NOTE(note_C, 400); // 春日的细雨
    WRITE_NOTE(0, 0.05);
    WRITE_NOTE(note_C, 400);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_A, 1200);

    WRITE_NOTE(0, 400); // 墓碑的雏菊
    WRITE_NOTE(note_A, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_C, 400);
    WRITE_NOTE(note_D, 1200);
    WRITE_NOTE(0, 400);

    WRITE_NOTE(note_E, 800); // 我从来不会计算代价
    WRITE_NOTE(note_F, 800);
    WRITE_NOTE(note_G, 1200);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_G, 400);
    WRITE_NOTE(note_E, 200);
    WRITE_NOTE(note_G, 600);
    WRITE_NOTE(note_5B, 800);
    WRITE_NOTE(note_5C, 1200);

    WRITE_NOTE(note_C, 400); // 为了你可以纵身无底悬崖
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_G, 800);
    WRITE_NOTE(note_5A, 800);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5A, 600);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(0, 0.200);
    WRITE_NOTE(note_G, 600);
    WRITE_NOTE(0, 0.200);
    WRITE_NOTE(note_G, 800);
    WRITE_NOTE(note_D, 1600);

    WRITE_NOTE(note_E, 800); // 像条狗更像一个笑话
    WRITE_NOTE(note_F, 800);
    WRITE_NOTE(note_G, 1200);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_G, 400);
    WRITE_NOTE(note_E, 200);
    WRITE_NOTE(note_G, 600);
    WRITE_NOTE(note_5B, 800);
    WRITE_NOTE(note_5C, 1200);

    WRITE_NOTE(note_C, 400); // 也许我很傻但我不会怕
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_5B, 800);
    WRITE_NOTE(note_5A, 1200);
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_5A, 400);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5A, 600);
    WRITE_NOTE(note_5C, 800);
    WRITE_NOTE(note_5D, 1200);

    WRITE_NOTE(0, 400);
    WRITE_NOTE(note_G, 400); // 我愿意呀
    WRITE_NOTE(note_5C, 400);
    WRITE_NOTE(note_5B, 200);
    WRITE_NOTE(note_5C, 2400);

    WRITE_NOTE(0, 800);

    WRITE_NOTE(0, 400); // 人们都追寻着你
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_D, 200);
    WRITE_NOTE(note_E, 600);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_G, 400);

    WRITE_NOTE(0, 400); // 都曾把你当作唯一
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_D, 200);
    WRITE_NOTE(note_E, 600);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_G, 800);
    WRITE_NOTE(note_C, 1200);

    //	WRITE_NOTE( 0 , 4 );

    WRITE_NOTE(0, 400); // 最后却无能为力
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_D, 200);
    WRITE_NOTE(note_E, 600);
    //	WRITE_NOTE( note_D , 2 );
    WRITE_NOTE(note_D, 400);
    //	WRITE_NOTE( note_C , 1 );
    WRITE_NOTE(note_C, 400);
    WRITE_NOTE(note_D, 400);

    WRITE_NOTE(0, 400); // 人们都幻想着你
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_D, 200);
    WRITE_NOTE(note_E, 600);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_G, 400);

    WRITE_NOTE(0, 400); // 幻想你依偎他怀里
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_D, 200);
    WRITE_NOTE(note_E, 600);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_5C, 800);
    WRITE_NOTE(note_C, 1200);

    //	WRITE_NOTE( 0 , 4 );

    WRITE_NOTE(0, 400); // 一朝拥有一劳永逸
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_D, 200);
    WRITE_NOTE(note_E, 600);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_B, 400);
    WRITE_NOTE(note_A, 200);
    WRITE_NOTE(note_3G, 1000);

    WRITE_NOTE(0, 200); // 可是你不为谁守候
    WRITE_NOTE(note_3G, 200);
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_3G, 200);
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_3G, 200);
    WRITE_NOTE(note_G, 800);
    WRITE_NOTE(note_E, 600);
    WRITE_NOTE(note_D, 200);
    WRITE_NOTE(note_C, 400);
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_C, 800);

    WRITE_NOTE(0, 200);
    WRITE_NOTE(note_C, 400); // 不承诺永久
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_C, 400);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_A, 1200);

    WRITE_NOTE(0, 400); // 不轻易停留
    WRITE_NOTE(note_A, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_C, 400);
    WRITE_NOTE(note_D, 2000);

    WRITE_NOTE(0, 800);
    WRITE_NOTE(note_E, 800); // 我知道只有不断出发
    WRITE_NOTE(note_F, 800);
    WRITE_NOTE(note_G, 1200);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_G, 400);
    WRITE_NOTE(note_E, 200);
    WRITE_NOTE(note_G, 600);
    WRITE_NOTE(note_5B, 800);
    WRITE_NOTE(note_5C, 1200);

    WRITE_NOTE(note_C, 400); // 才能够紧随你纵情的步伐
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_G, 800);
    WRITE_NOTE(note_5A, 1000);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5A, 400);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_G, 600);
    WRITE_NOTE(0, 0.200);
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(0, 0.200);
    WRITE_NOTE(note_D, 1600);

    WRITE_NOTE(note_E, 800); // 就算是海角至天涯
    WRITE_NOTE(note_F, 800);
    WRITE_NOTE(note_G, 800);
    WRITE_NOTE(0, 0.1000);
    WRITE_NOTE(note_G, 400);
    WRITE_NOTE(note_E, 200);
    WRITE_NOTE(note_G, 600);
    WRITE_NOTE(note_5D, 800);
    WRITE_NOTE(note_5C, 1200);

    WRITE_NOTE(note_C, 400); // 青丝变白发
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_5C, 800);
    WRITE_NOTE(note_5A, 1000);

    WRITE_NOTE(note_G, 200); // 只等着你回答
    WRITE_NOTE(note_5A, 400);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5A, 600);
    WRITE_NOTE(note_5C, 800);
    WRITE_NOTE(note_5D, 1200);

    WRITE_NOTE(0, 400);
    WRITE_NOTE(note_D, 400); // 我愿意呀
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_D, 200);
    WRITE_NOTE(note_C, 2400);

    // 间奏略

    WRITE_NOTE(0, 800);
    WRITE_NOTE(note_E, 800); // 我从来不会计算代价
    WRITE_NOTE(note_F, 800);
    WRITE_NOTE(note_G, 1200);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_G, 400);
    WRITE_NOTE(note_E, 200);
    WRITE_NOTE(note_G, 600);
    WRITE_NOTE(note_5B, 800);
    WRITE_NOTE(note_5C, 1200);

    WRITE_NOTE(note_C, 400); // 为了你可以纵身无底悬崖
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_G, 800);
    WRITE_NOTE(note_5A, 1000);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5A, 400);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(0, 0.200);
    WRITE_NOTE(note_G, 600);
    WRITE_NOTE(0, 0.200);
    WRITE_NOTE(note_G, 800);
    WRITE_NOTE(note_D, 1600);

    WRITE_NOTE(note_E, 800); // 像条狗更像一个笑话
    WRITE_NOTE(note_F, 800);
    WRITE_NOTE(note_G, 1200);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_G, 400);
    WRITE_NOTE(note_E, 200);
    WRITE_NOTE(note_G, 600);
    WRITE_NOTE(note_5B, 800);
    WRITE_NOTE(note_5C, 1200);

    WRITE_NOTE(note_C, 400); // 也许我很傻但我不会怕
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_5C, 800);
    WRITE_NOTE(note_5A, 1200);
    WRITE_NOTE(0, t);
    WRITE_NOTE(note_5A, 400);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5A, 600);
    WRITE_NOTE(note_5C, 800);
    WRITE_NOTE(note_5D, 1200);

    WRITE_NOTE(note_E, 800); // 我知道只有不断出发
    WRITE_NOTE(note_F, 800);
    WRITE_NOTE(note_G, 1200);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_G, 400);
    WRITE_NOTE(note_E, 200);
    WRITE_NOTE(note_G, 600);
    WRITE_NOTE(note_5B, 800);
    WRITE_NOTE(note_5C, 1200);

    WRITE_NOTE(note_C, 400); // 才能够紧随你纵情的步伐
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_G, 800);
    WRITE_NOTE(note_5A, 1000);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5A, 400);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(0, 0.200);
    WRITE_NOTE(note_G, 600);
    WRITE_NOTE(0, 0.200);
    WRITE_NOTE(note_G, 800);
    //	WRITE_NOTE( note_D , 2 );
    //	WRITE_NOTE( 0 , 0.1 );
    WRITE_NOTE(note_D, 1600);

    WRITE_NOTE(note_E, 800); // 就算是海角至天涯
    WRITE_NOTE(note_F, 800);
    WRITE_NOTE(note_G, 800);
    WRITE_NOTE(0, 200);
    WRITE_NOTE(note_G, 400);
    WRITE_NOTE(note_E, 200);
    WRITE_NOTE(note_G, 600);
    WRITE_NOTE(note_5D, 800);
    WRITE_NOTE(note_5C, 1200);

    WRITE_NOTE(note_C, 400); // 青丝变白发
    WRITE_NOTE(note_D, 400);
    WRITE_NOTE(note_E, 400);
    WRITE_NOTE(note_5C, 800);
    WRITE_NOTE(note_5A, 1000);

    WRITE_NOTE(note_G, 200); // 只等着你回答
    WRITE_NOTE(note_5A, 400);
    WRITE_NOTE(note_G, 200);
    WRITE_NOTE(note_5A, 600);
    WRITE_NOTE(note_5C, 800);
    WRITE_NOTE(note_5D, 1200);

    WRITE_NOTE(0, 400);
    WRITE_NOTE(note_G, 400); // 我愿意呀
    WRITE_NOTE(note_5C, 400);
    WRITE_NOTE(note_5B, 200);
    WRITE_NOTE(note_5C, 2400);

    // 尾
    WRITE_NOTE(0, 800);
    WRITE_NOTE(note_E, 800);
    WRITE_NOTE(note_D, 800);
    WRITE_NOTE(note_C, 800);
    WRITE_NOTE(note_G, 800);
    WRITE_NOTE(note_C, 800);
    WRITE_NOTE(note_D, 800);
    WRITE_NOTE(note_E, 800);
    WRITE_NOTE(note_F, 800);
    WRITE_NOTE(note_G, 800);
    WRITE_NOTE(note_F, 800);

    WRITE_NOTE(note_E, 800);
    WRITE_NOTE(note_D, 800);
    WRITE_NOTE(note_C, 800);
    WRITE_NOTE(note_D, 800);
    WRITE_NOTE(note_E, 800);
    WRITE_NOTE(note_F, 800);
    WRITE_NOTE(note_E, 800);
    WRITE_NOTE(note_D, 800);
    WRITE_NOTE(note_C, 800);
    WRITE_NOTE(note_G, 800);

    WRITE_NOTE(note_E, 800);
    WRITE_NOTE(note_D, 800);
    WRITE_NOTE(note_E, 800);
    WRITE_NOTE(note_F, 800);
    WRITE_NOTE(note_G, 800);
    WRITE_NOTE(note_F, 800);
    WRITE_NOTE(note_E, 800);
    WRITE_NOTE(note_D, 800);

    WRITE_NOTE(note_E, 800);
    WRITE_NOTE(note_D, 800);
    WRITE_NOTE(note_E, 800);
    WRITE_NOTE(note_F, 800);

    WRITE_NOTE(note_E, 800);
    WRITE_NOTE(note_D, 800);
    WRITE_NOTE(note_C, 800);
    WRITE_NOTE(note_G, 800);
    WRITE_NOTE(note_C, 800);
    WRITE_NOTE(note_D, 800);
    WRITE_NOTE(note_E, 800);
    WRITE_NOTE(note_F, 800);
    WRITE_NOTE(note_G, 800);
    WRITE_NOTE(note_F, 800);
    WRITE_NOTE(note_E, 800);
    WRITE_NOTE(note_D, 800);

    return MUSIC_INFO;
}
