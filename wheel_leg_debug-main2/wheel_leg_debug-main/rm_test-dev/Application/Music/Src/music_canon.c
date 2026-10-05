#include "music_canon.h"

#include "bsp_buzzer.h"
#include "music.h"
#include "stm32h7xx_hal.h"

// clang-format off
#define dolll   66
#define sdolll  70
#define relll   74
#define srelll  78
#define milll   83
#define falll   88
#define sfalll  93
#define solll   98
#define ssolll  104
#define lalll   110
#define slalll  117
#define silll   124

#define doll   131
#define sdoll  139
#define rell   147
#define srell  156
#define mill   165
#define fall   175
#define sfall  185
#define soll   196
#define ssoll  208
#define lall   220
#define slall  233
#define sill   247

#define dol   262
#define sdol  277
#define rel   294
#define srel  311
#define mil   330
#define fal   349
#define sfal  370
#define sol   392
#define ssol  415
#define lal   440
#define slal  466
#define sil   494

#define don   523
#define sdon  554
#define ren   578
#define sren  622
#define min   659
#define fan   698
#define sfan  740
#define son   784
#define sson  831
#define lan   880
#define slan  932
#define sin   988

#define doh   1046
#define sdoh  1046
#define reh   1175
#define sreh  1245
#define mih   1318
#define fah   1397
#define sfah  1480
#define soh   1568
#define ssoh  1661
#define lah   1760
#define slah  1865
#define sih   1976

#define dohh   2092
#define sdohh  2092
#define rehh   2350
#define srehh  2490
#define mihh   2636
#define fahh   2794
#define sfahh  2960
#define sohh   3136
#define ssohh  3322
#define lahh   3520
#define slahh  3730
#define sihh   3952

#define dohhh   4184
#define sdohhh  4184
#define rehhh   4700
#define srehhh  4980
#define mihhh   5272
#define fahhh   5588
#define sfahhh  5920
#define sohhh   6272
#define ssohhh  6644
#define lahhh   7040
#define slahhh  7460
#define sihhh   7904
// clang-format on

#define NOTE_NUM 240
static Note Notes[NOTE_NUM]; // Array of notes
static MusicInfo_s MUSIC_INFO;

/*-------------------- User functions --------------------*/

MusicInfo_s MusicCanonInit(void)
{
    MUSIC_INFO.notes = Notes;

    // SleepNote(800);

    // 第一行
    //  WRITE_NOTE(min*1.1225,800);
    //  SleepNote(800);
    //  WRITE_NOTE(ren*1.1225,800);
    //  SleepNote(800);

    // WRITE_NOTE(don*1.1225,800);
    // SleepNote(800);
    // WRITE_NOTE(sil*1.1225,800);
    // SleepNote(800);

    // WRITE_NOTE(lal*1.1225,800);
    // SleepNote(800);
    // WRITE_NOTE(sol*1.1225,800);
    // SleepNote(800);

    // WRITE_NOTE(lal*1.1225,800);
    // SleepNote(800);
    // WRITE_NOTE(sil*1.1225,800);
    // SleepNote(800);

    // 第二行
    //  WRITE_NOTE(mih*1.1225,800);
    //  SleepNote(800);
    //  WRITE_NOTE(reh*1.1225,800);
    //  WRITE_NOTE(fan*1.1225,800);

    // WRITE_NOTE(doh*1.1225,800);
    // SleepNote(800);
    // WRITE_NOTE(sin*1.1225,800);
    // WRITE_NOTE(son*1.1225,800);

    // WRITE_NOTE(lan*1.1225,800);
    // SleepNote(800);
    // WRITE_NOTE(son*1.1225,800);
    // WRITE_NOTE(min*1.1225,800);

    // WRITE_NOTE(fan*1.1225,800);
    // SleepNote(800);
    // WRITE_NOTE(son*1.1225,800);
    // WRITE_NOTE(sin*1.1225,800);

    // 第三行
    //  WRITE_NOTE(doh*1.1225,400);
    //  WRITE_NOTE(sin*1.1225,400);
    //  WRITE_NOTE(doh*1.1225,400);
    //  WRITE_NOTE(min*1.1225,400);
    //  WRITE_NOTE(son*1.1225,800);
    //  WRITE_NOTE(sin*1.1225,800);

    // WRITE_NOTE(doh*1.1225,800);
    // WRITE_NOTE(mih*1.1225,800);
    // WRITE_NOTE(soh*1.1225,400);
    // WRITE_NOTE(mih*1.1225,400);
    // WRITE_NOTE(soh*1.1225,400);
    // WRITE_NOTE(lah*1.1225,400);

    // WRITE_NOTE(fah*1.1225,400);
    // WRITE_NOTE(mih*1.1225,400);
    // WRITE_NOTE(reh*1.1225,400);
    // WRITE_NOTE(fah*1.1225,400);
    // WRITE_NOTE(mih*1.1225,400);
    // WRITE_NOTE(reh*1.1225,400);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(sin*1.1225,400);

    // WRITE_NOTE(lan*1.1225,400);
    // WRITE_NOTE(fan*1.1225,400);
    // WRITE_NOTE(doh*1.1225,800);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(fan*1.1225,400);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(sin*1.1225,400);

    // 第四行
    //  WRITE_NOTE(doh*1.1225,400);
    //  WRITE_NOTE(sin*1.1225,400);
    //  WRITE_NOTE(doh*1.1225,400);
    //  WRITE_NOTE(fan*1.1225,400);
    //  WRITE_NOTE(son*1.1225,400);
    //  WRITE_NOTE(sil*1.1225,400);
    //  WRITE_NOTE(sin*1.1225,400);
    //  WRITE_NOTE(fan*1.1225,400);

    // WRITE_NOTE(doh*1.1225,800);
    // WRITE_NOTE(mih*1.1225,400);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(soh*1.1225,400);
    // WRITE_NOTE(mih*1.1225,400);
    // WRITE_NOTE(soh*1.1225,400);
    // WRITE_NOTE(lah*1.1225,400);

    // WRITE_NOTE(fah*1.1225,400);
    // WRITE_NOTE(mih*1.1225,400);
    // WRITE_NOTE(reh*1.1225,400);
    // WRITE_NOTE(fah*1.1225,400);
    // WRITE_NOTE(mih*1.1225,400);
    // WRITE_NOTE(reh*1.1225,400);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(sin*1.1225,400);

    // WRITE_NOTE(lan*1.1225,400);
    // WRITE_NOTE(son*1.1225,400);
    // WRITE_NOTE(fan*1.1225,400);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(doh*1.1225,600);
    // WRITE_NOTE(ren*1.1225,200);
    // WRITE_NOTE(sin*1.1225,400);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);

    // 第五行
    //  WRITE_NOTE(mih*1.1225,600);
    //  WRITE_NOTE(son*1.1225,200);
    //  WRITE_NOTE(mih*1.1225,400);
    //  WRITE_NOTE(doh*1.1225,200);
    //  WRITE_NOTE(reh*1.1225,800);
    //  WRITE_NOTE(mih*1.1225,200);
    //  WRITE_NOTE(fah*1.1225,200);
    //  WRITE_NOTE(mih*1.1225,200);
    //  WRITE_NOTE(reh*1.1225,200);
    //  WRITE_NOTE(min*1.1225,200);

    // WRITE_NOTE(doh*1.1225,600);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,400);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(min*1.1225,400);
    // WRITE_NOTE(son*1.1225,400);
    // WRITE_NOTE(don*1.1225,200);

    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(don*1.1225,400);
    // WRITE_NOTE(sin*1.1225,400);
    // WRITE_NOTE(don*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(don*1.1225,200);
    // WRITE_NOTE(son*1.1225,400);
    // WRITE_NOTE(min*1.1225,400);
    // WRITE_NOTE(son*1.1225,400);
    // WRITE_NOTE(don*1.1225,200);
    // WRITE_NOTE(don*1.1225,200);

    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(don*1.1225,400);
    // WRITE_NOTE(don*1.1225,200);
    // WRITE_NOTE(fan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(ren*1.1225,400);
    // WRITE_NOTE(sin*1.1225,400);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);

    // 第六行
    //  WRITE_NOTE(mih*1.1225,800);
    //  WRITE_NOTE(mih*1.1225,200);
    //  WRITE_NOTE(reh*1.1225,200);
    //  WRITE_NOTE(doh*1.1225,200);
    //  WRITE_NOTE(reh*1.1225,400);
    //  WRITE_NOTE(son*1.1225,400);
    //  WRITE_NOTE(mih*1.1225,200);
    //  WRITE_NOTE(fah*1.1225,200);
    //  WRITE_NOTE(mih*1.1225,200);
    //  WRITE_NOTE(reh*1.1225,200);
    //  WRITE_NOTE(min*1.1225,200);

    // WRITE_NOTE(reh*1.1225,600);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,400);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(min*1.1225,400);
    // WRITE_NOTE(son*1.1225,400);
    // WRITE_NOTE(don*1.1225,200);

    // WRITE_NOTE(lan*1.1225,600);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,600);
    // WRITE_NOTE(don*1.1225,200);
    // WRITE_NOTE(don*1.1225,200);

    WRITE_NOTE(lan * 1.1225, 200);
    WRITE_NOTE(don * 1.1225, 400);
    WRITE_NOTE(don * 1.1225, 200);
    WRITE_NOTE(fan * 1.1225, 200);
    WRITE_NOTE(lan * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(reh * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 600);
    WRITE_NOTE(doh * 1.1225, 600);
    WRITE_NOTE(reh * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);

    // 第七行
    WRITE_NOTE(soh * 1.1225, 400);
    WRITE_NOTE(mih * 1.1225, 200);
    WRITE_NOTE(fah * 1.1225, 200);
    WRITE_NOTE(soh * 1.1225, 400);
    WRITE_NOTE(mih * 1.1225, 200);
    WRITE_NOTE(fah * 1.1225, 200);
    WRITE_NOTE(soh * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(lan * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(reh * 1.1225, 200);
    WRITE_NOTE(mih * 1.1225, 200);
    WRITE_NOTE(fah * 1.1225, 200);

    WRITE_NOTE(mih * 1.1225, 400);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(reh * 1.1225, 200);
    WRITE_NOTE(mih * 1.1225, 400);
    WRITE_NOTE(min * 1.1225, 200);
    WRITE_NOTE(fan * 1.1225, 200);
    WRITE_NOTE(son * 1.1225, 200);
    WRITE_NOTE(lan * 1.1225, 200);
    WRITE_NOTE(son * 1.1225, 200);
    WRITE_NOTE(fan * 1.1225, 200);
    WRITE_NOTE(son * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);

    WRITE_NOTE(lan * 1.1225, 400);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(lan * 1.1225, 400);
    WRITE_NOTE(son * 1.1225, 200);
    WRITE_NOTE(fan * 1.1225, 200);
    WRITE_NOTE(son * 1.1225, 200);
    WRITE_NOTE(fan * 1.1225, 200);
    WRITE_NOTE(min * 1.1225, 200);
    WRITE_NOTE(fan * 1.1225, 200);
    WRITE_NOTE(son * 1.1225, 200);
    WRITE_NOTE(lan * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);

    WRITE_NOTE(lan * 1.1225, 400);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 400);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(lan * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(reh * 1.1225, 200);
    WRITE_NOTE(mih * 1.1225, 200);
    WRITE_NOTE(fah * 1.1225, 200);
    WRITE_NOTE(soh * 1.1225, 200);
    // 第八行
    WRITE_NOTE(soh * 1.1225, 400);
    WRITE_NOTE(mih * 1.1225, 200);
    WRITE_NOTE(fah * 1.1225, 200);
    WRITE_NOTE(soh * 1.1225, 400);
    WRITE_NOTE(mih * 1.1225, 200);
    WRITE_NOTE(fah * 1.1225, 200);
    WRITE_NOTE(soh * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(lan * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(reh * 1.1225, 200);
    WRITE_NOTE(mih * 1.1225, 200);
    WRITE_NOTE(fah * 1.1225, 200);

    WRITE_NOTE(mih * 1.1225, 400);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(reh * 1.1225, 200);
    WRITE_NOTE(mih * 1.1225, 400);
    WRITE_NOTE(min * 1.1225, 200);
    WRITE_NOTE(fan * 1.1225, 200);
    WRITE_NOTE(son * 1.1225, 200);
    WRITE_NOTE(lan * 1.1225, 200);
    WRITE_NOTE(son * 1.1225, 200);
    WRITE_NOTE(fan * 1.1225, 200);
    WRITE_NOTE(son * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);

    WRITE_NOTE(lan * 1.1225, 400);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(lan * 1.1225, 400);
    WRITE_NOTE(son * 1.1225, 200);
    WRITE_NOTE(fan * 1.1225, 200);
    WRITE_NOTE(son * 1.1225, 200);
    WRITE_NOTE(fan * 1.1225, 200);
    WRITE_NOTE(min * 1.1225, 200);
    WRITE_NOTE(fan * 1.1225, 200);
    WRITE_NOTE(son * 1.1225, 200);
    WRITE_NOTE(lan * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);

    WRITE_NOTE(lan * 1.1225, 400);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 400);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(son * 1.1225, 200);
    WRITE_NOTE(lan * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(reh * 1.1225, 200);
    WRITE_NOTE(mih * 1.1225, 200);
    WRITE_NOTE(fah * 1.1225, 200);
    WRITE_NOTE(soh * 1.1225, 200);
    // 第九行
    WRITE_NOTE(soh * 1.1225, 400);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(reh * 1.1225, 200);
    WRITE_NOTE(mih * 1.1225, 400);
    WRITE_NOTE(reh * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(reh * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(reh * 1.1225, 200);
    WRITE_NOTE(mih * 1.1225, 200);
    WRITE_NOTE(reh * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);

    WRITE_NOTE(doh * 1.1225, 400);
    WRITE_NOTE(lan * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 400);
    WRITE_NOTE(don * 1.1225, 200);
    WRITE_NOTE(ren * 1.1225, 200);
    WRITE_NOTE(min * 1.1225, 200);
    WRITE_NOTE(fan * 1.1225, 200);
    WRITE_NOTE(min * 1.1225, 200);
    WRITE_NOTE(ren * 1.1225, 200);
    WRITE_NOTE(min * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);

    WRITE_NOTE(lan * 1.1225, 400);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(lan * 1.1225, 400);
    WRITE_NOTE(son * 1.1225, 200);
    WRITE_NOTE(fan * 1.1225, 200);
    WRITE_NOTE(son * 1.1225, 200);
    WRITE_NOTE(fan * 1.1225, 200);
    WRITE_NOTE(min * 1.1225, 200);
    WRITE_NOTE(fan * 1.1225, 200);
    WRITE_NOTE(son * 1.1225, 200);
    WRITE_NOTE(lan * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);

    WRITE_NOTE(lan * 1.1225, 400);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 400);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(lan * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(reh * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(lan * 1.1225, 200);
    WRITE_NOTE(sin * 1.1225, 200);
    // 第十行
    WRITE_NOTE(doh * 1.1225, 200);
    WRITE_NOTE(dol * 1.1225, 200);
    WRITE_NOTE(sol * 1.1225, 400);
    WRITE_NOTE(don * 1.1225, 400);
    WRITE_NOTE(min * 1.1225, 400);
    WRITE_NOTE(sol * 1.1225, 400);
    WRITE_NOTE(ren * 1.1225, 400);
    WRITE_NOTE(son * 1.1225, 400);
    WRITE_NOTE(sin * 1.1225, 400);

    WRITE_NOTE(lal * 1.1225, 400);
    WRITE_NOTE(min * 1.1225, 400);
    WRITE_NOTE(lan * 1.1225, 400);
    WRITE_NOTE(doh * 1.1225, 400);
    WRITE_NOTE(mil * 1.1225, 400);
    WRITE_NOTE(son * 1.1225, 400);
    WRITE_NOTE(sin * 1.1225, 400);
    WRITE_NOTE(mih * 1.1225, 400);

    WRITE_NOTE(fal * 1.1225, 400);
    WRITE_NOTE(don * 1.1225, 400);
    WRITE_NOTE(fan * 1.1225, 400);
    WRITE_NOTE(lan * 1.1225, 400);
    WRITE_NOTE(dol * 1.1225, 400);
    WRITE_NOTE(min * 1.1225, 400);
    WRITE_NOTE(son * 1.1225, 400);
    WRITE_NOTE(doh * 1.1225, 400);

    WRITE_NOTE(fal * 1.1225, 400);
    WRITE_NOTE(fan * 1.1225, 400);
    WRITE_NOTE(lan * 1.1225, 400);
    WRITE_NOTE(doh * 1.1225, 400);
    WRITE_NOTE(sol * 1.1225, 400);
    WRITE_NOTE(ren * 1.1225, 400);
    WRITE_NOTE(son * 1.1225, 400);
    WRITE_NOTE(sin * 1.1225, 400);

    // 第十一行
    //  	WRITE_NOTE(don*1.1225,400);
    //  WRITE_NOTE(mih*1.1225,200);
    //  WRITE_NOTE(fah*1.1225,200);
    //  WRITE_NOTE(soh*1.1225,400);
    //  WRITE_NOTE(mih*1.1225,400);
    //  	WRITE_NOTE(sol*1.1225,400);
    //  WRITE_NOTE(reh*1.1225,200);
    //  WRITE_NOTE(mih*1.1225,200);
    //  WRITE_NOTE(fah*1.1225,400);
    //  WRITE_NOTE(reh*1.1225,400);

    // 	WRITE_NOTE(lal*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(mih*1.1225,400);
    // WRITE_NOTE(doh*1.1225,400);
    // 	WRITE_NOTE(mil*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(sin*1.1225,400);

    // 	WRITE_NOTE(fal*1.1225,200);
    // WRITE_NOTE(fan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(lan*1.1225,400);
    // 	WRITE_NOTE(dol*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(son*1.1225,400);

    // 	WRITE_NOTE(fal*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(son*1.1225,400);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,400);
    // WRITE_NOTE(sin*1.1225,400);
    // //第十二行
    // 	WRITE_NOTE(don*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(soh*1.1225,400);
    // WRITE_NOTE(mih*1.1225,400);
    // 	WRITE_NOTE(sol*1.1225,400);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(fah*1.1225,400);
    // WRITE_NOTE(reh*1.1225,400);

    // 	WRITE_NOTE(lal*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(mih*1.1225,400);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,400);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(soh*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(mih*1.1225,400);
    // WRITE_NOTE(soh*1.1225,400);

    // WRITE_NOTE(lah*1.1225,400);
    // WRITE_NOTE(lah*1.1225,200);
    // WRITE_NOTE(soh*1.1225,200);
    // WRITE_NOTE(fah*1.1225,400);
    // WRITE_NOTE(lah*1.1225,400);
    // WRITE_NOTE(soh*1.1225,400);
    // WRITE_NOTE(soh*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(mih*1.1225,400);
    // WRITE_NOTE(soh*1.1225,400);

    // WRITE_NOTE(lah*1.1225,200);
    // WRITE_NOTE(soh*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(lah*1.1225,200);
    // WRITE_NOTE(soh*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(lah*1.1225,200);
    // WRITE_NOTE(soh*1.1225,200);
    // WRITE_NOTE(sih*1.1225,200);
    // WRITE_NOTE(lah*1.1225,200);
    // WRITE_NOTE(soh*1.1225,200);
    // WRITE_NOTE(sih*1.1225,200);
    // WRITE_NOTE(lah*1.1225,200);
    // WRITE_NOTE(soh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // //第十三行
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(son*1.1225,400);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,400);
    // WRITE_NOTE(son*1.1225,400);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);

    // WRITE_NOTE(doh*1.1225,600);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,400);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(min*1.1225,400);
    // WRITE_NOTE(son*1.1225,400);
    // WRITE_NOTE(don*1.1225,200);

    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(don*1.1225,400);
    // WRITE_NOTE(sin*1.1225,400);
    // WRITE_NOTE(don*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(don*1.1225,200);
    // WRITE_NOTE(son*1.1225,600);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);

    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(don*1.1225,400);
    // WRITE_NOTE(don*1.1225,200);
    // WRITE_NOTE(fan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(ren*1.1225,400);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(ren*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // //第十四行
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,400);
    // WRITE_NOTE(son*1.1225,400);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);

    // WRITE_NOTE(reh*1.1225,600);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(soh*1.1225,200);
    // WRITE_NOTE(sih*1.1225,200);
    // WRITE_NOTE(dohh*1.1225,200);

    // WRITE_NOTE(sih*1.1225,200);
    // WRITE_NOTE(lah*1.1225,200);
    // WRITE_NOTE(soh*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(soh*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);

    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(fan*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(fan*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // //第十五行
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(soh*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lah*1.1225,400);
    // WRITE_NOTE(soh*1.1225,400);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);

    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(min*1.1225,400);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(fan*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(ren*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(don*1.1225,200);

    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(don*1.1225,400);
    // WRITE_NOTE(don*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(don*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);

    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(don*1.1225,400);
    // WRITE_NOTE(don*1.1225,200);
    // WRITE_NOTE(fan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(ren*1.1225,400);
    // WRITE_NOTE(sin*1.1225,400);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // //第十六行
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(son*1.1225,400);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(soh*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lah*1.1225,400);
    // WRITE_NOTE(soh*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);

    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(min*1.1225,400);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(fan*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(ren*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(don*1.1225,200);

    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(don*1.1225,400);
    // WRITE_NOTE(don*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);

    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(ren*1.1225,200);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(ren*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // //第十七行
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(son*1.1225,400);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(soh*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lah*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(soh*1.1225,200);
    // WRITE_NOTE(fan*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);

    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(min*1.1225,400);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(fan*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(fan*1.1225,200);

    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(fan*1.1225,400);
    // WRITE_NOTE(fan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);

    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(ren*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(ren*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // //第十八行
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(soh*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lah*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(soh*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);

    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(min*1.1225,400);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(fah*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(min*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(fan*1.1225,200);

    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(fan*1.1225,400);
    // WRITE_NOTE(fan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,400);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);

    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(ren*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(ren*1.1225,200);
    // WRITE_NOTE(reh*1.1225,400);
    // //第十九行
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);

    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);

    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);

    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // //第二十行
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(mih*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);
    // WRITE_NOTE(reh*1.1225,200);

    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(doh*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);

    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);
    // WRITE_NOTE(son*1.1225,200);

    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(lan*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,200);
    // WRITE_NOTE(sin*1.1225,133);
    // WRITE_NOTE(sin*1.1225,133);
    // WRITE_NOTE(sin*1.1225,134);
    // //结尾
    // WRITE_NOTE(sin*1.1225,400);
    // WRITE_NOTE(doh*1.1225,800);
    // SleepNote(2000);

    return MUSIC_INFO;
}
