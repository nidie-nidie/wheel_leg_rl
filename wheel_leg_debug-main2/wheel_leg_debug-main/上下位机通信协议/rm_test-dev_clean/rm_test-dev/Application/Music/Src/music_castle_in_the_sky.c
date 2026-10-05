#include "music_castle_in_the_sky.h"

#include "bsp_buzzer.h"
#include "music.h"
#include "stm32h7xx_hal.h"

// clang-format off
#define	qdo    262 
#define qre    294
#define qmi    330		//q前缀为低音，1后缀为高音，s前缀为半音阶 
#define qfa    349
#define qso    392
#define qla    440
#define qsi    494
#define do     523
#define re     578
#define mi     659
#define fa     698
#define so     784
#define la     880
#define si     988
#define do1    1046
#define re1    1175
#define mi1    1318
#define fa1    1480
#define so1    1568
#define la1    1760
#define si1    1976
#define sqdo   277
#define sqre   311
#define sqfa   370
#define sqso   415
#define sqla   466
#define sdo    554
#define sre    622
#define sfa    740
#define sso    831
#define sla    932
#define sdo1   1046
#define sre1   1245
#define sfa1   1480
#define sso1   1661
#define sla1   1865
// clang-format on

#define NOTE_NUM 370
static Note Notes[NOTE_NUM]; // Array of notes
static MusicInfo_s MUSIC_INFO;

/*-------------------- User functions --------------------*/

MusicInfo_s MusicCastleInTheSkyInit(void)
{
    MUSIC_INFO.notes = Notes;

    int pai = 400, ban = 200;
    int ting = 128;

    SLEEP_NOTE(1000);

    WRITE_NOTE(la, ban);
    WRITE_NOTE(si, ban);
    SLEEP_NOTE(ting);

    WRITE_NOTE(do1, pai + ban);
    WRITE_NOTE(si, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(do1, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(mi1, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(si, 3 * pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(mi, ban);
    WRITE_NOTE(mi, ban);

    WRITE_NOTE(la, ban + pai);
    WRITE_NOTE(so, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(la, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(do1, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(so, 2 * pai);
    SLEEP_NOTE(ting);
    SLEEP_NOTE(pai);
    WRITE_NOTE(mi, ban);
    SLEEP_NOTE(ting / 2);
    WRITE_NOTE(mi, ban);
    SLEEP_NOTE(ting / 2);

    WRITE_NOTE(fa, pai + ban);
    WRITE_NOTE(mi, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(fa, ban);
    WRITE_NOTE(do1, ban + pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(mi, 2 * pai);
    SLEEP_NOTE(ting);
    SLEEP_NOTE(ban);
    WRITE_NOTE(do1, ban);
    SLEEP_NOTE(ting / 2);
    WRITE_NOTE(do1, ban);
    SLEEP_NOTE(ting / 2);
    WRITE_NOTE(do1, ban);
    SLEEP_NOTE(ting / 2);

    WRITE_NOTE(si, ban + pai);
    WRITE_NOTE(sfa, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(sfa, pai);
    WRITE_NOTE(si, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(si, 2 * pai);
    SLEEP_NOTE(ting);
    SLEEP_NOTE(pai);
    WRITE_NOTE(la, ban);
    WRITE_NOTE(si, ban);
    SLEEP_NOTE(ting);

    WRITE_NOTE(do1, pai + ban);
    WRITE_NOTE(si, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(do1, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(mi1, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(si, 2 * pai);
    SLEEP_NOTE(ting);
    SLEEP_NOTE(pai);
    WRITE_NOTE(mi, ban);
    SLEEP_NOTE(20);
    WRITE_NOTE(mi, ban);
    SLEEP_NOTE(ting);

    WRITE_NOTE(la, pai + ban);
    WRITE_NOTE(so, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(la, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(do1, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(so, 3 * pai);
    SLEEP_NOTE(ting + ban);
    WRITE_NOTE(mi, ban);
    SLEEP_NOTE(ting / 2);

    WRITE_NOTE(fa, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(do1, ban);
    WRITE_NOTE(si, ban);
    SLEEP_NOTE(20);
    WRITE_NOTE(si, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(do1, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(re1, ban);
    SLEEP_NOTE(20);
    WRITE_NOTE(re1, ban);
    SLEEP_NOTE(20);
    WRITE_NOTE(mi1, ban);
    SLEEP_NOTE(ting / 2);
    WRITE_NOTE(do1, pai);
    SLEEP_NOTE(ting + pai);

    WRITE_NOTE(do1, pai);
    WRITE_NOTE(si, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(la, ban);
    SLEEP_NOTE(20);
    WRITE_NOTE(la, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(si, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(sso, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(sso, 2 * pai);
    SLEEP_NOTE(ting + pai);
    WRITE_NOTE(do1, ban);
    WRITE_NOTE(re1, ban);
    SLEEP_NOTE(ting);

    WRITE_NOTE(mi1, pai + ban);
    WRITE_NOTE(re1, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(mi1, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(fa1, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(re1, 2 * pai);
    SLEEP_NOTE(pai + ting);
    WRITE_NOTE(so, ban);
    SLEEP_NOTE(20);
    WRITE_NOTE(so, ban);
    SLEEP_NOTE(ting);

    WRITE_NOTE(do1, ban);
    WRITE_NOTE(si, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(do1, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(mi1, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(mi1, 2 * pai);
    SLEEP_NOTE(ting + 2 * pai);

    WRITE_NOTE(la, ban);
    WRITE_NOTE(si, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(do1, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(si, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(re1, ban);
    SLEEP_NOTE(20);
    WRITE_NOTE(re1, ban);
    SLEEP_NOTE(ting);

    WRITE_NOTE(do1, pai + ban);
    WRITE_NOTE(so, ban);
    SLEEP_NOTE(20);
    WRITE_NOTE(so, pai);
    SLEEP_NOTE(pai + ting);

    WRITE_NOTE(fa1, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(mi1, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(re1, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(do1, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(mi1, 4 * pai);

    WRITE_NOTE(mi1, pai * 2);
    SLEEP_NOTE(pai + ting);
    WRITE_NOTE(mi1, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(la1, 2 * pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(so1, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(so1, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(mi1, ban);
    SLEEP_NOTE(ting / 2);
    WRITE_NOTE(re1, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(do1, pai);
    SLEEP_NOTE(ting + ban);
    WRITE_NOTE(do1, ban);
    SLEEP_NOTE(ting);

    WRITE_NOTE(re1, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(do1, ban);
    WRITE_NOTE(re1, ban);
    SLEEP_NOTE(20);
    WRITE_NOTE(re1, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(so1, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(mi1, 2 * pai);
    SLEEP_NOTE(ting + pai);
    WRITE_NOTE(mi, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(la1, 2 * pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(so1, 2 * pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(mi1, ban);
    WRITE_NOTE(re1, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(do1, 2 * pai);
    SLEEP_NOTE(ting + ban);
    WRITE_NOTE(do1, ban);
    SLEEP_NOTE(ting);

    WRITE_NOTE(re1, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(do1, ban);
    WRITE_NOTE(re1, ban);
    SLEEP_NOTE(20);
    WRITE_NOTE(re1, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(si, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(la, 2 * pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(la, ban);
    WRITE_NOTE(si, ban);

    WRITE_NOTE(do1, pai + ban);
    WRITE_NOTE(si, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(do1, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(mi1, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(si, 3 * pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(mi, ban);
    WRITE_NOTE(mi, ban);

    WRITE_NOTE(la, ban + pai);
    WRITE_NOTE(so, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(la, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(do1, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(so, 2 * pai);
    SLEEP_NOTE(ting);
    SLEEP_NOTE(pai);
    WRITE_NOTE(mi, ban);
    SLEEP_NOTE(ting / 2);
    WRITE_NOTE(mi, ban);
    SLEEP_NOTE(ting / 2);

    WRITE_NOTE(fa, pai + ban);
    WRITE_NOTE(mi, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(fa, ban);
    WRITE_NOTE(do1, ban + pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(mi, 2 * pai);
    SLEEP_NOTE(ting);
    SLEEP_NOTE(ban);
    WRITE_NOTE(do1, ban);
    SLEEP_NOTE(ting / 2);
    WRITE_NOTE(do1, ban);
    SLEEP_NOTE(ting / 2);
    WRITE_NOTE(do1, ban);
    SLEEP_NOTE(ting / 2);

    WRITE_NOTE(si, ban + pai);
    WRITE_NOTE(sfa, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(sfa, pai);
    WRITE_NOTE(si, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(si, 2 * pai);
    SLEEP_NOTE(ting);
    SLEEP_NOTE(pai);
    WRITE_NOTE(la, ban);
    WRITE_NOTE(si, ban);
    SLEEP_NOTE(ting);

    WRITE_NOTE(do1, pai + ban);
    WRITE_NOTE(si, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(do1, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(mi1, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(si, 2 * pai);
    SLEEP_NOTE(ting);
    SLEEP_NOTE(pai);
    WRITE_NOTE(mi, ban);
    SLEEP_NOTE(20);
    WRITE_NOTE(mi, ban);
    SLEEP_NOTE(ting);

    WRITE_NOTE(la, pai + ban);
    WRITE_NOTE(so, ban);
    SLEEP_NOTE(ting);
    WRITE_NOTE(la, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(do1, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(so, 3 * pai);
    SLEEP_NOTE(ting + ban);
    WRITE_NOTE(mi, ban);
    SLEEP_NOTE(ting / 2);

    WRITE_NOTE(fa, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(do1, ban);
    WRITE_NOTE(si, ban);
    SLEEP_NOTE(20);
    WRITE_NOTE(si, pai);
    SLEEP_NOTE(ting);
    WRITE_NOTE(do1, pai);
    SLEEP_NOTE(ting);

    WRITE_NOTE(re1, ban);
    SLEEP_NOTE(20);
    WRITE_NOTE(re1, ban);
    SLEEP_NOTE(20);
    WRITE_NOTE(mi1, ban);
    SLEEP_NOTE(ting / 2);
    WRITE_NOTE(do1, pai);
    SLEEP_NOTE(ting + pai);

    WRITE_NOTE(la, 4 * pai);

    return MUSIC_INFO;
}
