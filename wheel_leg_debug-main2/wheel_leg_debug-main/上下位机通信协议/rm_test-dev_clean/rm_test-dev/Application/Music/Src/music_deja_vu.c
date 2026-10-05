#include "music_deja_vu.h"

#include "bsp_buzzer.h"
#include "music.h"
#include "stm32h7xx_hal.h"
#include "music_typedef.h"

// https://www.douyin.com/user/self?from_tab_name=main&modal_id=7009575449315642657&showSubTab=video&showTab=favorite_collection

// clang-format off
#define Z__no 0
#define Bb_do 440
#define Bb_re 494
#define Bb_mi 554
#define Bb_fa 587
#define Bb_so 659
#define Bb_la 740
#define Bb_si 831
// clang-format on

#define ONE_FOURTH 250
#define ONE_FOURTH_HALF 125
#define ONE_FOURTH_HALF_HALF 63

#define NOTE_NUM 200
static Note Notes[NOTE_NUM]; // Array of notes

static uint32_t last_note_id = 0; // Index of the last note
static uint32_t write_id = 1;     // Index of the note to be written
static uint32_t play_id = 1;      // Index of the note to be played

static uint32_t start_time = 0; // Start time of the music
static uint32_t now = 0;

/*-------------------- Private functions --------------------*/
static void WriteNote(int note, float Long)
{
    Notes[write_id].note = note;
    Notes[write_id].Long = Long;
    Notes[write_id].end = Notes[write_id - 1].end + Long;
    write_id++;
}

static void SleepNote(float Long) { WriteNote(0, Long); }

/*-------------------- User functions --------------------*/
void MusicDejaVuPlay(void)
{
    now = HAL_GetTick();
    if (now - start_time >= Notes[play_id].end)
    {
        play_id++;
        if (play_id > last_note_id)
        {
            play_id = 1;
            start_time = now;
        }

        buzzer_note(Notes[play_id].note, 0.05);
    }
}

void MusicDejaVuInit(void)
{
    SleepNote(1500);

    // do re mi fa so la si
    // 1  2  3  4  5  6  7

    // 2 0_ 4_ 0_ 4_ 0_ 5_
    WriteNote(Bb_re, ONE_FOURTH);      // 2
    WriteNote(Z__no, ONE_FOURTH_HALF); // 0_
    WriteNote(Bb_fa, ONE_FOURTH_HALF); // 4_
    WriteNote(Z__no, ONE_FOURTH_HALF); // 0_
    WriteNote(Bb_fa, ONE_FOURTH_HALF); // 4_
    WriteNote(Z__no, ONE_FOURTH_HALF); // 0_
    WriteNote(Bb_so, ONE_FOURTH_HALF); // 5_

    // 5 0_ 1`_ 1`_ 7_ 7
    WriteNote(Bb_so, ONE_FOURTH);          // 5
    WriteNote(Z__no, ONE_FOURTH_HALF);     // 0_
    WriteNote(Bb_do * 2, ONE_FOURTH_HALF); // 1`_
    WriteNote(Bb_do * 2, ONE_FOURTH_HALF); // 1`_
    WriteNote(Bb_si, ONE_FOURTH_HALF);     // 7_
    WriteNote(Bb_si, ONE_FOURTH);          // 7

    // 2 6__ 1`__ 6_ 0_ 6_ 0_ 6_
    WriteNote(Bb_re, ONE_FOURTH);               // 2
    WriteNote(Bb_la, ONE_FOURTH_HALF_HALF);     // 6__
    WriteNote(Bb_do * 2, ONE_FOURTH_HALF_HALF); // 1`__
    WriteNote(Bb_la, ONE_FOURTH_HALF);          // 6_
    WriteNote(Z__no, ONE_FOURTH_HALF);          // 0_
    WriteNote(Bb_la, ONE_FOURTH_HALF);          // 6_
    WriteNote(Z__no, ONE_FOURTH_HALF);          // 0_
    WriteNote(Bb_la, ONE_FOURTH_HALF);          // 6_

    // 0_ 5 1`__ 2`__ 1`_ 1`_ 2`
    WriteNote(Z__no, ONE_FOURTH_HALF);          // 0_
    WriteNote(Bb_so, ONE_FOURTH);               // 5
    WriteNote(Bb_do * 2, ONE_FOURTH_HALF_HALF); // 1`__
    WriteNote(Bb_re * 2, ONE_FOURTH_HALF_HALF); // 2`__
    WriteNote(Bb_do * 2, ONE_FOURTH_HALF);      // 1`_
    WriteNote(Bb_do * 2, ONE_FOURTH_HALF);      // 1`_
    WriteNote(Bb_re * 2, ONE_FOURTH);           // 2`

    // 2 0_ 4_ 0_ 4_ 0_ 5_
    WriteNote(Bb_re, ONE_FOURTH);      // 2
    WriteNote(Z__no, ONE_FOURTH_HALF); // 0_
    WriteNote(Bb_fa, ONE_FOURTH_HALF); // 4_
    WriteNote(Z__no, ONE_FOURTH_HALF); // 0_
    WriteNote(Bb_fa, ONE_FOURTH_HALF); // 4_
    WriteNote(Z__no, ONE_FOURTH_HALF); // 0_
    WriteNote(Bb_so, ONE_FOURTH_HALF); // 5_

    // 5 0_ 1`_ 1`_ 7_ 7
    WriteNote(Bb_so, ONE_FOURTH);          // 5
    WriteNote(Z__no, ONE_FOURTH_HALF);     // 0_
    WriteNote(Bb_do * 2, ONE_FOURTH_HALF); // 1`_
    WriteNote(Bb_do * 2, ONE_FOURTH_HALF); // 1`_
    WriteNote(Bb_si, ONE_FOURTH_HALF);     // 7_
    WriteNote(Bb_si, ONE_FOURTH);          // 7

    // 4_ 0_ 6__ 1`__ 6_ 0_ 6_ 1`
    WriteNote(Bb_fa, ONE_FOURTH_HALF);          // 4_
    WriteNote(Z__no, ONE_FOURTH_HALF);          // 0_
    WriteNote(Bb_la, ONE_FOURTH_HALF_HALF);     // 6__
    WriteNote(Bb_do * 2, ONE_FOURTH_HALF_HALF); // 1`__
    WriteNote(Bb_la, ONE_FOURTH_HALF);          // 6_
    WriteNote(Z__no, ONE_FOURTH_HALF);          // 0_
    WriteNote(Bb_la, ONE_FOURTH_HALF);          // 6_
    WriteNote(Bb_do * 2, ONE_FOURTH);           // 1`

    // 另一个？

    // // 2.._6.._  2._6.._  1._4._  1._5.._
    // WriteNote(Bb_re / 4, ONE_FOURTH_HALF);  // 2.._
    // WriteNote(Bb_la / 4, ONE_FOURTH_HALF);  // 6.._
    // WriteNote(Bb_re / 2, ONE_FOURTH_HALF);  // 2._
    // WriteNote(Bb_la / 4, ONE_FOURTH_HALF);  // 6.._
    // WriteNote(Bb_do / 2, ONE_FOURTH_HALF);  // 1._
    // WriteNote(Bb_fa / 2, ONE_FOURTH_HALF);  // 4._
    // WriteNote(Bb_do / 2, ONE_FOURTH_HALF);  // 1._
    // WriteNote(Bb_so / 4, ONE_FOURTH_HALF);  // 5.._

    // //2._5._  5.._2._  5._5.._  2._5._
    // WriteNote(Bb_re / 2, ONE_FOURTH_HALF);  // 2._
    // WriteNote(Bb_so / 2, ONE_FOURTH_HALF);  // 5._
    // WriteNote(Bb_so / 4, ONE_FOURTH_HALF);  // 5.._
    // WriteNote(Bb_re / 2, ONE_FOURTH_HALF);  // 2._
    // WriteNote(Bb_so / 2, ONE_FOURTH_HALF);  // 5._
    // WriteNote(Bb_so / 4, ONE_FOURTH_HALF);  // 5.._
    // WriteNote(Bb_re / 2, ONE_FOURTH_HALF);  // 2._
    // WriteNote(Bb_so / 2, ONE_FOURTH_HALF);  // 5._

    // // 2.._6.._  2._4.._ 1._4._  1._5.._
    // WriteNote(Bb_re / 4, ONE_FOURTH_HALF);  // 2.._
    // WriteNote(Bb_la / 4, ONE_FOURTH_HALF);  // 6.._
    // WriteNote(Bb_re / 2, ONE_FOURTH_HALF);  // 2._
    // WriteNote(Bb_fa / 4, ONE_FOURTH_HALF);  // 4.._
    // WriteNote(Bb_do / 2, ONE_FOURTH_HALF);  // 1._
    // WriteNote(Bb_fa / 2, ONE_FOURTH_HALF);  // 4._
    // WriteNote(Bb_do / 2, ONE_FOURTH_HALF);  // 1._
    // WriteNote(Bb_so / 4, ONE_FOURTH_HALF);  // 5.._

    // // 2._5._  5.._2._  5._5.._  2._5._
    // WriteNote(Bb_re / 2, ONE_FOURTH_HALF);  // 2._
    // WriteNote(Bb_so / 2, ONE_FOURTH_HALF);  // 5._
    // WriteNote(Bb_so / 4, ONE_FOURTH_HALF);  // 5.._
    // WriteNote(Bb_re / 2, ONE_FOURTH_HALF);  // 2._
    // WriteNote(Bb_so / 2, ONE_FOURTH_HALF);  // 5._
    // WriteNote(Bb_so / 4, ONE_FOURTH_HALF);  // 5.._
    // WriteNote(Bb_re / 2, ONE_FOURTH_HALF);  // 2._
    // WriteNote(Bb_so / 2, ONE_FOURTH_HALF);  // 5._

    // // 2.._6.._  2._4.._ 1._4._  1._5.._
    // WriteNote(Bb_re / 4, ONE_FOURTH_HALF);  // 2.._
    // WriteNote(Bb_la / 4, ONE_FOURTH_HALF);  // 6.._
    // WriteNote(Bb_re / 2, ONE_FOURTH_HALF);  // 2._
    // WriteNote(Bb_fa / 4, ONE_FOURTH_HALF);  // 4.._
    // WriteNote(Bb_do / 2, ONE_FOURTH_HALF);  // 1._
    // WriteNote(Bb_fa / 2, ONE_FOURTH_HALF);  // 4._
    // WriteNote(Bb_do / 2, ONE_FOURTH_HALF);  // 1._
    // WriteNote(Bb_so / 4, ONE_FOURTH_HALF);  // 5.._

    // // 2._5._  5.._2._  5._5.._  2._5._
    // WriteNote(Bb_re / 2, ONE_FOURTH_HALF);  // 2._
    // WriteNote(Bb_so / 2, ONE_FOURTH_HALF);  // 5._
    // WriteNote(Bb_so / 4, ONE_FOURTH_HALF);  // 5.._
    // WriteNote(Bb_re / 2, ONE_FOURTH_HALF);  // 2._
    // WriteNote(Bb_so / 2, ONE_FOURTH_HALF);  // 5._
    // WriteNote(Bb_so / 4, ONE_FOURTH_HALF);  // 5.._
    // WriteNote(Bb_re / 2, ONE_FOURTH_HALF);  // 2._
    // WriteNote(Bb_so / 2, ONE_FOURTH_HALF);  // 5._

    // // 2.._6.._  2._4.._ 1._4._  4.._1._
    // WriteNote(Bb_re / 4, ONE_FOURTH_HALF);  // 2.._
    // WriteNote(Bb_la / 4, ONE_FOURTH_HALF);  // 6.._
    // WriteNote(Bb_re / 2, ONE_FOURTH_HALF);  // 2._
    // WriteNote(Bb_fa / 4, ONE_FOURTH_HALF);  // 4.._
    // WriteNote(Bb_do / 2, ONE_FOURTH_HALF);  // 1._
    // WriteNote(Bb_fa / 2, ONE_FOURTH_HALF);  // 4._
    // WriteNote(Bb_fa / 4, ONE_FOURTH_HALF);  // 4.._
    // WriteNote(Bb_do / 2, ONE_FOURTH_HALF);  // 1._

    // 另一个？

    // 2.._6.._  2._4.._ 1._4._  4.._1._

    last_note_id = write_id - 1;
    write_id = 1;
}
/*------------------------------ End of File ------------------------------*/
