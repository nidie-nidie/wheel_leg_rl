#include "music_hao_yun_lai.h"

#include "bsp_buzzer.h"
#include "music.h"
#include "stm32h7xx_hal.h"
#include "music_typedef.h"

#define NOTE_NUM 15
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
void MusicMeowPlay(void)
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

        buzzer_note(Notes[play_id].note, 0.1);
    }
}

void MusicMeowInit(void)
{
    SleepNote(500);

    WriteNote(1200, 80);
    WriteNote(1000, 100);
    WriteNote(800, 120);

    last_note_id = write_id - 1;
    write_id = 1;
}
/*------------------------------ End of File ------------------------------*/
