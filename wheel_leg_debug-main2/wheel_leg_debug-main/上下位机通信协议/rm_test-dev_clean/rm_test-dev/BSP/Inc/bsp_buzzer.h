#ifndef BSP_BUZZER_H
#define BSP_BUZZER_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "struct_typedef.h"
    extern void buzzer_on(uint16_t psc, uint16_t pwm);
    extern void buzzer_off(void);
    extern void buzzer_note(uint16_t note, float volume);

#endif
