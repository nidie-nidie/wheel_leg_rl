#include "pace_excitation.h"

#include <math.h>

#define PACE_TWO_PI_F 6.28318530717958647692f
#define PACE_EXCITATION_RAMP_S 0.5f

static float pace_excitation_envelope(float time_s, float duration_s)
{
    float scale = 1.0f;
    float x;
    if ((time_s <= 0.0f) || (time_s >= duration_s))
    {
        return 0.0f;
    }
    if (time_s < PACE_EXCITATION_RAMP_S)
    {
        scale = time_s / PACE_EXCITATION_RAMP_S;
    }
    if ((duration_s - time_s) < PACE_EXCITATION_RAMP_S)
    {
        const float end_scale = (duration_s - time_s) / PACE_EXCITATION_RAMP_S;
        if (end_scale < scale)
        {
            scale = end_scale;
        }
    }
    x = (scale < 0.0f) ? 0.0f : ((scale > 1.0f) ? 1.0f : scale);
    return x * x * (3.0f - (2.0f * x));
}

float pace_excitation_chirp(float time_s,
                            float duration_s,
                            float amplitude,
                            float frequency_start_hz,
                            float frequency_end_hz,
                            float phase_rad)
{
    float sweep_rate;
    float phase;
    if (duration_s <= 0.0f)
    {
        return 0.0f;
    }
    sweep_rate = (frequency_end_hz - frequency_start_hz) / duration_s;
    phase = PACE_TWO_PI_F * ((frequency_start_hz * time_s) +
                            (0.5f * sweep_rate * time_s * time_s)) + phase_rad;
    return amplitude * pace_excitation_envelope(time_s, duration_s) * sinf(phase);
}

float pace_excitation_validation(float time_s,
                                 float duration_s,
                                 float amplitude,
                                 float frequency_start_hz,
                                 float frequency_end_hz,
                                 float phase_rad)
{
    const float primary = pace_excitation_chirp(time_s, duration_s, amplitude,
                                                frequency_end_hz, frequency_start_hz,
                                                phase_rad);
    const float secondary = 0.35f * amplitude *
                            pace_excitation_envelope(time_s, duration_s) *
                            sinf(PACE_TWO_PI_F * 1.37f * time_s + phase_rad * 0.5f);
    return primary + secondary;
}
