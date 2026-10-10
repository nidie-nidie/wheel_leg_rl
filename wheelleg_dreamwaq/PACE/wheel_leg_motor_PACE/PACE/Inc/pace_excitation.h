#ifndef PACE_EXCITATION_H
#define PACE_EXCITATION_H

#ifdef __cplusplus
extern "C" {
#endif

float pace_excitation_chirp(float time_s,
                            float duration_s,
                            float amplitude,
                            float frequency_start_hz,
                            float frequency_end_hz,
                            float phase_rad);
float pace_excitation_validation(float time_s,
                                 float duration_s,
                                 float amplitude,
                                 float frequency_start_hz,
                                 float frequency_end_hz,
                                 float phase_rad);

#ifdef __cplusplus
}
#endif

#endif
