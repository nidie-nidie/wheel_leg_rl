#ifndef SIM_PORT_BMI088DRIVER_H
#define SIM_PORT_BMI088DRIVER_H

typedef struct
{
    float Accel[3];
    float Gyro[3];
} BMI088_t;

extern BMI088_t BMI088;

#endif

