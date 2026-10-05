#include "BMI088Middleware.h"
#include "main.h"

SPI_HandleTypeDef *BMI088_SPI;

/**
 * @brief RESET the BMI088_ACCEL_NS
 * @note GPIO_x: GPIOC
 * @note GPIO_PIN_x: GPIO_PIN_0
 */
void BMI088_ACCEL_NS_L(void)
{
    HAL_GPIO_WritePin(ACCEL_CS_GPIO_Port, ACCEL_CS_Pin, GPIO_PIN_RESET);
}
/**
 * @brief SET the BMI088_ACCEL_NS
 * @note GPIO_x: GPIOC
 * @note GPIO_PIN_x: GPIO_PIN_0
 */
void BMI088_ACCEL_NS_H(void)
{
    HAL_GPIO_WritePin(ACCEL_CS_GPIO_Port, ACCEL_CS_Pin, GPIO_PIN_SET);
}
/**
 * @brief RESET the BMI088_GYRO_NS
 * @note GPIO_x: GPIOC
 * @note GPIO_PIN_x: GPIO_PIN_3
 */
void BMI088_GYRO_NS_L(void)
{
    HAL_GPIO_WritePin(GYRO_CS_GPIO_Port, GYRO_CS_Pin, GPIO_PIN_RESET);
}
/**
 * @brief RESET the BMI088_GYRO_NS
 * @note GPIO_x: GPIOC
 * @note GPIO_PIN_x: GPIO_PIN_3
 */
void BMI088_GYRO_NS_H(void)
{
    HAL_GPIO_WritePin(GYRO_CS_GPIO_Port, GYRO_CS_Pin, GPIO_PIN_SET);
}

uint8_t BMI088_read_write_byte(uint8_t txdata)
{
    uint8_t rx_data;
    HAL_SPI_TransmitReceive(BMI088_SPI, &txdata, &rx_data, 1, 1000);
    return rx_data;
}
