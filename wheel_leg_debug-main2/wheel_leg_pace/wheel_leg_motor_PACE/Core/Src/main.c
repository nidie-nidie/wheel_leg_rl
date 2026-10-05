#include "main.h"

#include "cmsis_os.h"
#include "dma.h"
#include "fdcan.h"
#include "gpio.h"
#include "pace_app.h"
#include "tim.h"
#include "usart.h"

void SystemClock_Config(void);
void PeriphCommonClock_Config(void);
void MX_FREERTOS_Init(void);

int main(void)
{
    HAL_Init();
    SystemClock_Config();
    PeriphCommonClock_Config();

    MX_GPIO_Init();
    MX_DMA_Init();
    MX_FDCAN3_Init();
    MX_UART7_Init();
    MX_TIM5_Init();
    MX_TIM6_Init();

    if (!pace_app_init())
    {
        Error_Handler();
    }

    MX_FREERTOS_Init();
    osKernelStart();

    while (1)
    {
    }
}

void SystemClock_Config(void)
{
    RCC_OscInitTypeDef oscillator = {0};
    RCC_ClkInitTypeDef clocks = {0};

    HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
    while (!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY))
    {
    }

    oscillator.OscillatorType = RCC_OSCILLATORTYPE_HSI48 |
                                RCC_OSCILLATORTYPE_HSI |
                                RCC_OSCILLATORTYPE_HSE;
    oscillator.HSEState = RCC_HSE_ON;
    oscillator.HSIState = RCC_HSI_DIV1;
    oscillator.HSICalibrationValue = 64U;
    oscillator.HSI48State = RCC_HSI48_ON;
    oscillator.PLL.PLLState = RCC_PLL_ON;
    oscillator.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    oscillator.PLL.PLLM = 6U;
    oscillator.PLL.PLLN = 100U;
    oscillator.PLL.PLLP = 1U;
    oscillator.PLL.PLLQ = 4U;
    oscillator.PLL.PLLR = 2U;
    oscillator.PLL.PLLRGE = RCC_PLL1VCIRANGE_2;
    oscillator.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
    oscillator.PLL.PLLFRACN = 0U;
    if (HAL_RCC_OscConfig(&oscillator) != HAL_OK)
    {
        Error_Handler();
    }

    clocks.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
                       RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2 |
                       RCC_CLOCKTYPE_D3PCLK1 | RCC_CLOCKTYPE_D1PCLK1;
    clocks.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    clocks.SYSCLKDivider = RCC_SYSCLK_DIV1;
    clocks.AHBCLKDivider = RCC_HCLK_DIV4;
    clocks.APB3CLKDivider = RCC_APB3_DIV2;
    clocks.APB1CLKDivider = RCC_APB1_DIV2;
    clocks.APB2CLKDivider = RCC_APB2_DIV2;
    clocks.APB4CLKDivider = RCC_APB4_DIV2;
    if (HAL_RCC_ClockConfig(&clocks, FLASH_LATENCY_2) != HAL_OK)
    {
        Error_Handler();
    }
}

void PeriphCommonClock_Config(void)
{
    RCC_PeriphCLKInitTypeDef peripheral_clock = {0};
    peripheral_clock.PeriphClockSelection = RCC_PERIPHCLK_FDCAN;
    peripheral_clock.PLL2.PLL2M = 24U;
    peripheral_clock.PLL2.PLL2N = 200U;
    peripheral_clock.PLL2.PLL2P = 2U;
    peripheral_clock.PLL2.PLL2Q = 2U;
    peripheral_clock.PLL2.PLL2R = 2U;
    peripheral_clock.PLL2.PLL2RGE = RCC_PLL2VCIRANGE_0;
    peripheral_clock.PLL2.PLL2VCOSEL = RCC_PLL2VCOMEDIUM;
    peripheral_clock.PLL2.PLL2FRACN = 0U;
    peripheral_clock.FdcanClockSelection = RCC_FDCANCLKSOURCE_PLL2;
    if (HAL_RCCEx_PeriphCLKConfig(&peripheral_clock) != HAL_OK)
    {
        Error_Handler();
    }
}

void Error_Handler(void)
{
    __disable_irq();
    while (1)
    {
    }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
    (void)file;
    (void)line;
}
#endif
