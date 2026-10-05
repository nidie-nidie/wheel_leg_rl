#include "FreeRTOS.h"
#include "cmsis_os.h"
#include "main.h"
#include "pace_app.h"
#include "task.h"

static osThreadId pace_capture_task_handle;
static uint32_t pace_capture_task_buffer[768];
static osStaticThreadDef_t pace_capture_task_control;

static osThreadId pace_transport_task_handle;
static uint32_t pace_transport_task_buffer[256];
static osStaticThreadDef_t pace_transport_task_control;

static StaticTask_t pace_idle_task_control;
static StackType_t pace_idle_task_stack[configMINIMAL_STACK_SIZE];

void vApplicationGetIdleTaskMemory(StaticTask_t **task_control,
                                   StackType_t **task_stack,
                                   uint32_t *stack_size)
{
    *task_control = &pace_idle_task_control;
    *task_stack = pace_idle_task_stack;
    *stack_size = configMINIMAL_STACK_SIZE;
}

void MX_FREERTOS_Init(void)
{
    osThreadStaticDef(PACE_Capture, pace_app_capture_task, osPriorityAboveNormal,
                      0, 768, pace_capture_task_buffer, &pace_capture_task_control);
    pace_capture_task_handle = osThreadCreate(osThread(PACE_Capture), 0);

    osThreadStaticDef(PACE_Transport, pace_app_transport_task, osPriorityNormal,
                      0, 256, pace_transport_task_buffer, &pace_transport_task_control);
    pace_transport_task_handle = osThreadCreate(osThread(PACE_Transport), 0);

    if ((pace_capture_task_handle == 0) || (pace_transport_task_handle == 0))
    {
        Error_Handler();
    }
}
