#include "bsp_fdcan.h"

void bsp_can_init(void)
{
	can_filter_init();
	HAL_FDCAN_Start(&hfdcan1);
	HAL_FDCAN_Start(&hfdcan2);
	HAL_FDCAN_Start(&hfdcan3);
	HAL_FDCAN_ActivateNotification(&hfdcan1, FDCAN_IT_RX_FIFO0_NEW_MESSAGE, 0);
	HAL_FDCAN_ActivateNotification(&hfdcan2, FDCAN_IT_RX_FIFO0_NEW_MESSAGE, 0);
	HAL_FDCAN_ActivateNotification(&hfdcan3, FDCAN_IT_RX_FIFO0_NEW_MESSAGE, 0);
}

void can_filter_init(void)
{
	FDCAN_FilterTypeDef fdcan_filter;

	fdcan_filter.IdType = FDCAN_STANDARD_ID;
	fdcan_filter.FilterIndex = 0;
	fdcan_filter.FilterType = FDCAN_FILTER_MASK;
	fdcan_filter.FilterConfig = FDCAN_FILTER_TO_RXFIFO0;
	fdcan_filter.FilterID1 = 0x00;
	fdcan_filter.FilterID2 = 0x00;

	HAL_FDCAN_ConfigFilter(&hfdcan1, &fdcan_filter);
	HAL_FDCAN_ConfigGlobalFilter(&hfdcan1, FDCAN_REJECT, FDCAN_REJECT, FDCAN_REJECT_REMOTE, FDCAN_REJECT_REMOTE);
	HAL_FDCAN_ConfigFifoWatermark(&hfdcan1, FDCAN_CFG_RX_FIFO0, 1);
	//	HAL_FDCAN_ConfigFifoWatermark(&hfdcan1, FDCAN_CFG_RX_FIFO1, 1);
	//	HAL_FDCAN_ActivateNotification(&hfdcan1, FDCAN_IT_TX_COMPLETE, FDCAN_TX_BUFFER0);
}

uint8_t fdcanx_send_data(hcan_t *hfdcan, uint16_t id, uint8_t *data, uint32_t len)
{
	FDCAN_TxHeaderTypeDef pTxHeader;
	pTxHeader.Identifier = id;
	pTxHeader.IdType = FDCAN_STANDARD_ID;
	pTxHeader.TxFrameType = FDCAN_DATA_FRAME;

	if (len <= 8)
		pTxHeader.DataLength = len;
	if (len == 12)
		pTxHeader.DataLength = FDCAN_DLC_BYTES_12;
	if (len == 16)
		pTxHeader.DataLength = FDCAN_DLC_BYTES_16;
	if (len == 20)
		pTxHeader.DataLength = FDCAN_DLC_BYTES_20;
	if (len == 24)
		pTxHeader.DataLength = FDCAN_DLC_BYTES_24;
	if (len == 32)
		pTxHeader.DataLength = FDCAN_DLC_BYTES_32;
	if (len == 48)
		pTxHeader.DataLength = FDCAN_DLC_BYTES_48;
	if (len == 64)
		pTxHeader.DataLength = FDCAN_DLC_BYTES_64;

	pTxHeader.ErrorStateIndicator = FDCAN_ESI_ACTIVE;
	pTxHeader.BitRateSwitch = FDCAN_BRS_ON;
	pTxHeader.FDFormat = FDCAN_FD_CAN;
	pTxHeader.TxEventFifoControl = FDCAN_NO_TX_EVENTS;
	pTxHeader.MessageMarker = 0;

	if (HAL_FDCAN_AddMessageToTxFifoQ(hfdcan, &pTxHeader, data) != HAL_OK)
		return 1;
	return 0;
}

uint8_t fdcanx_receive(hcan_t *hfdcan, uint16_t *rec_id, uint8_t *buf)
{
	FDCAN_RxHeaderTypeDef pRxHeader;
	uint8_t len;

	if (HAL_FDCAN_GetRxMessage(hfdcan, FDCAN_RX_FIFO0, &pRxHeader, buf) == HAL_OK)
	{
		*rec_id = pRxHeader.Identifier;
		if (pRxHeader.DataLength <= FDCAN_DLC_BYTES_8)
			len = pRxHeader.DataLength;
		if (pRxHeader.DataLength <= FDCAN_DLC_BYTES_12)
			len = 12;
		if (pRxHeader.DataLength <= FDCAN_DLC_BYTES_16)
			len = 16;
		if (pRxHeader.DataLength <= FDCAN_DLC_BYTES_20)
			len = 20;
		if (pRxHeader.DataLength <= FDCAN_DLC_BYTES_24)
			len = 24;
		if (pRxHeader.DataLength <= FDCAN_DLC_BYTES_32)
			len = 32;
		if (pRxHeader.DataLength <= FDCAN_DLC_BYTES_48)
			len = 48;
		if (pRxHeader.DataLength <= FDCAN_DLC_BYTES_64)
			len = 64;

		return len;
	}
	return 0;
}
/*user data struct start*/

/*user data struct end*/
uint8_t rx_data1[8] = {0};
uint16_t rec_id1;
void fdcan1_rx_callback(void)
{
	// rec_id:数据包id rx_data:数据
	fdcanx_receive(&hfdcan1, &rec_id1, rx_data1);
	
	    

		
	/*user decode fcn start*/
			 if(rec_id1 == 0x205) 
		{
        M3508_fbkdata(&lift_motor, rx_data1);
    }
			 if(rec_id1 == 0x206) 
		{      
        M2006_fbkdata(&trans, rx_data1);
    }
		
		
		if(rec_id1==motor_lf.id)
    M3508_fbkdata(&motor_lf,rx_data1);
    if(rec_id1==motor_lb.id)
    M3508_fbkdata(&motor_lb,rx_data1);
    if(rec_id1==motor_rf.id)
    M3508_fbkdata(&motor_rf,rx_data1);
    if(rec_id1==motor_rb.id)
    M3508_fbkdata(&motor_rb,rx_data1);
		
		
	/*user decode fcn end*/
}
uint8_t rx_data2[8] = {0};
uint16_t rec_id2;
void fdcan2_rx_callback(void)
{
	// rec_id:数据包id rx_data:数据
	fdcanx_receive(&hfdcan2, &rec_id2, rx_data2);
	/*user decode fcn start*/
	    if(rec_id2 == 0x03) {

        dm_fdkdata(&max_motor, rx_data2);
			
    }
    if(rec_id2 == 0x04) {
  
        dm_fdkdata(&min_motor, rx_data2);
    }

	/*user decode fcn end*/
}
uint8_t rx_data3[32] = {0};
uint16_t rec_id3;
void fdcan3_rx_callback(void)
{
	// rec_id:数据包id rx_data:数据
	fdcanx_receive(&hfdcan3, &rec_id3, rx_data3);
	/*user decode fcn start*/
	    // 接收 CAN3 数据
    if(rec_id3 == 0x03) {

        dm_fdkdata(&finesse_motor, rx_data3);

    }
    if(rec_id3 == 0x04) {

        dm_fdkdata(&pitch_motor, rx_data3);
    }
    if(rec_id3 == 0x201) {
       
        M2006_fbkdata(&roll, rx_data3);
    }
	
	/*user decode fcn end*/
}

void HAL_FDCAN_RxFifo0Callback(FDCAN_HandleTypeDef *hfdcan, uint32_t RxFifo0ITs)
{
	if (hfdcan == &hfdcan1)
	{
		fdcan1_rx_callback();
	}
	if (hfdcan == &hfdcan2)
	{
		fdcan2_rx_callback();
	}
	if (hfdcan == &hfdcan3)
	{
		fdcan3_rx_callback();
	}
}
