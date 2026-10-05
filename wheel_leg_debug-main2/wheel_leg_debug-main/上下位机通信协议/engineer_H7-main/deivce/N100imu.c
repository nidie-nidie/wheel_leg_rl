#include "N100imu.h"

static const uint8_t CRC8Table[] = {
    0, 94, 188, 226, 97, 63, 221, 131, 194,
    156, 126, 32, 163, 253, 31, 65, 157, 195,
    33, 127, 252, 162, 64, 30, 95, 1, 227,
    189, 62, 96, 130, 220, 35, 125, 159,
    193, 66, 28, 254, 160, 225, 191, 93,
    3, 128, 222, 60, 98, 190, 224, 2, 92,
    223, 129, 99, 61, 124, 34, 192, 158, 29,
    67, 161, 255, 70, 24, 250, 164, 39, 121,
    155, 197, 132, 218, 56, 102, 229, 187,
    89, 7, 219, 133, 103, 57, 186, 228, 6,
    88, 25, 71, 165, 251, 120, 38, 196, 154,
    101, 59, 217, 135, 4, 90, 184, 230, 167,
    249, 27, 69, 198, 152, 122, 36, 248, 166,
    68, 26, 153, 199, 37, 123, 58, 100, 134,
    216, 91, 5, 231, 185, 140, 210, 48, 110,
    237, 179, 81, 15, 78, 16, 242, 172, 47, 113,
    147, 205, 17, 79, 173, 243, 112, 46, 204, 146,
    211, 141, 111, 49, 178, 236, 14, 80,
    175, 241, 19, 77, 206, 144, 114, 44,
    109, 51, 209, 143, 12, 82, 176, 238,
    50, 108, 142, 208, 83, 13, 239, 177,
    240, 174, 76, 18, 145, 207, 45, 115, 202,
    148, 118, 40, 171, 245, 23, 73, 8, 86, 180,
    234, 105, 55, 213, 139, 87, 9, 235, 181, 54,
    104, 138, 212, 149, 203, 41, 119, 244, 170,
    72, 22, 233, 183, 85, 11, 136, 214, 52, 106,
    43, 117, 151, 201, 74, 20, 246, 168, 116,
    42, 200, 150, 21, 75, 169, 247, 182, 232, 10, 84, 215, 137, 107, 53};

uint8_t CRC8_Table(uint8_t *p, uint8_t counter)
{
    uint8_t crc8 = 0;
    for (int i = 0; i < counter; i++)
    {
        uint8_t value = p[i];
        uint8_t new_index = crc8 ^ value;
        crc8 = CRC8Table[new_index];
    }
    return (crc8);
}
static const uint16_t CRC16Table[256] =
    {
        0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5,
        0x60C6, 0x70E7, 0x8108, 0x9129, 0xA14A, 0xB16B,
        0xC18C, 0xD1AD, 0xE1CE, 0xF1EF, 0x1231, 0x0210,
        0x3273, 0x2252, 0x52B5, 0x4294, 0x72F7, 0x62D6,
        0x9339, 0x8318, 0xB37B, 0xA35A, 0xD3BD, 0xC39C,
        0xF3FF, 0xE3DE, 0x2462, 0x3443, 0x0420, 0x1401,
        0x64E6, 0x74C7, 0x44A4, 0x5485, 0xA56A, 0xB54B,
        0x8528, 0x9509, 0xE5EE, 0xF5CF, 0xC5AC, 0xD58D,
        0x3653, 0x2672, 0x1611, 0x0630, 0x76D7, 0x66F6,
        0x5695, 0x46B4, 0xB75B, 0xA77A, 0x9719, 0x8738,
        0xF7DF, 0xE7FE, 0xD79D, 0xC7BC, 0x48C4, 0x58E5,
        0x6886, 0x78A7, 0x0840, 0x1861, 0x2802, 0x3823,
        0xC9CC, 0xD9ED, 0xE98E, 0xF9AF, 0x8948, 0x9969,
        0xA90A, 0xB92B, 0x5AF5, 0x4AD4, 0x7AB7, 0x6A96,
        0x1A71, 0x0A50, 0x3A33, 0x2A12, 0xDBFD, 0xCBDC,
        0xFBBF, 0xEB9E, 0x9B79, 0x8B58, 0xBB3B, 0xAB1A,
        0x6CA6, 0x7C87, 0x4CE4, 0x5CC5, 0x2C22, 0x3C03,
        0x0C60, 0x1C41, 0xEDAE, 0xFD8F, 0xCDEC, 0xDDCD,
        0xAD2A, 0xBD0B, 0x8D68, 0x9D49, 0x7E97, 0x6EB6,
        0x5ED5, 0x4EF4, 0x3E13, 0x2E32, 0x1E51, 0x0E70,
        0xFF9F, 0xEFBE, 0xDFDD, 0xCFFC, 0xBF1B, 0xAF3A,
        0x9F59, 0x8F78, 0x9188, 0x81A9, 0xB1CA, 0xA1EB,
        0xD10C, 0xC12D, 0xF14E, 0xE16F, 0x1080, 0x00A1,
        0x30C2, 0x20E3, 0x5004, 0x4025, 0x7046, 0x6067,
        0x83B9, 0x9398, 0xA3FB, 0xB3DA, 0xC33D, 0xD31C,
        0xE37F, 0xF35E, 0x02B1, 0x1290, 0x22F3, 0x32D2,
        0x4235, 0x5214, 0x6277, 0x7256, 0xB5EA, 0xA5CB,
        0x95A8, 0x8589, 0xF56E, 0xE54F, 0xD52C, 0xC50D,
        0x34E2, 0x24C3, 0x14A0, 0x0481, 0x7466, 0x6447,
        0x5424, 0x4405, 0xA7DB, 0xB7FA, 0x8799, 0x97B8,
        0xE75F, 0xF77E, 0xC71D, 0xD73C, 0x26D3, 0x36F2,
        0x0691, 0x16B0, 0x6657, 0x7676, 0x4615, 0x5634,
        0xD94C, 0xC96D, 0xF90E, 0xE92F, 0x99C8, 0x89E9,
        0xB98A, 0xA9AB, 0x5844, 0x4865, 0x7806, 0x6827,
        0x18C0, 0x08E1, 0x3882, 0x28A3, 0xCB7D, 0xDB5C, 0xEB3F,
        0xFB1E, 0x8BF9, 0x9BD8, 0xABBB, 0xBB9A, 0x4A75, 0x5A54,
        0x6A37, 0x7A16, 0x0AF1, 0x1AD0, 0x2AB3, 0x3A92, 0xFD2E,
        0xED0F, 0xDD6C, 0xCD4D, 0xBDAA, 0xAD8B, 0x9DE8, 0x8DC9,
        0x7C26, 0x6C07, 0x5C64, 0x4C45, 0x3CA2, 0x2C83, 0x1CE0,
        0x0CC1, 0xEF1F, 0xFF3E, 0xCF5D, 0xDF7C, 0xAF9B, 0xBFBA,
        0x8FD9, 0x9FF8, 0x6E17, 0x7E36, 0x4E55, 0x5E74, 0x2E93, 0x3EB2, 0x0ED1, 0x1EF0};
uint16_t CRC16_Table(uint8_t *p, uint8_t counter)
{
    uint16_t crc16 = 0;
    for (int i = 0; i < counter; i++)
    {
        uint8_t value = p[i];
        crc16 = CRC16Table[((crc16 >> 8) ^ value) & 0xff] ^ (crc16 << 8);
    }
    return (crc16);
}

float HEX_to_Float_New(uint8_t *data, uint8_t mode)
{
    float fa = 0;
    uint8_t uc[4];
    if (mode == 0)
    {
        uc[3] = data[0];
        uc[2] = data[1];
        uc[1] = data[2];
        uc[0] = data[3];
    }
    else
    {
        uc[0] = data[0];
        uc[1] = data[1];
        uc[2] = data[2];
        uc[3] = data[3];
    }
    memcpy(&fa, uc, 4);
    return fa;
}
//时间戳解包函数
long long timestamp(uint8_t Data_1, uint8_t Data_2, uint8_t Data_3, uint8_t Data_4)
{
    unsigned long temp;  // 32位16进制数
    uint16_t H_16, L_16; //存放高16位、低16位
    H_16 = Data_1 << 8 | Data_2;
    L_16 = Data_3 << 8 | Data_4;
    //将融合成16位的数据组合成32位数据
    temp = (unsigned long)H_16 << 16 | (unsigned long)L_16;
    return temp;
}
void n100imu_fbkdata(N100imu_t *n100)
{
    float dt = DWT_GetDeltaT(&(n100->tsm.count));
    n100->tsm.no_data_time += dt;
    int pack_ok = 0;
    int pack_loss = 0;
    for (int i = 0; i < 100; i++)
    {
        if (n100->tsm.rx_pack_state[i] == 1)
            pack_ok++;
        else if (n100->tsm.rx_pack_state[i] == 2)
            pack_loss++;
        else
            break;
    }
    if ((pack_loss + pack_ok) > 0)
        n100->tsm.pack_loss_rate = ((float)pack_ok) / ((float)(pack_loss + pack_ok));
    /*****************************/
    if (n100->frame_header.data_type == 0x40) // msg_imu
    {
        uint8_t temp[4] = {0};
        temp[0] = n100->data_buf[7], temp[1] = n100->data_buf[8], temp[2] = n100->data_buf[9], temp[3] = n100->data_buf[10];
        n100->msg_imu.gyroscope_x = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[11], temp[1] = n100->data_buf[12], temp[2] = n100->data_buf[13], temp[3] = n100->data_buf[14];
        n100->msg_imu.gyroscope_y = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[15], temp[1] = n100->data_buf[16], temp[2] = n100->data_buf[17], temp[3] = n100->data_buf[18];
        n100->msg_imu.gyroscope_z = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[19], temp[1] = n100->data_buf[20], temp[2] = n100->data_buf[21], temp[3] = n100->data_buf[22];
        n100->msg_imu.accelerometer_x = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[23], temp[1] = n100->data_buf[24], temp[2] = n100->data_buf[25], temp[3] = n100->data_buf[26];
        n100->msg_imu.accelerometer_y = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[27], temp[1] = n100->data_buf[28], temp[2] = n100->data_buf[29], temp[3] = n100->data_buf[30];
        n100->msg_imu.accelerometer_z = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[31], temp[1] = n100->data_buf[32], temp[2] = n100->data_buf[33], temp[3] = n100->data_buf[34];
        n100->msg_imu.magnetometer_x = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[35], temp[1] = n100->data_buf[36], temp[2] = n100->data_buf[37], temp[3] = n100->data_buf[38];
        n100->msg_imu.magnetometer_y = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[39], temp[1] = n100->data_buf[40], temp[2] = n100->data_buf[41], temp[3] = n100->data_buf[42];
        n100->msg_imu.magnetometer_z = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[43], temp[1] = n100->data_buf[44], temp[2] = n100->data_buf[45], temp[3] = n100->data_buf[46];
        n100->msg_imu.imu_temperature = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[47], temp[1] = n100->data_buf[48], temp[2] = n100->data_buf[49], temp[3] = n100->data_buf[50];
        n100->msg_imu.Pressure = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[51], temp[1] = n100->data_buf[52], temp[2] = n100->data_buf[53], temp[3] = n100->data_buf[54];
        n100->msg_imu.pressure_temperature = HEX_to_Float_New(temp, 1);

        //下面是数据解包过程
        n100->msg_imu.Timestamp = timestamp(n100->data_buf[58], n100->data_buf[57], n100->data_buf[56], n100->data_buf[55]);
    }
    if (n100->frame_header.data_type == 0x41) // msg_AHRS
    {
        uint8_t temp[4];
        temp[0] = n100->data_buf[7], temp[1] = n100->data_buf[8], temp[2] = n100->data_buf[9], temp[3] = n100->data_buf[10];
        n100->msg_AHRS.RollSpeed = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[11], temp[1] = n100->data_buf[12], temp[2] = n100->data_buf[13], temp[3] = n100->data_buf[14];
        n100->msg_AHRS.PitchSpeed = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[15], temp[1] = n100->data_buf[16], temp[2] = n100->data_buf[17], temp[3] = n100->data_buf[18];
        n100->msg_AHRS.HeadingSpeed = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[19], temp[1] = n100->data_buf[20], temp[2] = n100->data_buf[21], temp[3] = n100->data_buf[22];
        n100->msg_AHRS.Roll = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[23], temp[1] = n100->data_buf[24], temp[2] = n100->data_buf[25], temp[3] = n100->data_buf[26];
        n100->msg_AHRS.Pitch = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[27], temp[1] = n100->data_buf[28], temp[2] = n100->data_buf[29], temp[3] = n100->data_buf[30];
        n100->msg_AHRS.Heading = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[31], temp[1] = n100->data_buf[32], temp[2] = n100->data_buf[33], temp[3] = n100->data_buf[34];
        n100->msg_AHRS.Qw = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[35], temp[1] = n100->data_buf[36], temp[2] = n100->data_buf[37], temp[3] = n100->data_buf[38];
        n100->msg_AHRS.Qx = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[39], temp[1] = n100->data_buf[40], temp[2] = n100->data_buf[41], temp[3] = n100->data_buf[42];
        n100->msg_AHRS.Qy = HEX_to_Float_New(temp, 1);

        temp[0] = n100->data_buf[43], temp[1] = n100->data_buf[44], temp[2] = n100->data_buf[45], temp[3] = n100->data_buf[46];
        n100->msg_AHRS.Qz = HEX_to_Float_New(temp, 1);
        n100->msg_AHRS.Timestamp = timestamp(n100->data_buf[50], n100->data_buf[49], n100->data_buf[48], n100->data_buf[47]); // unit: us
    }
}

void n100imu_call(N100imu_t *n100, uint8_t data[])
{
    //移位
    for (int i = 0; i < 99; i++)
        n100->tsm.rx_pack_state[99 - i] = n100->tsm.rx_pack_state[98 - i];
    if (data[sizeof(n100->frame_header) - 1] == CRC8_Table(data, sizeof(n100->frame_header) - 1))
    // if (Verify_CRC8_Check_Sum(data, sizeof(n100->frame_header)))
    {
        memcpy(&n100->frame_header, data, sizeof(n100->frame_header));
        uint16_t get_crc16 = CRC16_Table(&data[sizeof(n100->frame_header) + 2], n100->frame_header.data_length);
        uint16_t crc16 = ((uint16_t)(data[sizeof(n100->frame_header)]) << 8) + data[sizeof(n100->frame_header) + 1];
        if (get_crc16 == crc16)
        {
            memcpy(n100->data_buf, data, 100);
            n100->tsm.rx_pack_state[0] = 1;
        }
        else
            n100->tsm.rx_pack_state[0] = 2;
    }
    else
        n100->tsm.rx_pack_state[0] = 2;
    n100->tsm.no_data_time = 0;
    DWT_GetDeltaT(&(n100->tsm.count)); //更新记时点
}
