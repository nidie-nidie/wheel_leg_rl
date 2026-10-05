#ifndef FIFO_H
#define FIFO_H

// refer to https://blog.csdn.net/linyt/article/details/53355355 to get idea

// 声明 一个 结构体 kfifo

typedef struct KFIFO
{
    unsigned char *buffer; /* the buffer holding the data */
    unsigned int size;     /* the size of the allocated buffer */
    unsigned int in;       /* data is added at offset (in % size) */
    unsigned int out;      /* data is extracted from off. (out % size) */
    /*STM32 只有一个核心，同一时刻只能写或者读，因此不需要*/
    //    volatile unsigned int *lock; /* protects concurrent modifications */
} kfifo;

unsigned int roundup_pow_of_two(unsigned int date_roundup_pow_of_two);

kfifo *kfifo_alloc(unsigned int size);

unsigned int __kfifo_put(kfifo *fifo, unsigned char *buffer, unsigned int len);

unsigned int __kfifo_get(kfifo *fifo, unsigned char *buffer, unsigned int len);

unsigned int can_fifo_get(kfifo *fifo, unsigned char *buffer, unsigned int len);

void kfifo_reset(kfifo *fifo);

unsigned int kfifo_len(kfifo *fifo);

#endif