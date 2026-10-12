/* PACE STM32H723VG: preserve the Keil project's 1MB flash and
 * 128KB DTCM + 128KB AXI allocations. UART DMA lives only in AXI. */
ENTRY(Reset_Handler)
MEMORY
{
  FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 1024K
  DTCM (xrw) : ORIGIN = 0x20000000, LENGTH = 128K
  AXI (xrw) : ORIGIN = 0x24000000, LENGTH = 128K
}
_estack = ORIGIN(DTCM) + LENGTH(DTCM);
_Min_Heap_Size = 0x400;
_Min_Stack_Size = 0x2000;
SECTIONS
{
  .isr_vector : { . = ALIGN(4); KEEP(*(.isr_vector)) . = ALIGN(4); } > FLASH
  .text :
  {
    . = ALIGN(4); *(.text) *(.text*) *(.glue_7) *(.glue_7t)
    KEEP(*(.init)) KEEP(*(.fini)) . = ALIGN(4); _etext = .;
  } > FLASH
  .rodata : { . = ALIGN(4); *(.rodata) *(.rodata*) . = ALIGN(4); } > FLASH
  .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } > FLASH
  .ARM.exidx : { __exidx_start = .; *(.ARM.exidx*) __exidx_end = .; } > FLASH
  .preinit_array :
  { PROVIDE_HIDDEN(__preinit_array_start = .); KEEP(*(.preinit_array*)) PROVIDE_HIDDEN(__preinit_array_end = .); } > FLASH
  .init_array :
  { PROVIDE_HIDDEN(__init_array_start = .); KEEP(*(SORT(.init_array.*))) KEEP(*(.init_array*)) PROVIDE_HIDDEN(__init_array_end = .); } > FLASH
  .fini_array :
  { PROVIDE_HIDDEN(__fini_array_start = .); KEEP(*(SORT(.fini_array.*))) KEEP(*(.fini_array*)) PROVIDE_HIDDEN(__fini_array_end = .); } > FLASH
  _sidata = LOADADDR(.data);
  .data :
  { . = ALIGN(4); _sdata = .; *(.data) *(.data*) . = ALIGN(4); _edata = .; } > DTCM AT> FLASH
  .bss (NOLOAD) :
  { . = ALIGN(4); _sbss = .; __bss_start__ = .; *(.bss) *(.bss*) *(COMMON) . = ALIGN(4); _ebss = .; __bss_end__ = .; } > DTCM
  ._user_heap_stack (NOLOAD) :
  { . = ALIGN(8); PROVIDE(end = .); PROVIDE(_end = .); . += _Min_Heap_Size; . += _Min_Stack_Size; . = ALIGN(8); } > DTCM
  /* NOLOAD RX storage is filled by DMA before the app consumes received bytes. */
  .pace_dma ORIGIN(AXI) (NOLOAD) :
  { . = ALIGN(32); __pace_dma_start__ = .; KEEP(*(.pace_dma)) KEEP(*(.pace_dma.*)) . = ALIGN(32); __pace_dma_end__ = .; } > AXI
  .ARM.attributes 0 : { *(.ARM.attributes) }
  /DISCARD/ : { *(.note.GNU-stack) }
}
ASSERT(__pace_dma_start__ == 0x24000000, "PACE DMA must start in AXI SRAM")
ASSERT(__pace_dma_end__ <= ORIGIN(AXI) + LENGTH(AXI), "PACE DMA exceeds AXI SRAM")
ASSERT(_ebss + _Min_Heap_Size + _Min_Stack_Size <= _estack, "DTCM data/heap/stack overflow")
