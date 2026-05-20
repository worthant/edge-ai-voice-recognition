#pragma once
#include "esp_err.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maps the 'model' partition into the CPU address space (XIP from flash).
 * On success, *out_ptr points to the model flatbuffer and *out_size is the
 * partition size in bytes. Pointer is valid for the lifetime of the program. */
esp_err_t model_loader_mmap(const void **out_ptr, size_t *out_size);

#ifdef __cplusplus
}
#endif
