#ifndef GOL_H
#define GOL_H

#include <stdint.h>

/* Use extern C on g++ compiler not gcc
 * Uses C linkage so c++ dont mangle the names. */
#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle: callers never see the C++ class behind it, only a pointer.
 * This keeps our CUDA/C++ internals free to change without breaking ABI. */
typedef struct GolHandle GolHandle;

/* Allocates device memory and copies initial_state (row-major, height*width
 * bytes, 0 or 1) onto the GPU. Returns NULL on failure. */
GolHandle* gol_create(int width, int height, const uint8_t* initial_state);

/* Advances the simulation by n_steps generations. */
void gol_step(GolHandle* h, int n_steps);

/* Copies the current generation back into a host buffer you own
 * (must be height*width bytes). */
void gol_get_state(GolHandle* h, uint8_t* out_state);

/* Frees device memory and the handle. */
void gol_destroy(GolHandle* h);

#ifdef __cplusplus
}
#endif

#endif
