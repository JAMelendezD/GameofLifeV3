// -----------------------------------------------------------------------------
// gameoflife.cu : Conways game of life in C++ with a cuda kernel.
// -----------------------------------------------------------------------------

#include "gameoflife.h"

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// CPU implementation of the game of life.
// ---------------------------------------------------------------------------
static void gol_step_kernel(
    const uint8_t* in,
    uint8_t* out,
    int width,
    int height
) {
    // Toroidal (wrap-around) boundary conditions lambda function.
    auto wrap = [](int v, int n) { return (v + n) % n; };

    // parallel on CPUs
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int alive_neighbours = 0;

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    // self do nothing
                    if (dx == 0 && dy == 0) continue;

                    // get all PBC neighbor positions
                    int nx = wrap(x + dx, width);
                    int ny = wrap(y + dy, height);

                    // sum them to get the alive total
                    alive_neighbours += in[ny * width + nx];
                }
            }

            // am I alive or dead?
            uint8_t self = in[y * width + x];

            // Conway's rules: a live cell with 2-3 neighbours survives,
            // a dead cell with exactly 3 neighbours is born.
            uint8_t next =
                (self &&
                 (alive_neighbours == 2 ||
                  alive_neighbours == 3)) ||
                (!self && alive_neighbours == 3);

            out[y * width + x] = next;
        }
    }
}

// ---------------------------------------------------------------------------
// C API implementation.
// ---------------------------------------------------------------------------
struct GolHandle {
    int width;
    int height;
    // two buffers
    std::vector<uint8_t> a;
    std::vector<uint8_t> b;
    bool a_is_current = true;

    GolHandle(int w, int h)
        : width(w), height(h),
          a(static_cast<std::size_t>(w) * h),
          b(static_cast<std::size_t>(w) * h) {}
};

extern "C" GolHandle* gol_create(int width, int height, const uint8_t* initial_state) {
    try {
        auto* h = new GolHandle(width, height);
        std::memcpy(
            h->a.data(),
            initial_state,
            static_cast<std::size_t>(width) * height);
        return h;
    } catch (...) {
        return nullptr;
    }
}

extern "C" void gol_step(GolHandle* h, int n_steps) {
    if (!h) return;

    for (int i = 0; i < n_steps; ++i) {
        // who is current
        const uint8_t* src = h->a_is_current ? h->a.data() : h->b.data();
        // who is next
        uint8_t* dst       = h->a_is_current ? h->b.data() : h->a.data();
        gol_step_kernel(src, dst, h->width, h->height);
        h->a_is_current = !h->a_is_current;  // swap which buffer is "current"
    }
}

extern "C" void gol_get_state(GolHandle* h, uint8_t* out_state) {
    if (!h) return;
    const uint8_t* src = h->a_is_current ? h->a.data() : h->b.data();
    std::memcpy(out_state, src, static_cast<std::size_t>(h->width) * h->height);
}

extern "C" void gol_destroy(GolHandle* h) {
    delete h;
}
