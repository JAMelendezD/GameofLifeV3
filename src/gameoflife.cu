// -----------------------------------------------------------------------------
// gameoflife.cu : Conways game of life in C++ with a cuda kernel.
// -----------------------------------------------------------------------------

#include "gameoflife.h"

#include <cstdint>
#include <cstddef>
#include <stdexcept>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// RAII wrapper allocates memory and frees it in an automatic manner when
// it goes out of scope
// ---------------------------------------------------------------------------
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        void* p = nullptr;
        if (cudaMalloc(&p, count_ * sizeof(T)) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed");
        }
        ptr_ = static_cast<T*>(p);
    }

    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }

    // Disable copying existing buffer from the constructor 
    DeviceBuffer(const DeviceBuffer&) = delete;
    // Disable copying existing buffer by assignment
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Move is allowed: new object takes ownership of other,
    // leaving other empty.
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    // Move assignment: destroy current resource, take ownership of other,
    // leaving other empty.
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    // getter for the buffer pointer
    T* get() const noexcept { return ptr_; }
    // getter for the number of elements in the buffer
    std::size_t size() const noexcept { return count_; }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

// ---------------------------------------------------------------------------
// Cuda kernel for the game of life
// ---------------------------------------------------------------------------
__global__ void gol_step_kernel(const uint8_t* in, uint8_t* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;  // guard: grid may be padded past the edge

    // Toroidal (wrap-around) boundary conditions lambda function.
    auto wrap = [](int v, int n) { return (v + n) % n; };

    int alive_neighbours = 0;
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            // self do nothing
            if (dx == 0 && dy == 0) continue;
            // get all PBC neigbor positions
            int nx = wrap(x + dx, width);
            int ny = wrap(y + dy, height);
            // sum them to get the alive total
            alive_neighbours += in[ny * width + nx];
        }
    }

    // am I alive or dead?
    uint8_t self = in[y * width + x];
    // Conway's rules: a live cell with 2-3 neighbours survives, a dead cell
    // with exactly 3 neighbours is born, everything else dies / stays dead.
    uint8_t next = (self && (alive_neighbours == 2 || alive_neighbours == 3)) ||
                   (!self && alive_neighbours == 3);
    out[y * width + x] = next;
}

// ---------------------------------------------------------------------------
// C API implementation.
// ---------------------------------------------------------------------------
struct GolHandle {
    int width;
    int height;
    // two buffers 
    DeviceBuffer<uint8_t> a;
    DeviceBuffer<uint8_t> b;
    bool a_is_current = true;

    GolHandle(int w, int h)
        : width(w), height(h),
          a(static_cast<std::size_t>(w) * h),
          b(static_cast<std::size_t>(w) * h) {}
};

extern "C" GolHandle* gol_create(int width, int height, const uint8_t* initial_state) {
    try {
        auto* h = new GolHandle(width, height);
        cudaMemcpy(h->a.get(), initial_state,
                   static_cast<std::size_t>(width) * height,
                   cudaMemcpyHostToDevice);
        return h;
    } catch (...) {
        return nullptr;
    }
}

extern "C" void gol_step(GolHandle* h, int n_steps) {
    if (!h) return;

    // Block size 16x16 = 256 threads/block is a common, safe default.
    // Grid size is just "enough blocks to cover the board".
    dim3 block(16, 16);
    dim3 grid((h->width + block.x - 1) / block.x,
              (h->height + block.y - 1) / block.y);

    for (int i = 0; i < n_steps; ++i) {
        // who is current
        const uint8_t* src = h->a_is_current ? h->a.get() : h->b.get();
        // who is next
        uint8_t* dst       = h->a_is_current ? h->b.get() : h->a.get();
        gol_step_kernel<<<grid, block>>>(src, dst, h->width, h->height);
        h->a_is_current = !h->a_is_current;  // swap which buffer is "current"
    }
    cudaDeviceSynchronize();  // block until the GPU actually finishes
}

extern "C" void gol_get_state(GolHandle* h, uint8_t* out_state) {
    if (!h) return;
    const uint8_t* src = h->a_is_current ? h->a.get() : h->b.get();
    cudaMemcpy(out_state, src,
               static_cast<std::size_t>(h->width) * h->height,
               cudaMemcpyDeviceToHost);
}

extern "C" void gol_destroy(GolHandle* h) {
    delete h;
}
