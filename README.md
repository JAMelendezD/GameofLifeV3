# CUDA + modern C++ + Python, via Conway's Game of Life
   detailed balance. The standard fix is a **checkerboard (red-black)
   update**: color the grid like a checkerboard, update all "black" cells
   in parallel (they only ever depend on "red" neighbours, which are frozen
   this half-step), then update all "red" cells the same way. Two kernel
   launches per full sweep instead of one.
