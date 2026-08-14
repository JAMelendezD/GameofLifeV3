"""
Thin ctypes wrapper around libgameoflife.so.
ctypes is the simplest of several options for calling a C API from Python
(others: cffi, pybind11, Cython). It needs no compilation step of its own
"""

import ctypes
import os

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))

class GameOfLife:

    def __init__(self, initial_state: np.ndarray, backend="gpu"):

        if backend == "gpu":
            library = "libgameoflife.so"
        elif backend == "cpu":
            library = "libgameoflife_cpu.so"
        else:
            raise ValueError("backend must be 'cpu' or 'gpu'")


        _lib = ctypes.CDLL(os.path.join(_here, "..", "lib", library))

        # Describing argtypes/restype isn't strictly required, but it lets ctypes
        # validate arguments and do the right pointer conversions instead of
        # silently doing the wrong thing.
        _lib.gol_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8)]
        _lib.gol_create.restype = ctypes.c_void_p

        _lib.gol_step.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _lib.gol_step.restype = None

        _lib.gol_get_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8)]
        _lib.gol_get_state.restype = None

        _lib.gol_destroy.argtypes = [ctypes.c_void_p]
        _lib.gol_destroy.restype = None

        self._lib = _lib

        initial_state = np.ascontiguousarray(initial_state, dtype=np.uint8)
        self.height, self.width = initial_state.shape

        ptr = initial_state.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self._handle = _lib.gol_create(self.width, self.height, ptr)
        if not self._handle:
            raise RuntimeError("gol_create failed - check your CUDA device and build")

    def step(self, n: int = 1) -> None:
        """Advance the simulation by n generations, entirely on the GPU."""
        self._lib.gol_step(self._handle, n)

    def state(self) -> np.ndarray:
        """Copy the current board back to the CPU as a (height, width) uint8 array."""
        buf = np.empty((self.height, self.width), dtype=np.uint8)
        ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self._lib.gol_get_state(self._handle, ptr)
        return buf

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.gol_destroy(handle)
            self._handle = None
