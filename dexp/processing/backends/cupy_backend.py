from typing import Any

import numpy

from dexp.processing.backends.backend import Backend


class CupyBackend(Backend):

    def __init__(self, device=0, enable_cub: bool = True, enable_cutensor: bool = True):
        """ Instanciates a Numpy-based Image Processing backend

        """
        self.device = device
        from cupy.cuda import cub, cutensor
        cub.available = enable_cub
        cutensor.available = enable_cutensor

        from cupy.cuda import set_allocator, set_pinned_memory_allocator
        # Disable memory pool for device memory (GPU)
        set_allocator(None)
        # Disable memory pool for pinned memory (CPU).
        set_pinned_memory_allocator(None)

        ## Important: Leave this, this is to make sure that the ndimage package works properly!
        exec("import cupyx.scipy.ndimage")

    def close(self):
        # Nothing to do
        pass

    def to_numpy(self, array, dtype=None, copy: bool = False) -> numpy.ndarray:
        import cupy
        if cupy.get_array_module(array) == cupy:
            array = cupy.asnumpy(array)
            if dtype:
                return array.astype(dtype, copy=copy)
            elif copy:
                return array.copy()
            else:
                return array
        else:
            if dtype:
                return array.astype(dtype, copy=copy)
            elif copy:
                return array.copy()
            else:
                return array

    def to_backend(self, array, dtype=None, copy: bool = False) -> Any:
        import cupy
        if cupy.get_array_module(array) == cupy:
            if dtype:
                return array.astype(dtype, copy=copy)
            elif copy:
                return array.copy()
            else:
                return array
        else:
            with cupy.cuda.Device(self.device):
                return cupy.asarray(array, dtype=dtype)

    def get_xp_module(self, array=None) -> Any:
        if array is not None:
            import cupy
            return cupy.get_array_module(array)
        else:
            import cupy
            return cupy

    def get_sp_module(self, array=None) -> Any:
        if array is not None:
            import cupyx
            return cupyx.scipy.get_array_module(array)
        else:
            import cupyx
            return cupyx.scipy