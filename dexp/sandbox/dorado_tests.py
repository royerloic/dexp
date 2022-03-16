from arbol import asection, aprint
from napari import gui_qt, Viewer

from dexp.datasets.zarr_dataset import ZDataset
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.multiview_lightsheet.fusion.simview import simview_fuse_2C2L

filepath = '/mnt/raid0/dexp_datasets/photomanip/ch0.zarr'

zdataset = ZDataset(path=filepath, mode='r')

aprint(zdataset.channels())

C0L0 = zdataset.get_stack('C0L0', 4)
C0L1 = zdataset.get_stack('C0L1', 4)
C1L0 = zdataset.get_stack('C1L0', 4)
C1L1 = zdataset.get_stack('C1L1', 4)

with CupyBackend() as backend:
    with asection("simview_fuse_2I2D"):
        CxLx, model = simview_fuse_2C2L(C0L0, C0L1, C1L0, C1L1,
                                        clip_too_high=1024)
    aprint(f"Model = {model}")

    with asection("to_numpy"):
        CxLx = Backend.to_numpy(CxLx)

    with gui_qt():
        def _c(array):
            return Backend.to_numpy(array)


        viewer = Viewer()
        viewer.add_image(_c(C0L0), name='C0L0', contrast_limits=(0, 1000), scale=(4, 1, 1), blending='additive', visible=False)
        viewer.add_image(_c(C0L1), name='C0L1', contrast_limits=(0, 1000), scale=(4, 1, 1), blending='additive', visible=False)
        viewer.add_image(_c(C1L0), name='C1L0', contrast_limits=(0, 1000), scale=(4, 1, 1), blending='additive', visible=False)
        viewer.add_image(_c(C1L1), name='C1L1', contrast_limits=(0, 1000), scale=(4, 1, 1), blending='additive', visible=False)
        viewer.add_image(_c(CxLx), name='CxLx', contrast_limits=(0, 1200), scale=(4, 1, 1), blending='additive', colormap='viridis')
        # viewer.add_image(_c(CxLx_deconvolved), name='CxLx_deconvolved', contrast_limits=(0, 1000), scale=(4, 1, 1), blending='additive', colormap='viridis')
