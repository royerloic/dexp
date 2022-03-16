import numpy

from dexp.datasets.zarr_dataset import ZDataset
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.fusion.tg_fusion import fuse_tg_nd
from dexp.processing.registration.warp_multiscale_nd import register_warp_multiscale_nd
from dexp.processing.restoration.dehazing import dehaze
from dexp.processing.utils.normalise import normalise_functions
from dexp.utils.timeit import timeit

dataset1_path = '/mnt/raid0/pisces_datasets/488beads_TL100_range1000um_step0.31_4um_20ms_dualv_1tp_without_stripeReduc_resampled_intervel_1.zarr'
dataset2_path = '/mnt/raid0/pisces_datasets/data1_beads_TL100_range1200um_step0.31_6um_50ms_dualv_1tp_2_resampled_intervel_1.zarr/'
dataset3_path = '/mnt/raid0/pisces_datasets/data2_fish_TL100_range1300um_step0.31_6um_20ms_dualv_300tp_2_resampled_intervel_1_first10tp.zarr'
dataset4_path = '/mnt/raid0/dexp_datasets/tail/raw.zarr'

zdataset = ZDataset(path=dataset4_path, mode='r')

print(zdataset.channels())

tp = 3

view1 = zdataset.get_stack('v0c0_rot', tp)  # [..., 0:-500]
view2 = zdataset.get_stack('v1c0_rot', tp)  # [..., 500:]

# view1 = view1[:,1765-256:1765+256, 1929-1024:1929+1024]
# view2 = view2[:,1765-256:1765+256, 1929-1024:1929+1024]

print(f"view1 shape={view1.shape}, dtype={view1.dtype}")
print(f"view2 shape={view2.shape}, dtype={view2.dtype}")

with CupyBackend(0, enable_memory_pool=False) as backend:
    print(backend)

    with timeit("dehaze"):
        view1 = dehaze(view1, size=25, in_place=True)
        view2 = dehaze(view2, size=25, in_place=True)

    with timeit("to_backend"):
        view1 = Backend.to_backend(view1, dtype=numpy.float32)
        view2 = Backend.to_backend(view2, dtype=numpy.float32)

    with timeit("normalise"):
        norm_fun1, _ = normalise_functions(view1, low=0, high=1024, quantile=0.0001)
        norm_fun2, _ = normalise_functions(view2, low=0, high=1024, quantile=0.0001)

        view1 = norm_fun1(view1)
        view2 = norm_fun2(view2)

    print(f"1: min={view1.min()}, max={view1.max()}")
    print(f"2: min={view2.min()}, max={view2.max()}")

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #
    #     viewer = Viewer()
    #
    #     viewer.add_image(_c(view1), name='view1', colormap='bop blue', blending='additive')
    #     viewer.add_image(_c(view2), name='view2', colormap='bop orange', blending='additive')

    edge_filter = False

    with timeit("register_warp_multiscale_nd"):
        model = register_warp_multiscale_nd(view1, view2,
                                            num_iterations=5,
                                            confidence_threshold=0.3,
                                            edge_filter=edge_filter,
                                            denoise_input_sigma=1)

    with timeit("unwarp"):
        model.integral = True
        view1_reg, view2_reg = model.apply_pair(view1, view2)

    view1 = Backend.to_numpy(view1)
    view2 = Backend.to_numpy(view2)

    with timeit("fuse"):
        fused = fuse_tg_nd(view1_reg, view2_reg)

    from napari import Viewer, gui_qt

    with gui_qt():
        def _c(array):
            return Backend.to_numpy(array)


        viewer = Viewer()
        viewer.add_image(_c(view1), name='view1', colormap='bop blue', blending='additive')
        viewer.add_image(_c(view2), name='view2', colormap='bop orange', blending='additive')
        viewer.add_image(_c(view2_reg), name='view2_reg', colormap='bop orange', blending='additive')
        viewer.add_image(_c(fused), name='fused', colormap='bop purple', blending='additive')
        viewer.add_image(_c(model.vector_field[..., 0]), name='vx', colormap='viridis', blending='additive')
        viewer.add_image(_c(model.vector_field[..., 1]), name='vy', colormap='viridis', blending='additive')
        viewer.add_image(_c(model.vector_field[..., 2]), name='vz', colormap='viridis', blending='additive')
