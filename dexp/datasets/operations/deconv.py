from arbol.arbol import aprint
from joblib import Parallel, delayed
from skimage.transform import downscale_local_mean

from dexp.optics.psf.standard_psfs import nikon16x08na, olympus20x10na
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.deconvolution.lr_deconvolution import lucy_richardson_deconvolution
from dexp.processing.utils.scatter_gather_i2i import scatter_gather_i2i
from dexp.utils.timeit import timeit


def dataset_deconv(dataset,
                   path,
                   channels,
                   slicing,
                   store,
                   compression,
                   compression_level,
                   overwrite,
                   chunksize,
                   method,
                   num_iterations,
                   max_correction,
                   power,
                   blind_spot,
                   back_projection,
                   objective,
                   dxy,
                   dz,
                   xy_size,
                   z_size,
                   downscalexy2,
                   workers,
                   devices,
                   check):
    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(path, mode, store)

    for channel in dataset._selected_channels(channels):

        array = dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True)

        if slicing is not None:
            array = array[slicing]

        shape = array.shape
        dim = len(shape)

        if dim == 3:
            chunks = dataset._default_chunks[1:]
        elif dim == 4:
            chunks = dataset._default_chunks

        dest_array = dest_dataset.add_channel(name=channel,
                                              shape=shape,
                                              dtype=array.dtype,
                                              chunks=chunks,
                                              codec=compression,
                                              clevel=compression_level)

        psf_kwargs = {'dxy': dxy * (2 if downscalexy2 else 1),
                      'dz': dz,
                      'xy_size': xy_size,
                      'z_size': z_size}

        if objective == 'nikon16x08na':
            psf_kernel = nikon16x08na(**psf_kwargs)
        elif objective == 'olympus20x10na':
            psf_kernel = olympus20x10na(**psf_kwargs)

        def process(tp, device):

            with CupyBackend(device):
                try:
                    aprint(f"Starting to process time point: {tp} ...")
                    tp_array = array[tp].compute()
                    if downscalexy2:
                        tp_array = downscale_local_mean(tp_array, factors=(1, 2, 2)).astype(tp_array.dtype)

                    if method == 'lr':
                        min_value = tp_array.min()
                        max_value = tp_array.max()

                        def f(image):
                            return lucy_richardson_deconvolution(image=image,
                                                                 psf=psf_kernel,
                                                                 num_iterations=num_iterations,
                                                                 max_correction=max_correction,
                                                                 normalise_minmax=(min_value, max_value),
                                                                 power=power,
                                                                 blind_spot=blind_spot,
                                                                 blind_spot_mode='median+uniform',
                                                                 blind_spot_axis_exclusion=(0,),
                                                                 back_projection=back_projection
                                                                 )

                        with timeit("lucy_richardson_deconvolution"):
                            tp_array = scatter_gather_i2i(f, tp_array, chunks=chunksize, margins=max(xy_size, z_size))
                    else:
                        raise ValueError(f"Unknown deconvolution mode: {method}")

                    tp_array = Backend.to_numpy(tp_array, dtype=dest_array.dtype, force_copy=False)
                    dest_array[tp] = tp_array
                    aprint(f"Done processing time point: {tp} .")

                except Exception as error:
                    aprint(error)
                    aprint(f"Error occurred while copying time point {tp} !")
                    import traceback
                    traceback.print_exc()

        if workers == -1:
            workers = len(devices)

        if workers > 1:
            Parallel(n_jobs=workers, backend='threading')(delayed(process)(tp, devices[tp % len(devices)]) for tp in range(0, shape[0]))
        else:
            for tp in range(0, shape[0]):
                process(tp, devices[0])

    aprint(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()
    dest_dataset.close()
