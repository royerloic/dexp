from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.utils.fit_shape import fit_shape


def test_fit_shape_numpy():
    backend = NumpyBackend()
    _test_fit_shape(backend)


def test_fit_shape_cupy():
    try:
        backend = CupyBackend()
        _test_fit_shape(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_fit_shape(backend, length_xy=128):
    xp = backend.get_xp_module()

    array_1 = xp.random.uniform(0, 1, size=(31, 10, 17))
    array_2 = xp.random.uniform(0, 1, size=(32, 9, 18))

    array_2_fit = fit_shape(backend, array_2.copy(), shape=array_1.shape)

    assert array_2_fit is not array_2
    assert array_2_fit.shape == array_1.shape

    from napari import Viewer, gui_qt
    with gui_qt():
        viewer = Viewer()
        viewer.add_image(array_1, name='array_1')
        viewer.add_image(array_2, name='array_2')
        viewer.add_image(array_2_fit, name='array_2_fit')