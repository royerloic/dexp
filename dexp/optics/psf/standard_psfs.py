import numpy

from dexp.optics.psf.microscope_psf import MicroscopePSF


def nikon16x08na(xy_size=17, z_size=17, dxy=0.485, dz=2, wvl=0.561):
    return generate_psf(dxy=dxy,
                        dz=dz,
                        xy_size=xy_size,
                        z_size=z_size,
                        M=16,
                        NA=0.8,
                        n=1.33,
                        wd=3000,
                        tl=165.0 * 1.0e+3,
                        wvl=wvl)


def olympus20x10na(xy_size=17, z_size=17, dxy=0.439, dz=1.5, wvl=0.561):
    return generate_psf(dxy=dxy,
                        dz=dz,
                        xy_size=xy_size,
                        z_size=z_size,
                        M=16,
                        NA=0.8,
                        n=1.33,
                        wd=2000,
                        tl=133.0 * 1.0e+3,
                        wvl=wvl)


def generate_psf(dxy, dz, xy_size, z_size, M=16, NA=0.8, n=1.33, wd=3000, tl=133.0 * 1.0e+3, wvl=561.):
    """
    Generates a 3D PSF array.

    :param dxy: voxel dimension along xy (microns)
    :param dz: voxel dimension along z (microns)
    :param xy_size: size of PSF kernel along x and y (odd integer)
    :param z_size: size of PSF kernel along z (odd integer)

    """

    psf_gen = MicroscopePSF()

    # Microscope parameters.
    psf_gen.parameters["M"] = M  # magnification
    psf_gen.parameters["NA"] = NA  # numerical aperture
    psf_gen.parameters["ni0"] = n
    psf_gen.parameters["ni"] = n
    psf_gen.parameters["ns"] = n
    psf_gen.parameters["ti0"] = wd
    psf_gen.parameters["zd0"] = tl

    lz = (z_size) * dz
    z_offset = -(lz - 2 * dz) / 2
    pz = numpy.arange(0, lz, dz)

    # gLXYZParticleScan(self, dxy, xy_size, pz, normalize = True, wvl = 0.6, zd = None, zv = 0.0):
    psf_xyz_array = psf_gen.gLXYZParticleScan(
        dxy=dxy,
        xy_size=xy_size,
        pz=pz,
        zv=z_offset,
        wvl=wvl
    )

    psf_xyz_array /= psf_xyz_array.sum()

    return psf_xyz_array