![fishcolorproj](https://user-images.githubusercontent.com/1870994/113943035-b61b0c00-97b6-11eb-8cfd-ac78e2976ebb.png)
# **dexp** | Light-sheet Dataset EXploration and Processing 

**dexp** is a [napari](https://napari.org/), [CuPy](https://cupy.dev/), [Zarr](https://zarr.readthedocs.io/en/stable/), and [DASK](https://dask.org/) based library for managing, processing and visualizing light-sheet microscopy datasets. It consists in light-sheet specialised image processing functions (equalisation, denoising, dehazing, registration, fusion, stabilization, deskewing, deconvolution), visualization functions (napari-based viewing, 2D/3D rendering, video compositing and resizing, mp4 generation), as well as dataset management functions (copy, crop, concatenation, tiff conversion). Almost all functions are GPU accelerated via [CuPy](https://cupy.dev/) but also have a [numpy](https://numpy.org/)-based fallback option for testing on small datasets. In addition to a functional API, DEXP offers a command line interface that makes it very easy for non-coders to pipeline large processing jobs all the way from a large multi-terabyte raw dataset to fully processed and rendered video in MP4 format. 


## How to install **dexp**

### Prerequisites:

**dexp** works on OSX and Windows, but it is recomended to use the latest version of Ubuntu.
We recommend a machine with a top-of-the-line NVIDIA graphics card (min 12G to be confortable).

First, make sure to have a [working python installation](https://github.com/royerlab/dexp/wiki/install_python) 
Second, make sure to have a compatible and functional [CUDA installation](https://github.com/royerlab/dexp/wiki/install_cuda)

Once these prerequisites are satified, you can install **dexp**.

### Installation:

**dexp** can simply be installed with:

```
pip install dexp
```

Instakling this way will not install [CuPy](https://cupy.dev/), to install cupy you have to also specify
which CUDA version you have installed, for example the following installs **dexp** for CUDA 11.2:

```
pip install dexp[cuda112]
```
Other available CUDa versions (from [CuPy](https://cupy.dev/)) are: cuda111, cuda110, cuda102, cuda101, cuda100.


### How to use **dexp** ?

First you need a dataset aqquired on a light-sheet microscope, see [here](https://github.com/royerlab/dexp/wiki/dexp_datasets) for supported microscopes and formats.

Second, you can use any of the commands [here](https://github.com/royerlab/dexp/wiki/dexp_commands) to process your data.

### Example usage







  
 




  
  






