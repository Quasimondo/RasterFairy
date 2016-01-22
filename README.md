# RasterFairy
The purpose of Raster Fairy is to transform any kind of 2D point cloud into a regular raster whilst trying to preserve the neighborhood relations that were present in the original cloud. A typical use case is if you have a similarity clustering of images and want to show the images in a regular table structure.

Requirements
------------

* [numpy](numpy.scipy.org) > =1.7.1

Installation
------------

Version 1.0 is not at proper Python package yet, so just copy the files
```
rasterfairy.py
prime.py 
```
into your application folder to use it


Usage
-----

Basic usage:

```
import rasterfairy

#xy should be a numpy array with a shape (number of points,2) 
grid_xy = rasterfairy.transformPointCloud2D(xy)
#grid_xy will contain the points in the same order but aligned to a grid
```

