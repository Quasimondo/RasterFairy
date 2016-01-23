# RasterFairy
The purpose of Raster Fairy is to transform any kind of 2D point cloud into a regular raster whilst trying to preserve the neighborhood relations that were present in the original cloud. A typical use case is if you have a similarity clustering of images and want to show the images in a regular table structure.

![](http://i.imgur.com/HWOsmGC.gif)


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

To-Do
-----
* Add hexagonal circle grid
* Add optional warp class
* Add random neighbor swapping post-processing
* Look into further improving splitting process


A note about porting this to other languages
-----

If you want to port this algorithm to another language like C++, Javascript or COBOL I'm very happy about it.
Only there is a little thing about "porting etiquette" I want to mention - yes, it will take you some
work to translate those 500+ lines of code into the language of your choice and you might have to change
a few things to make it work. Nevertheless, the algorithm stays the same and yes - I'm probably quite
vain here - but I like to read my name. In big letters. Bigger than yours. And don't even think about
writing anything like "insipired by". So the proper titling for a port will read something like
"Raster Fairy by Mario Klingemann, C++ port by YOU".  
