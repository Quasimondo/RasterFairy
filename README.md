# RasterFairy
The purpose of Raster Fairy is to transform any kind of 2D point cloud into a regular raster whilst trying to preserve the neighborhood relations that were present in the original cloud. A typical use case is if you have a similarity clustering of images and want to show the images in a regular table structure.

![](http://i.imgur.com/HWOsmGC.gif)


Requirements
------------
* Python 2.7 (Python 3 is not supported yet)
* [numpy](numpy.scipy.org) > =1.7.1
* [scipy](www.scipy.org) - only for coonswarp and rfoptimizer

Installation
------------

From the root directory, run:
```
pip install .
```

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
* Look into further improving splitting process
* Add Python 3 support


Related Projects
-----
As I learned after publishing there is a very similar earlier developed technique called IsoMatch
by O. Fried, S. DiVerdi, M. Halber, E. Sizikova and A. Finkelstein. 
Unfortunately I was not aware of it during my research and their solution works differently, but
you might want to check it out and see if it's better suited to your requirements:
[IsoMatch](http://gfx.cs.princeton.edu/pubs/Fried_2015_ICI/index.php)
[Codebase](https://github.com/ohadf/isomatch)

Another related technique is Kernelized Sorting by Novi Quadrianto, Le Song, Alex J. Smola. from 2009 
[Kernelized Sorting](http://users.sussex.ac.uk/~nq28/kernelized_sorting.html)

Kyle McDonald's [CloudToGrid](https://github.com/kylemcdonald/CloudToGrid) project is a Python-based implementation of the Hungarian method.

A note about porting this to other languages
-----

If you want to port this algorithm to another language like C++, Javascript or COBOL I'm very happy about it.
Only there is a little thing about "porting etiquette" I want to mention - yes, it will take you some
work to translate those 500+ lines of code into the language of your choice and you might have to change
a few things to make it work. Nevertheless, the algorithm stays the same and yes - I'm probably quite
vain here - but I like to read my name. In big letters. Bigger than yours. And don't even think about
writing anything like "insipired by". So the proper titling for a port will read something like
"Raster Fairy by Mario Klingemann, C++ port by YOU".  
