# RasterFairy
The purpose of Raster Fairy is to transform any kind of 2D point cloud into a regular raster whilst trying to preserve the neighborhood relations that were present in the original cloud. A typical use case is if you have a similarity clustering of images and want to show the images in a regular table structure.

![](http://i.imgur.com/HWOsmGC.gif)

Example Input Arrangment TSNE:
![](https://i.imgur.com/5um2xVS.jpeg)
Example Output Arrangment RasterFairy:
![](https://i.imgur.com/lhZkeWe.jpeg)


Why RasterFairy?
-----
Why not just use the Hungarian algorithm? Mapping a point cloud to a grid is a linear assignment problem, and solvers like the Hungarian or Jonker-Volgenant algorithm find the provably optimal assignment. The catch is cost: solving it exactly is roughly O(n³) with an n×n cost matrix, which means about 30 seconds and a lot of RAM for 4,096 points and practical infeasibility beyond ~10,000. RasterFairy's recursive slicing is a heuristic, but a good one — in benchmarks it stays within a few percent of the exact solution on neighborhood preservation while running about 200× faster, and it keeps working at sizes where exact solvers give up. Space-filling-curve tricks (sorting both cloud and grid along a Hilbert curve) are faster still but noticeably worse, since curve locality only preserves neighborhoods in one direction. If you don't have a 2D layout to preserve and just want images arranged by feature similarity, a different tool fits better: methods like FLAS (Barthel et al. 2023) sort high-dimensional vectors onto a grid directly, skipping the projection step entirely. RasterFairy is for the case where the 2D arrangement itself — a t-SNE or UMAP embedding, or any layout you care about — is the thing you want to keep.


Requirements
------------
* Python 3
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
Issues
-----
* Sometimes the subdivision algorithm fails in strange ways and creates a sub-optimal arrangment. In cases like that a pre-processing of the incoming xy coordinates via coonswarp.rectifyCloud is often able to fix it. Alternatively picking a different column/row arrangement can also help.

Recent Changes
-----
* Input point clouds are now normalized into the grid coordinate space internally, so inputs at any scale work correctly. Previously, small-range inputs (e.g. normalized coordinates in `[0,1]`) collapsed under integer quantization and produced garbage grids.
* Slicing now sorts on the true floating-point coordinates instead of prematurely rounding them to integers, improving neighborhood preservation.
* Grid arrangement computation is now O(√n) (a simple divisor scan) instead of factorial in the number of prime factors, which greatly speeds up large point counts (e.g. n = 4096).
* Removed the internal `rasterfairy.prime` module, which is no longer needed.

To-Do
-----
* Look into further improving splitting process


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

Citations
-----
If you are using this work in a scientific publication, cite as Mario Klingemann. 2015. RasterFairy. https://github.com/Quasimondo/RasterFairy
