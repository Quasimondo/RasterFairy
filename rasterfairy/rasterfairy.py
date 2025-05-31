#
# Raster Fairy v1.0.3,
# released 22.01.2016
#
# The purpose of Raster Fairy is to transform any kind of 2D point cloud into
# a regular raster whilst trying to preserve the neighborhood relations that
# were present in the original cloud. If you feel the name is a bit silly and
# you can also call it "RF-Transform".
#
# NOTICE: if you use this algorithm in an academic publication, paper or 
# research project please cite it either as "Raster Fairy by Mario Klingemann" 
# or "RF-Transform by Mario Klingemann"
#
#
# 
# Copyright (c) 2016, Mario Klingemann, mario@quasimondo.com
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Mario Klingemann nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL MARIO KLINGEMANN BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
This module implements the Raster Fairy algorithm.

The purpose of Raster Fairy is to transform any kind of 2D point cloud into
a regular raster whilst trying to preserve the neighborhood relations that
were present in the original cloud.
"""

import numpy as np
import rasterfairy.prime as prime
from functools import cmp_to_key
import math


def transformPointCloud2D( points2d, target = None, autoAdjustCount = True, proportionThreshold = 0.4):
    """Transforms a 2D point cloud to a regular raster grid.

    This is the main function of the Raster Fairy algorithm. It takes a list of
    2D points and maps them to a target grid, which can be a rectangle,
    a PIL image (acting as a mask), or a pre-defined raster mask dictionary.

    Args:
        points2d: A NumPy array of shape (N, 2) representing the input
            point cloud.
        target: The target grid. Can be:
            - None: Automatically determines a rectangular grid.
            - tuple (width, height): Defines a rectangular grid.
            - PIL.Image.Image: An image used as a mask. Black pixels are
              grid cells, white pixels are empty.
            - dict: A raster mask dictionary with keys 'width', 'height',
              'mask' (NumPy array), 'count', and 'hex' (boolean).
        autoAdjustCount: Boolean. If True, and the target is a raster mask,
            the mask will be adjusted (by flipping 0s to 1s or vice versa)
            to match the number of points in points2d if they differ.
        proportionThreshold: Float. When target is None, if the best
            rectangular arrangement has an aspect ratio below this threshold,
            an incomplete square grid is used instead.

    Returns:
        A tuple (gridPoints2d, (width, height)):
            - gridPoints2d: A NumPy array of shape (N, 2) representing the
              transformed grid points.
            - (width, height): A tuple representing the dimensions of the
              target grid.
        Returns False if the target grid is too small or has insufficient
        grid points.
    """
    pointCount = len(points2d)
    rasterMask = None

    if target is None:
        arrangements = getRectArrangements(pointCount)
        if not arrangements: # Handle empty arrangements list
             width = int(math.sqrt(pointCount))
             height = int(math.ceil(float(pointCount)/float(width)))
             print(("no good rectangle found for",pointCount,"points, using incomplete square",width,"*",height))
             target = {'width':width,'height':height,'mask':np.zeros((height,width),dtype=int), 'count':width*height, 'hex': False}
        else:
            target_rect = arrangements[0]
            if float(target_rect[0]) / float(target_rect[1])<proportionThreshold:
                width = int(math.sqrt(pointCount))
                height = int(math.ceil(float(pointCount)/float(width)))
                print(("no good rectangle found for",pointCount,"points, using incomplete square",width,"*",height))
                target = {'width':width,'height':height,'mask':np.zeros((height,width),dtype=int), 'count':width*height, 'hex': False}
            else:
                # Convert tuple to dict for consistency if it's a simple rectangle
                target = {'width': target_rect[0], 'height': target_rect[1], 'mask': np.zeros((target_rect[1], target_rect[0]), dtype=int), 'count': target_rect[0] * target_rect[1], 'hex': False}


    if type(target) is tuple and len(target)==2:
        #print("using rectangle target")
        if target[0] * target[1] < pointCount:
            print(("ERROR: target rectangle is too small to hold data: Rect is",target[0],"*",target[1],"=",target[0] * target[1]," vs ",pointCount," data points"))
            return False
        width = target[0]
        height = target[1]
        # Ensure rasterMask is initialized for rectangular targets too
        rasterMask = {'width': width, 'height': height, 'mask': np.zeros((height, width), dtype=int), 'count': width * height, 'hex': False}

    elif "PIL." in str(type(target)):
        #print("using bitmap image target")
        rasterMask = getRasterMaskFromImage(target)
        width = rasterMask['width']
        height = rasterMask['height']
    elif 'mask' in target and 'count' in target and 'width' in target and 'height' in target:
        #print("using raster mask target")
        rasterMask = target
        width = rasterMask['width']
        height = rasterMask['height']

    if not (rasterMask is None) and rasterMask['mask'].shape[0]*rasterMask['mask'].shape[1]-np.sum( rasterMask['mask'].flat) < len(points2d):
        print("ERROR: raster mask target does not have enough grid points to hold data")
        return False

    if not (rasterMask is None) and (rasterMask['count']!=len(points2d)):
        mask_flat = rasterMask['mask'].flatten() # Use a different variable name
        count = len(points2d)
        if autoAdjustCount is True:

            if count > rasterMask['count']:
                ones = np.nonzero(mask_flat)[0]
                np.random.shuffle(ones)
                mask_flat[ones[0:count-rasterMask['count']]] = 0
            elif count < rasterMask['count']:
                zeros = np.nonzero(1-mask_flat)[0]
                np.random.shuffle(zeros)
                mask_flat[zeros[0:rasterMask['count']-count]] = 1
        else:
            if count > rasterMask['count']:
                ones = np.nonzero(mask_flat)[0]
                mask_flat[ones[rasterMask['count']]-count:] = 0 # Corrected index
            elif count < rasterMask['count']:
                zeros = np.nonzero(1-mask_flat)[0]
                mask_flat[zeros[0:rasterMask['count']-count]] = 1 # Corrected index

        new_mask = mask_flat.reshape((rasterMask['height'], rasterMask['width']))
        rasterMask = {'width':rasterMask['width'],'height':rasterMask['height'],'mask':new_mask, 'count':count, 'hex': rasterMask['hex']}
    quadrants = [{'points':points2d, 'grid':[0,0,width,height], 'indices':np.arange(pointCount)}]
    i = 0
    failedSlices = 0
    while i < len(quadrants) and len(quadrants) < pointCount:
        if ( len(quadrants[i]['points']) > 1 ):
            slices = sliceQuadrant(quadrants[i], mask = rasterMask)
            if len(slices)>1:
                del quadrants[i]
                quadrants += slices
                i = 0
            else:
                failedSlices += 1
        else:
            i+=1
    if failedSlices>0:
        print("WARNING - There might be a problem with the data. Try using autoAdjustCount=True as a workaround or check if you have points with identical coordinates in your set.")

    gridPoints2d = points2d.copy()

    if not (rasterMask is None) and rasterMask['hex'] is True:
        f = math.sqrt(3.0)/2.0 
        offset = -0.5
        if np.argmin(rasterMask['mask'][0]) > np.argmin(rasterMask['mask'][1]):
            offset = 0.5
        for q in quadrants:
            if q['grid'][1]%2==0:
                q['grid'][0]-=offset
            q['grid'][1] *= f

    for q in quadrants:
        gridPoints2d[q['indices'][0]] = np.array(q['grid'][0:2],dtype=float)


    return gridPoints2d, (width, height)


def sliceQuadrant( quadrant, mask = None ):
    """Slices a quadrant of points into smaller sub-quadrants.

    This function is a core part of the recursive subdivision in the
    Raster Fairy algorithm. It splits a given quadrant either based on its
    grid dimensions (if no mask is provided) or based on the distribution
    of points within a raster mask.

    Args:
        quadrant: A dictionary representing the quadrant to slice.
            It should contain:
            - 'points': NumPy array of 2D points in this quadrant.
            - 'grid': List [x, y, width, height] defining the grid area.
            - 'indices': NumPy array of original indices of the points.
        mask: Optional. A raster mask dictionary (as used in
            `transformPointCloud2D`). If provided, the slicing will try to
            balance the number of points in sub-quadrants according to the
            mask.

    Returns:
        A list of dictionaries, where each dictionary represents a
        new sub-quadrant with the same structure as the input `quadrant`.
        Returns a list with a single element (the original quadrant) if
        no effective slice could be made.
    """
    xy = quadrant['points']
    grid = quadrant['grid']
    indices = quadrant['indices']
    slices = []
        
    if mask is None:
        
        if grid[2]>1:
            sliceXCount = 2
            while (grid[2]%sliceXCount!=0):
                sliceXCount+=1
        else:
            sliceXCount = grid[3]

        if grid[3]>1:
            sliceYCount = 2
            while (grid[3]%sliceYCount!=0):
                sliceYCount+=1
        else:
            sliceYCount = grid[2]
        
        splitX = (sliceXCount<sliceYCount or (sliceXCount==sliceYCount and grid[2]>grid[3]))
        if splitX:
            order = np.lexsort((xy[:,1].astype(int),xy[:,0].astype(int)))
            sliceCount = sliceXCount
            sliceSize  = grid[2] // sliceCount
            pointsPerSlice = grid[3] * sliceSize
            gridOffset = grid[0]
        else:
            order = np.lexsort((xy[:,0].astype(int),xy[:,1].astype(int)))    
            sliceCount = sliceYCount
            sliceSize = grid[3] // sliceCount
            pointsPerSlice = grid[2] * sliceSize
            gridOffset = grid[1]
        for i in range(sliceCount):
            sliceObject = {}
            # HOTFIX: indices must be integers!
            pointsPerSlice = int(pointsPerSlice)
            sliceObject['points'] = xy[order[i*pointsPerSlice:(i+1)*pointsPerSlice]]
            if len(sliceObject['points'])>0:
                sliceObject['indices'] = indices[order[i*pointsPerSlice:(i+1)*pointsPerSlice]]
                if splitX:
                    sliceObject['grid'] = [gridOffset,grid[1],sliceSize,grid[3]]
                    gridOffset += sliceObject['grid'][2]
                else:
                    sliceObject['grid'] = [grid[0],gridOffset,grid[2],sliceSize]
                    gridOffset += sliceObject['grid'][3]
                slices.append(sliceObject)  
            
    else:
        
        maskSlice = mask['mask'][grid[1]:grid[1]+grid[3],grid[0]:grid[0]+grid[2]]
        cols,rows = maskSlice.shape
        pointCountInMask = min(cols*rows - np.sum(maskSlice),len(indices))
        columnCounts = cols - np.sum(maskSlice, axis=0)
        splitColumn = countX  = 0
        while splitColumn < rows and countX < (pointCountInMask>>1):
            countX += columnCounts[splitColumn]
            splitColumn+=1
        
        rowCounts = rows-np.sum(maskSlice,axis=1)
        splitRow = countY = 0
        while splitRow < cols and countY < (pointCountInMask>>1):
            countY += rowCounts[splitRow]
            splitRow+=1
        
        order = np.lexsort((xy[:,1].astype(int),xy[:,0].astype(int)))
        slicesX = []
        if countX > 0:
            sliceObject = {} 
            newOrder = order[:countX]
            sliceObject['points'] = xy[newOrder]
            sliceObject['indices'] = indices[newOrder]
            sliceObject['grid'] = [grid[0], grid[1], splitColumn, grid[3]]
            cropGrid(mask['mask'],sliceObject['grid'])
            slicesX.append(sliceObject)    

        if countX < len(order):
            sliceObject = {} 
            newOrder = order[countX:]
            sliceObject['points'] = xy[newOrder]
            sliceObject['indices'] = indices[newOrder]
            sliceObject['grid'] = [grid[0]+splitColumn, grid[1], grid[2]-splitColumn, grid[3]]
            cropGrid(mask['mask'],sliceObject['grid'])
            slicesX.append(sliceObject)   
        
        order = np.lexsort((xy[:,0].astype(int),xy[:,1].astype(int)))
        slicesY = []
        if countY > 0:
            sliceObject = {} 
            newOrder = order[:countY]
            sliceObject['points'] = xy[newOrder]
            sliceObject['indices'] = indices[newOrder]
            sliceObject['grid'] = [grid[0], grid[1], grid[2],splitRow]
            cropGrid(mask['mask'],sliceObject['grid'])
            slicesY.append(sliceObject)  

        if countY < len(order):
            sliceObject = {} 
            newOrder = order[countY:]
            sliceObject['points'] = xy[newOrder]
            sliceObject['indices'] = indices[newOrder]
            sliceObject['grid'] = [grid[0], grid[1]+splitRow, grid[2], grid[3]-splitRow]
            cropGrid(mask['mask'],sliceObject['grid'])
            slicesY.append(sliceObject)   
        
        if len(slicesX)==1:
            slices = slicesY
        elif len(slicesY)==1:
            slices = slicesX
        else:
            prop1 = float(slicesX[0]['grid'][2]) / float(slicesX[0]['grid'][3])
            prop2 = float(slicesX[1]['grid'][2]) / float(slicesX[1]['grid'][3])
            if prop1 > 1.0:
                prop1 = 1.0 / prop1
            if prop2 > 1.0:
                prop2 = 1.0 / prop2 
            ratioX = max(abs(1.0 - prop1),abs(1.0 - prop2))
            
            prop1 = float(slicesY[0]['grid'][2]) / float(slicesY[0]['grid'][3])
            prop2 = float(slicesY[1]['grid'][2]) / float(slicesY[1]['grid'][3])
            if prop1 > 1.0:
                prop1 = 1.0 / prop1
            if prop2 > 1.0:
                prop2 = 1.0 / prop2 
            ratioY = max(abs(1.0 - prop1),abs(1.0 - prop2))
            if ratioX < ratioY:
                slices = slicesX
            else:
                slices = slicesY
             
    return slices

def cropGrid(mask,grid):
    """Adjusts a grid definition to tightly crop around non-masked areas.

    Given a larger mask and a grid (defined by [x, y, width, height])
    within that mask, this function modifies the grid in-place to remove
    empty rows and columns from its edges, based on the mask.

    Args:
        mask: A 2D NumPy array representing the larger mask, where 0 means
            an active cell and 1 means a masked-out cell.
        grid: A list [x, y, width, height] defining the subgrid to crop.
            This list is modified in-place.

    Returns:
        None. The `grid` list is modified directly.
    """
    maskSlice = mask[grid[1]:grid[1]+grid[3],grid[0]:grid[0]+grid[2]]
    cols,rows = maskSlice.shape
    columnCounts = cols - np.sum(maskSlice, axis=0)
    for i in range(len(columnCounts)):
        if columnCounts[i]>0:
            break
        grid[0]+=1
        grid[2]-=1
    
    for i in range(len(columnCounts)-1,-1,-1):
        if columnCounts[i]>0:
            break
        grid[2]-=1
    
    rowCounts = rows-np.sum(maskSlice,axis=1)
    for i in range(len(rowCounts)):
        if rowCounts[i]>0:
            break
        grid[1]+=1
        grid[3]-=1
    
    for i in range(len(rowCounts)-1,-1,-1):
        if rowCounts[i]>0:
            break
        grid[3]-=1
    

def getRasterMaskFromImage( img ):
    """Converts a PIL image to a raster mask dictionary.

    The image is converted to monochrome. Black pixels in the image become
    active grid cells (value 0) in the mask, and white pixels become
    masked-out cells (value 1).

    Args:
        img: A PIL.Image.Image object.

    Returns:
        A raster mask dictionary with keys:
            - 'width': Width of the image.
            - 'height': Height of the image.
            - 'mask': A 2D NumPy array (height, width) of integers (0 or 1).
            - 'count': The number of active (0) cells in the mask.
            - 'hex': Boolean, False for image-derived masks.
    """
    img = img.convert('1')
    mask_data = np.array(img.getdata())&1 # Use a different variable name
    return {'width':img.width,'height':img.height,'mask':mask_data.reshape((img.height, img.width)), 'count':len(mask_data)- np.sum(mask_data), 'hex':False}


def getCircleRasterMask( r, innerRingRadius = 0, rasterCount = None, autoAdjustCount = True  ):
    """Creates a circular or ring-shaped raster mask.

    Args:
        r: Integer, the outer radius of the circle.
        innerRingRadius: Integer, optional. If > 0, creates a ring mask
            with this inner radius.
        rasterCount: Integer, optional. If provided and `autoAdjustCount`
            is True, the mask will be adjusted to have exactly this many
            active cells.
        autoAdjustCount: Boolean. If True and `rasterCount` is provided,
            adjusts the mask to match `rasterCount`.

    Returns:
        A raster mask dictionary, similar to `getRasterMaskFromImage`,
        representing the circular or ring mask. 'hex' is False.
    """
    d = r*2+1
    p = np.ones((d,d),dtype=int)
    IG = (r<<1) - 3
    IDGR = -6
    IDGD = (r<<2) - 10
    
    IX = 0
    IY = r

    while IY >= IX: 
        p[r-IX:r+IX+1,r+IY] = p[r-IX:r+IX+1,r-IY] = p[r-IY:r+IY+1,r+IX] = p[r-IY:r+IY+1,r-IX] = 0
        if IG<0: 
            IG = IG+IDGD 
            IDGD -= 8 
            IY-=1 
        else: 
            IG += IDGR 
            IDGD -=4 
        IDGR -= 4 
        IX+=1 
    
    if innerRingRadius > 0 and innerRingRadius < r:
        r2 = r
        r = innerRingRadius
        IG = (r<<1) - 3
        IDGR = -6
        IDGD = (r<<2) - 10

        IX = 0
        IY = r

        while IY >= IX: 
            p[r2-IX:r2+IX+1,r2+IY] = p[r2-IX:r2+IX+1,r2-IY] = p[r2-IY:r2+IY+1,r2+IX] = p[r2-IY:r2+IY+1,r2-IX] = 1
            if IG<0: 
                IG = IG+IDGD 
                IDGD -= 8 
                IY-=1 
            else: 
                IG += IDGR 
                IDGD -=4 
            IDGR -= 4 
            IX+=1 
    
    
    count = p.shape[0] * p.shape[1] - np.sum(p.flat)
    if not rasterCount is None and autoAdjustCount and count != rasterCount: 
        p = p.flatten()&1
        if count < rasterCount:
            ones = np.nonzero(p)[0]
            np.random.shuffle(ones)
            p[ones[0:rasterCount-count]] = 0
            count = rasterCount
        elif count > rasterCount:
            zeros = np.nonzero(1-p)[0]
            np.random.shuffle(zeros)
            p[zeros[0:count-rasterCount]] = 1
            count = rasterCount        
        p = p.reshape((d,d))
    #print("adjusted count",p.shape[0] * p.shape[1]- np.sum(p))
    return {'width':d,'height':d,'mask':p, 'count':count}
        
def getRectArrangements(n):
    """Generates possible rectangular arrangements (width, height) for n items.

    It finds prime factors of n and then generates all possible ways to
    multiply these factors to form two numbers (width and height).
    The arrangements are sorted by their aspect ratio (closer to square is better).

    Args:
        n: An integer representing the total number of items.

    Returns:
        A list of tuples (width, height), sorted by how close their
        aspect ratio is to 1 (square). Returns an empty list if n <= 0.
    """
    if n <= 0:
        return []
    p_instance = prime.Prime() # Use a different variable name
    f = p_instance.getPrimeFactors(n)
    f_count = len(f)
    ma = multiplyArray(f)
    arrangements = set([(1,ma)])

    if (f_count > 1):
        perms = set(p_instance.getPermutations(f)) # Use the new variable name
        for perm_val in perms: # Use a different variable name
            for i in range(1,f_count):
                v1 = multiplyArray(perm_val[0:i])
                v2 = multiplyArray(perm_val[i:])
                arrangements.add((min(v1, v2),max(v1, v2)))


    return sorted(list(arrangements), key=cmp_to_key(proportion_sort), reverse=True)

def getShiftedAlternatingRectArrangements(n):
    """Generates arrangements for hexagonal grids with alternating row lengths (shifted).

    These arrangements are suitable for hexagonal grids where rows are
    offset, creating a more compact packing.

    Args:
        n: The total number of items to arrange.

    Returns:
        A list of dictionaries, where each dictionary represents an
        arrangement with keys:
            - 'hex': True (indicating hexagonal type).
            - 'rows': A list of integers, row lengths.
            - 'type': 'alternating'.
    """
    arrangements = set([])
    for x in range(1,n >> 1):
        v = 2 * x + 1
        if n % v == x:
            arrangements.add((x, x + 1, ((n // v) | 0) * 2 + 1))
        
    for x in range(2,1 + (n >> 1)):
        v = 2 * x - 1
        if n % v == x:
            arrangements.add((x, x - 1, ((n // v) | 0) * 2 + 1))
    
    result = []
    for a in arrangements:
        nn = n
        d = []
        i = 0
        while nn > 0:
            d.append(a[i])
            nn-=a[i]
            i = 1 - i
        result.append({'hex':True,'rows':d,'type':'alternating'})
    return result


def getShiftedSymmetricArrangements(n):
    """Generates symmetric arrangements for shifted hexagonal grids.

    Args:
        n: The total number of items to arrange.

    Returns:
        A list of dictionaries, each representing a symmetric hexagonal
        arrangement with keys 'hex' (True), 'rows', and 'type' ('symmetric').
    """
    hits = []
    for i in range(1, n >> 1):
        d_list = [] # Use a different variable name
        count = n
        row_val = i # Use a different variable name
        d_list.append(row_val)
        while True:
            count -= row_val * 2
            if count == row_val + 1:
                if i != row_val:
                    d_list.append(row_val+1)
                    d_list+=d_list[0:-1][::-1]
                    hits.append({'hex':True,'rows':d_list,'type':'symmetric'})
                    break
            elif count <= 0:
                break
            row_val+=1
            d_list.append(row_val)
    return hits

def getShiftedTriangularArrangement(n):
    """Generates a triangular arrangement for shifted hexagonal grids.

    This corresponds to "triangular numbers" for hexagonal packing.

    Args:
        n: The total number of items. Must be a triangular number for
           hexagonal grids (n = k*(k+1)/2 where rows increase by 1).

    Returns:
        A list containing a single arrangement dictionary if n is a
        hexagonal triangular number (keys: 'hex': True, 'rows', 'type':
        'triangular'), or an empty list otherwise.
    """
    t_val = math.sqrt(8 * n + 1) # Use a different variable name
    if t_val != math.floor(t_val):
        return []

    arrangement_list = [] # Use a different variable name
    i = 1
    current_n = n # Use a different variable name
    while current_n>0:
        arrangement_list.append(i)
        current_n-=i
        i+=1

    return [{'hex':True,'rows':arrangement_list,'type':'triangular'}]

def getAlternatingRectArrangements(n):
    """Generates arrangements for rectangular grids with alternating row lengths.

    These are for standard rectangular grids, not shifted hexagonal ones.

    Args:
        n: The total number of items to arrange.

    Returns:
        A list of dictionaries, each representing an arrangement with keys:
            - 'hex': False.
            - 'rows': A list of integers, row lengths.
            - 'type': 'alternating'.
    """
    arrangements = set([])
    for x in range(1,n >> 1):
        v = 2 * x + 2
        if n % v == x:
            arrangements.add((x, x + 2, ((n // v) | 0) * 2 + 1))
        
    for x in range(2,1 + (n >> 1)):
        v = 2 * x - 2
        if n % v == x:
            arrangements.add((x, x -2, ((n // v) | 0) * 2 + 1))
    
    result = []
    for a in arrangements:
        nn = n
        d = []
        i = 0
        while nn > 0:
            d.append(a[i])
            nn-=a[i]
            i = 1 - i
        result.append({'hex':False,'rows':d,'type':'alternating'})
    return result

def getSymmetricArrangements(n):
    """Generates symmetric arrangements for standard rectangular grids.

    Args:
        n: The total number of items to arrange.

    Returns:
        A list of dictionaries, each representing a symmetric rectangular
        arrangement with keys 'hex' (False), 'rows', and 'type' ('symmetric').
    """
    hits = []
    for i in range(1, n >> 1):
        d_list = [] # Use a different variable name
        count = n
        row_val = i # Use a different variable name
        d_list.append(row_val)
        while True:
            count -= row_val * 2
            if count == row_val + 2:
                if i != row_val:
                    d_list.append(row_val+2)
                    d_list+=d_list[0:-1][::-1]
                    hits.append({'hex':False,'rows':d_list,'type':'symmetric'})
                    break
            elif count <= 0:
                break
            row_val+=2
            d_list.append(row_val)
    return hits

def getTriangularArrangement(n):
    """Generates a triangular arrangement for standard rectangular grids.

    This corresponds to "square triangular numbers" where rows increase by 2
    (1, 3, 5, ...).

    Args:
        n: The total number of items. Must be a perfect square.

    Returns:
        A list containing a single arrangement dictionary if n is a
        perfect square (keys: 'hex': False, 'rows', 'type': 'triangular'),
        or an empty list otherwise.
    """
    t_val = math.sqrt(n) # Use a different variable name
    if t_val != math.floor(t_val):
        return []

    arrangement_list = [] # Use a different variable name
    i = 1
    current_n = n # Use a different variable name
    while current_n>0:
        arrangement_list.append(i)
        current_n-=i
        i+=2

    return [{'hex':False,'rows':arrangement_list,'type':'triangular'}]


def getArrangements(n, includeHexagonalArrangements = True,includeRectangularArrangements = True):
    """Gets a list of possible arrangements for n items.

    This function combines results from various arrangement generation
    functions (rectangular, hexagonal, symmetric, triangular, circular).

    Args:
        n: The total number of items to arrange.
        includeHexagonalArrangements: Boolean, whether to include hexagonal types.
        includeRectangularArrangements: Boolean, whether to include rectangular types.

    Returns:
        A list of arrangement dictionaries.
    """
    res = [] # Use a different variable name
    if includeHexagonalArrangements:
        res += getShiftedAlternatingRectArrangements(n)
        res += getShiftedSymmetricArrangements(n)
        res += getShiftedTriangularArrangement(n)
    if includeRectangularArrangements:
        res += getAlternatingRectArrangements(n)
        res += getSymmetricArrangements(n)
        res += getTriangularArrangement(n)
        bestr,bestrp,bestc = getBestCircularMatch(n)
        if bestc == n:
            res.append( getCircularArrangement(bestr,bestrp))
    return res

def arrangementListToRasterMasks( arrangements ):
    """Converts a list of arrangement dictionaries to raster mask dictionaries.

    Args:
        arrangements: A list of arrangement dictionaries (as returned by
            functions like `getArrangements`).

    Returns:
        A list of raster mask dictionaries, sorted by their aspect ratio
        (closer to square is better).
    """
    masks = []
    for i in range(len(arrangements)):
        masks.append(arrangementToRasterMask(arrangements[i]))

    return sorted(masks, key=cmp_to_key(arrangement_sort), reverse=True)


def arrangementToRasterMask( arrangement ):
    """Converts a single arrangement dictionary to a raster mask dictionary.

    Args:
        arrangement: An arrangement dictionary (e.g., from `getArrangements`).
            It must contain 'rows' (list of row lengths) and 'hex' (boolean).
            Optionally 'type'.

    Returns:
        A raster mask dictionary with keys 'width', 'height', 'mask',
        'count', 'hex', and 'type'.
    """
    rows_arr = np.array(arrangement['rows']) # Use a different variable name
    width = np.max(rows_arr)
    if arrangement['hex'] is True:
        width+=1
    height = len(rows_arr)
    current_mask = np.ones((height,width),dtype=int) # Use a different variable name
    for row_idx in range(len(rows_arr)): # Use a different variable name
        c_val = rows_arr[row_idx] # Use a different variable name
        current_mask[row_idx,(width-c_val)>>1:((width-c_val)>>1)+c_val] = 0

    return {'width':width,'height':height,'mask':current_mask, 'count':np.sum(rows_arr),'hex':arrangement['hex'],'type':arrangement.get('type', 'unknown')}

def rasterMaskToGrid( rasterMask ):
    """Converts a raster mask dictionary to a list of 2D grid point coordinates.

    Args:
        rasterMask: A raster mask dictionary (e.g., from
            `arrangementToRasterMask`). It must contain 'mask' (2D NumPy
            array), 'height', 'width', and 'hex' (boolean).

    Returns:
        A NumPy array of shape (N, 2) where N is the number of active
        cells in the mask. Each row is an [x, y] coordinate.
        Coordinates are adjusted for hexagonal grids if `rasterMask['hex']`
        is True.
    """
    grid_points = [] # Use a different variable name
    mask_arr = rasterMask['mask'] # Use a different variable name
    for y_coord in range(rasterMask['height']): # Use a different variable name
        for x_coord in range(rasterMask['width']): # Use a different variable name
            if mask_arr[y_coord,x_coord]==0:
                grid_points.append([x_coord,y_coord])

    grid_points_arr = np.array(grid_points,dtype=float) # Use a different variable name

    if not (rasterMask is None) and rasterMask['hex'] is True:
        f_val = math.sqrt(3.0)/2.0 # Use a different variable name
        offset_val = -0.5 # Use a different variable name
        if np.argmin(rasterMask['mask'][0]) > np.argmin(rasterMask['mask'][1]):
            offset_val = 0.5
        for i_idx in range(len(grid_points_arr)): # Use a different variable name
            if (grid_points_arr[i_idx][1]%2.0==0.0):
                grid_points_arr[i_idx][0]-=offset_val
            grid_points_arr[i_idx][1] *= f_val
    return grid_points_arr


def getBestCircularMatch(n):
    """Finds the best circular arrangement parameters for n items.

    It iterates through possible radii and adjustment factors to find
    a circle that contains a number of grid points closest to n,
    prioritizing exact matches or the smallest count greater than n.

    Args:
        n: The target number of items.

    Returns:
        A tuple (best_radius, best_radius_adjust_factor, best_count):
            - best_r: Optimal integer radius.
            - best_rp: Optimal radius adjustment factor (0.0 to 0.9).
            - best_c: The number of points in the best circle found.
    """
    bestc = n*2
    bestr = 0
    bestrp = 0.0

    minr = int(math.sqrt(n / math.pi)) if n > 0 else 1 # ensure minr is at least 1
    if minr == 0: minr = 1 # ensure minr is at least 1

    for rp_val in range(0,10): # Use a different variable name
        rpf = float(rp_val)/10.0
        for r_val in range(minr,minr+3): # Use a different variable name
            if r_val == 0: continue # Radius cannot be zero
            rlim = (r_val+rpf)*(r_val+rpf)
            c_count = 0 # Use a different variable name
            for y_coord in range(-r_val,r_val+1): # Use a different variable name
                yy = y_coord*y_coord
                for x_coord in range(-r_val,r_val+1): # Use a different variable name
                    if x_coord*x_coord+yy<rlim:
                        c_count+=1
            if c_count == n:
                return r_val,rpf,c_count

            if c_count>n and c_count < bestc:
                bestrp = rpf
                bestr = r_val
                bestc = c_count
    return bestr,bestrp,bestc

def getCircularArrangement(radius,adjustFactor):
    """Creates an arrangement dictionary for a circular layout.

    Args:
        radius: Integer, the main radius of the circle.
        adjustFactor: Float, a factor to fine-tune the circle's boundary.

    Returns:
        An arrangement dictionary with keys:
            - 'hex': False.
            - 'rows': A list of integers representing row lengths for the circle.
            - 'type': 'circular'.
    """
    rows_list = np.zeros( radius*2+1,dtype=int ) # Use a different variable name

    rlim = (radius+adjustFactor)*(radius+adjustFactor)
    for y_coord in range(-radius,radius+1): # Use a different variable name
        yy = y_coord*y_coord
        for x_coord in range(-radius,radius+1): # Use a different variable name
            if x_coord*x_coord+yy<rlim:
                rows_list[radius+y_coord]+=1

    return {'hex':False,'rows':rows_list,'type':'circular'}

def arrangement_sort(item):
    """Sort key for raster mask arrangements based on aspect ratio.

    Used to sort a list of raster mask dictionaries (output by
    `arrangementToRasterMask` or `arrangementListToRasterMasks`)
    so that masks closer to a square shape appear first.

    Args:
        item: A raster mask dictionary with 'width' and 'height' keys.

    Returns:
        An integer score proportional to the aspect ratio (closer to 1 is higher).
    """
    # Ensure width and height are not zero to avoid DivisionByZeroError
    width = item['width'] if item['width'] > 0 else 1
    height = item['height'] if item['height'] > 0 else 1
    return int(100000000*(abs(float(min(width,height)) / float(max(width,height)))))

def proportion_sort(item):
    """Sort key for (width, height) tuples based on aspect ratio.

    Used to sort a list of (width, height) tuples (output by
    `getRectArrangements`) so that pairs closer to a square shape
    appear first.

    Args:
        item: A tuple (width, height).

    Returns:
        An integer score proportional to the aspect ratio (closer to 1 is higher).
    """
    # Ensure width and height are not zero to avoid DivisionByZeroError
    val1 = item[0] if item[0] > 0 else 1
    val2 = item[1] if item[1] > 0 else 1
    return int(100000000*(abs(float(min(val1,val2)) / float(max(val1,val2)))))

def multiplyArray(arr):
    """Calculates the product of all numbers in a list.

    Args:
        arr: A list of numbers.

    Returns:
        The product of the numbers in the list. Returns 1 if the list is empty.
    """
    res = 1 # Use a different variable name
    for val in arr: # Use a different variable name
        res *= val
    return res
