#
# Raster Fairy v1.1.0,
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
from scipy.optimize import linear_sum_assignment
import math


def transformPointCloud2D(points2d, target=None, autoAdjustCount=True, 
                         proportionThreshold=0.4, hungarian_threshold=200,
                         emptyStrategy='random'):
    """Transforms a 2D point cloud to a regular raster grid.

    Args:
        points2d: A NumPy array of shape (N, 2) representing the input point cloud.
        target: The target grid (None, tuple, PIL image, or dict).
        autoAdjustCount: Boolean. If True, adjusts mask to match point count.
        proportionThreshold: Float. Aspect ratio threshold for rectangular grids.
        hungarian_threshold: Maximum points in sub-quadrant for applying Hungarian algorithm.
        emptyStrategy: String. Strategy for which cells to leave empty when
            there are more grid cells than points. Options:
            - 'random': Randomly distribute empty cells (default)
            - 'center': Keep center filled, empty outer cells
            - 'outer': Keep outer filled, empty center cells  
            - 'bottom': Empty cells from bottom rows
            - 'top': Empty cells from top rows
            - 'edges': Empty cells from the edges inward

    Returns:
        A tuple (gridPoints2d, (width, height)) or False if target too small.
    """
    
    if points2d is None or len(points2d) == 0:
        raise ValueError("points2d cannot be empty")
    
    if not isinstance(points2d, np.ndarray):
        points2d = np.array(points2d)
    
    if points2d.shape[1] != 2:
        raise ValueError("points2d must have shape (N, 2)")
        
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

    if not (rasterMask is None) and (rasterMask['count'] != len(points2d)):
        mask_flat = rasterMask['mask'].flatten()
        count = len(points2d)
        
        if autoAdjustCount is True:
            if count > rasterMask['count']:
                ones = np.nonzero(mask_flat)[0]
                np.random.shuffle(ones)
                mask_flat[ones[0:count-rasterMask['count']]] = 0
            elif count < rasterMask['count']:
                zeros = np.nonzero(1-mask_flat)[0]
                cells_to_mask = rasterMask['count'] - count
                
                # Apply empty strategy
                if emptyStrategy == 'random':
                    np.random.shuffle(zeros)
                    mask_flat[zeros[0:cells_to_mask]] = 1
                elif emptyStrategy == 'center':
                    # Mask cells closest to center
                    center_y, center_x = rasterMask['height'] / 2, rasterMask['width'] / 2
                    distances = []
                    for idx in zeros:
                        y, x = divmod(idx, rasterMask['width'])
                        dist = (x - center_x)**2 + (y - center_y)**2
                        distances.append((dist, idx))
                    distances.sort()  # Closest to center first
                    for i in range(cells_to_mask):
                        mask_flat[distances[i][1]] = 1
                elif emptyStrategy == 'outer':
                    # Mask cells farthest from center
                    center_y, center_x = rasterMask['height'] / 2, rasterMask['width'] / 2
                    distances = []
                    for idx in zeros:
                        y, x = divmod(idx, rasterMask['width'])
                        dist = (x - center_x)**2 + (y - center_y)**2
                        distances.append((dist, idx))
                    distances.sort(reverse=True)  # Farthest from center first
                    for i in range(cells_to_mask):
                        mask_flat[distances[i][1]] = 1
                elif emptyStrategy == 'bottom':
                    # Mask cells from bottom rows first
                    rows_to_mask = []
                    for idx in zeros:
                        y, x = divmod(idx, rasterMask['width'])
                        rows_to_mask.append((y, idx))  # Sort by row (y coordinate)
                    rows_to_mask.sort(reverse=True)  # Bottom rows first
                    for i in range(cells_to_mask):
                        mask_flat[rows_to_mask[i][1]] = 1
                elif emptyStrategy == 'top':
                    # Mask cells from top rows first
                    rows_to_mask = []
                    for idx in zeros:
                        y, x = divmod(idx, rasterMask['width'])
                        rows_to_mask.append((y, idx))
                    rows_to_mask.sort()  # Top rows first
                    for i in range(cells_to_mask):
                        mask_flat[rows_to_mask[i][1]] = 1
                elif emptyStrategy == 'edges':
                    # Mask cells from edges inward
                    edge_distances = []
                    for idx in zeros:
                        y, x = divmod(idx, rasterMask['width'])
                        # Distance to nearest edge
                        dist_to_edge = min(x, rasterMask['width']-1-x, y, rasterMask['height']-1-y)
                        edge_distances.append((dist_to_edge, idx))
                    edge_distances.sort()  # Closest to edge first
                    for i in range(cells_to_mask):
                        mask_flat[edge_distances[i][1]] = 1
                else:
                    # Default to random if unknown strategy
                    np.random.shuffle(zeros)
                    mask_flat[zeros[0:cells_to_mask]] = 1

            new_mask = mask_flat.reshape((rasterMask['height'], rasterMask['width']))
            rasterMask = {'width': rasterMask['width'], 'height': rasterMask['height'], 
                         'mask': new_mask, 'count': count, 'hex': rasterMask['hex']}
            
            
        
    quadrants = [{'points':points2d, 'grid':[0,0,width,height], 'indices':np.arange(pointCount)}]
    i = 0
    failedSlices = 0
    while i < len(quadrants) and len(quadrants) < pointCount:
        if ( len(quadrants[i]['points']) > 1 ):
            slices = sliceQuadrant(quadrants[i], mask=rasterMask, hungarian_threshold=hungarian_threshold)
            if len(slices) > 1:
                del quadrants[i]
                quadrants += slices
                # Don't reset i to 0, continue from current position
                i = min(i, len(quadrants) - 1)
            else:
                failedSlices += 1
                i += 1
        else:
            i += 1
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
        if len(q['indices']) > 0:
            # If quadrant has multiple points, assign the grid position to all of them
            for idx in q['indices']:
                gridPoints2d[idx] = np.array(q['grid'][0:2], dtype=float)


    return gridPoints2d, (width, height)


def sliceQuadrant(quadrant, mask=None, hungarian_threshold=50):
    """Slices a quadrant of points into smaller sub-quadrants.
    
    Uses Hungarian algorithm for optimal assignment when quadrant is small enough.

    Args:
        quadrant: A dictionary representing the quadrant to slice.
        mask: Optional raster mask dictionary.
        hungarian_threshold: Maximum number of points to use Hungarian algorithm on.

    Returns:
        A list of dictionaries representing new sub-quadrants.
    """
    xy = quadrant['points']
    grid = quadrant['grid']
    indices = quadrant['indices']
    
    # If the quadrant is small enough, use Hungarian algorithm
    if len(xy) <= hungarian_threshold and len(xy) > 1:
        return hungarianAssignment(quadrant, mask)
    
    # Otherwise, use the original recursive slicing method
    slices = []
    
    # ... rest of the original sliceQuadrant code ...
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
            xy_int = xy.astype(int)
            order = np.lexsort((xy_int[:, 1], xy_int[:, 0]))
            sliceCount = sliceXCount
            sliceSize  = grid[2] // sliceCount
            pointsPerSlice = int(grid[3] * sliceSize)
            gridOffset = grid[0]
        else:
            xy_int = xy.astype(int)
            order = np.lexsort((xy_int[:, 0], xy_int[:, 1]))
            sliceCount = sliceYCount
            sliceSize = grid[3] // sliceCount
            pointsPerSlice = int(grid[2] * sliceSize)
            gridOffset = grid[1]
        for i in range(sliceCount):
            sliceObject = {}
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
        # ... rest of the original mask-based slicing code ...
        maskSlice = mask['mask'][grid[1]:grid[1]+grid[3],grid[0]:grid[0]+grid[2]]
        rows, cols = maskSlice.shape
        
        pointCountInMask = min(rows*cols - np.sum(maskSlice),len(indices))
        
        if pointCountInMask <= 0:
            return [quadrant]
            
        columnCounts = rows - np.sum(maskSlice, axis=0)
        splitColumn = countX = 0
        while splitColumn < cols and countX < (pointCountInMask>>1):
            countX += columnCounts[splitColumn]
            splitColumn+=1
        
        rowCounts = cols - np.sum(maskSlice,axis=1)
        splitRow = countY = 0
        while splitRow < rows and countY < (pointCountInMask>>1):
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
            if sliceObject['grid'][2] > 0 and sliceObject['grid'][3] > 0:
                slicesX.append(sliceObject)    

        if countX < len(order):
            sliceObject = {} 
            newOrder = order[countX:]
            sliceObject['points'] = xy[newOrder]
            sliceObject['indices'] = indices[newOrder]
            sliceObject['grid'] = [grid[0]+splitColumn, grid[1], grid[2]-splitColumn, grid[3]]
            cropGrid(mask['mask'],sliceObject['grid'])
            if sliceObject['grid'][2] > 0 and sliceObject['grid'][3] > 0:
                slicesX.append(sliceObject)   
        
        order = np.lexsort((xy[:,0].astype(int),xy[:,1].astype(int)))
        slicesY = []
        if countY > 0:
            sliceObject = {} 
            newOrder = order[:countY]
            sliceObject['points'] = xy[newOrder]
            sliceObject['indices'] = indices[newOrder]
            sliceObject['grid'] = [grid[0], grid[1], grid[2], splitRow]
            cropGrid(mask['mask'],sliceObject['grid'])
            if sliceObject['grid'][2] > 0 and sliceObject['grid'][3] > 0:
                slicesY.append(sliceObject)  

        if countY < len(order):
            sliceObject = {} 
            newOrder = order[countY:]
            sliceObject['points'] = xy[newOrder]
            sliceObject['indices'] = indices[newOrder]
            sliceObject['grid'] = [grid[0], grid[1]+splitRow, grid[2], grid[3]-splitRow]
            cropGrid(mask['mask'],sliceObject['grid'])
            if sliceObject['grid'][2] > 0 and sliceObject['grid'][3] > 0:
                slicesY.append(sliceObject)   
        
        def safe_ratio(grid_info):
            width = max(grid_info['grid'][2], 1)
            height = max(grid_info['grid'][3], 1)
            ratio = min(width, height) / max(width, height)
            return ratio if ratio > 0 else 0.01
        
        if len(slicesX) == 0 and len(slicesY) == 0:
            slices = [quadrant]
        elif len(slicesX) <= 1:
            slices = slicesY if len(slicesY) > 0 else [quadrant]
        elif len(slicesY) <= 1:
            slices = slicesX if len(slicesX) > 0 else [quadrant]
        else:
            ratio1 = safe_ratio(slicesX[0])
            ratio2 = safe_ratio(slicesX[1]) if len(slicesX) > 1 else ratio1
            ratioX = max(abs(1.0 - ratio1), abs(1.0 - ratio2))
            
            ratio1 = safe_ratio(slicesY[0])
            ratio2 = safe_ratio(slicesY[1]) if len(slicesY) > 1 else ratio1
            ratioY = max(abs(1.0 - ratio1), abs(1.0 - ratio2))
            
            if ratioX < ratioY:
                slices = slicesX
            else:
                slices = slicesY
             
    return slices

def hungarianAssignment(quadrant, mask=None):
    """Uses Hungarian algorithm for optimal point-to-grid assignment.
    
    Args:
        quadrant: Dictionary with 'points', 'grid', and 'indices'.
        mask: Optional raster mask dictionary.
        
    Returns:
        List of individual quadrants with optimal assignments.
    """
    
    xy = quadrant['points']
    grid = quadrant['grid']
    indices = quadrant['indices']
    print(f"applying hungarian assignment for {len(indices)} indices")
    # Generate all valid grid positions
    grid_positions = generateGridPositions(grid, mask)
    
    # If we have more grid positions than points, select the best subset
    if len(grid_positions) > len(xy):
        grid_positions = selectBestGridPositions(xy, grid_positions, len(xy))
    elif len(grid_positions) < len(xy):
        # This shouldn't happen with proper grid sizing, but handle gracefully
        print(f"Warning: Not enough grid positions ({len(grid_positions)}) for points ({len(xy)})")
        return [quadrant]  # Fall back to original quadrant
    
    # Compute cost matrix (Euclidean distances)
    cost_matrix = computeCostMatrix(xy, grid_positions)
    
    # Solve assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Create individual quadrants for each point
    result_quadrants = []
    for i, (point_idx, grid_idx) in enumerate(zip(row_indices, col_indices)):
        result_quadrants.append({
            'points': xy[point_idx:point_idx+1],  # Single point
            'grid': grid_positions[grid_idx] + [1, 1],  # [x, y, width=1, height=1]
            'indices': indices[point_idx:point_idx+1]  # Single index
        })
    
    return result_quadrants

def generateGridPositions(grid, mask=None):
    """Generate all valid grid positions within the given grid bounds.
    
    Args:
        grid: List [x, y, width, height] defining the grid area.
        mask: Optional raster mask dictionary.
        
    Returns:
        List of [x, y] coordinates for valid grid positions.
    """
    positions = []
    
    if mask is None:
        # Simple rectangular grid
        for y in range(grid[1], grid[1] + grid[3]):
            for x in range(grid[0], grid[0] + grid[2]):
                positions.append([x, y])
    else:
        # Use mask to determine valid positions
        for y in range(grid[1], min(grid[1] + grid[3], mask['height'])):
            for x in range(grid[0], min(grid[0] + grid[2], mask['width'])):
                if y < mask['mask'].shape[0] and x < mask['mask'].shape[1]:
                    if mask['mask'][y, x] == 0:  # Valid position
                        positions.append([x, y])
    
    return positions

def selectBestGridPositions(points, grid_positions, num_needed):
    """Select the best subset of grid positions for the given points.
    
    This uses a greedy approach to select grid positions that minimize
    the total distance to the point cloud centroid.
    
    Args:
        points: NumPy array of point coordinates.
        grid_positions: List of [x, y] grid coordinates.
        num_needed: Number of grid positions to select.
        
    Returns:
        List of selected [x, y] grid coordinates.
    """
    if len(grid_positions) <= num_needed:
        return grid_positions
    
    # Calculate centroid of points
    centroid = np.mean(points, axis=0)
    
    # Calculate distances from each grid position to centroid
    grid_array = np.array(grid_positions)
    distances = np.sum((grid_array - centroid) ** 2, axis=1)
    
    # Select the closest positions
    selected_indices = np.argsort(distances)[:num_needed]
    
    return [grid_positions[i] for i in selected_indices]

def computeCostMatrix(points, grid_positions):
    """Compute the cost matrix for assignment (Euclidean distances).
    
    Args:
        points: NumPy array of shape (n, 2) with point coordinates.
        grid_positions: List of [x, y] grid coordinates.
        
    Returns:
        NumPy array of shape (n, m) with distances between points and grid positions.
    """
    n_points = len(points)
    n_grid = len(grid_positions)
    
    cost_matrix = np.zeros((n_points, n_grid))
    
    grid_array = np.array(grid_positions)
    
    for i, point in enumerate(points):
        # Calculate Euclidean distances from this point to all grid positions
        distances = np.sum((grid_array - point) ** 2, axis=1)
        cost_matrix[i, :] = distances
    
    return cost_matrix
    
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
    # Validate grid bounds
    if grid[2] <= 0 or grid[3] <= 0:
        return
    
    # Ensure grid is within mask bounds
    grid[0] = max(0, min(grid[0], mask.shape[1] - 1))
    grid[1] = max(0, min(grid[1], mask.shape[0] - 1))
    grid[2] = max(1, min(grid[2], mask.shape[1] - grid[0]))
    grid[3] = max(1, min(grid[3], mask.shape[0] - grid[1]))
    
    maskSlice = mask[grid[1]:grid[1]+grid[3],grid[0]:grid[0]+grid[2]]
    rows, cols = maskSlice.shape  # rows = height, cols = width
    
    if rows == 0 or cols == 0:
        return
    
    columnCounts = rows - np.sum(maskSlice, axis=0)  # Fixed: rows for height
    for i in range(len(columnCounts)):
        if columnCounts[i]>0:
            break
        grid[0]+=1
        grid[2]-=1
        if grid[2] <= 0:
            grid[2] = 1
            break
    
    for i in range(len(columnCounts)-1,-1,-1):
        if columnCounts[i]>0:
            break
        grid[2]-=1
        if grid[2] <= 0:
            grid[2] = 1
            break
    
    rowCounts = cols - np.sum(maskSlice,axis=1)  # Fixed: cols for width
    for i in range(len(rowCounts)):
        if rowCounts[i]>0:
            break
        grid[1]+=1
        grid[3]-=1
        if grid[3] <= 0:
            grid[3] = 1
            break
    
    for i in range(len(rowCounts)-1,-1,-1):
        if rowCounts[i]>0:
            break
        grid[3]-=1
        if grid[3] <= 0:
            grid[3] = 1
            break
    

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


    return sorted(list(arrangements), key=cmp_to_key(proportion_sort), reverse=False)

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


def getArrangements(n, includeHexagonalArrangements=True, includeRectangularArrangements=True):
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
    res = []
    if includeHexagonalArrangements:
        res += getShiftedAlternatingRectArrangements(n)
        res += getShiftedSymmetricArrangements(n)
        res += getShiftedTriangularArrangement(n)
        
        # Add shifted circular arrangement
        bestr, bestrp, bestc = getBestShiftedCircularMatch(n)
        if bestc == n:
            res.append(getShiftedCircularArrangement(bestr, bestrp))
            
    if includeRectangularArrangements:
        res += getAlternatingRectArrangements(n)
        res += getSymmetricArrangements(n)
        res += getTriangularArrangement(n)
        bestr, bestrp, bestc = getBestCircularMatch(n)
        if bestc == n:
            res.append(getCircularArrangement(bestr, bestrp))
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

    minr = max(1, int(math.sqrt(n / math.pi))) 
    
    for rp_val in range(10): 
        rpf = rp_val/10.0
        for r_val in range(minr,minr+3): 
            if r_val == 0: continue # Radius cannot be zero
            rlim = (r_val+rpf)*(r_val+rpf)
            c_count = 0 
            for y_coord in range(-r_val,r_val+1): 
                yy = y_coord*y_coord
                for x_coord in range(-r_val,r_val+1): 
                    if x_coord*x_coord+yy<rlim:
                        c_count+=1
            if c_count == n:
                return r_val,rpf,c_count

            if c_count>n and c_count < bestc:
                bestrp = rpf
                bestr = r_val
                bestc = c_count
    return bestr,bestrp,bestc

def getCircularArrangement(radius,adjustFactor=0.0):
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
    rows_list = np.zeros( radius*2+1,dtype=int ) 

    rlim = (radius+adjustFactor)*(radius+adjustFactor)
    for y_coord in range(-radius,radius+1): 
        yy = y_coord*y_coord
        for x_coord in range(-radius,radius+1): 
            if x_coord*x_coord+yy<rlim:
                rows_list[radius+y_coord]+=1

    return {'hex':False,'rows':rows_list,'type':'circular'}
    
    
def getBestShiftedCircularMatch(n):
    """Finds the best shifted circular (hexagonal packing) arrangement parameters for n items.
    
    In hexagonal packing, circles are arranged in a honeycomb pattern with each row
    shifted by half a cell width and rows spaced by sqrt(3)/2 times the cell width.
    
    Args:
        n: The target number of items.
        
    Returns:
        A tuple (best_radius, best_radius_adjust_factor, best_count):
            - best_r: Optimal integer radius.
            - best_rp: Optimal radius adjustment factor (0.0 to 0.9).
            - best_c: The number of points in the best circle found.
    """
    bestc = n * 2
    bestr = 0
    bestrp = 0.0
    
    # Estimate minimum radius based on hexagonal packing density
    # In hexagonal packing, each point occupies √3/2 ≈ 0.866 area units
    hex_density = math.sqrt(3) / 2
    estimated_radius = math.sqrt(n * hex_density / math.pi)
    minr = max(1, int(estimated_radius) - 3)  # Start a bit lower
    maxr = int(estimated_radius) + 5  # Check a wider range
    
    print(f"Estimated radius for {n} points: {estimated_radius:.2f}, checking range {minr}-{maxr}")
    
    for rp in range(10):
        rpf = rp / 10.0
        for r in range(minr, maxr + 1):
            if r == 0:
                continue
            
            radius_squared = (r + rpf) * (r + rpf)
            c = 0
            
            # Vertical spacing between rows in hexagonal packing
            row_spacing = math.sqrt(3) / 2
            
            # Calculate which rows fit within the circle
            max_row_index = int(math.sqrt(radius_squared) / row_spacing) + 1
            
            for row_index in range(-max_row_index, max_row_index + 1):
                y = row_index * row_spacing
                if y * y >= radius_squared:
                    continue
                
                # Maximum x coordinate for this row such that x² + y² ≤ radius²
                x_max = math.sqrt(radius_squared - y * y)
                
                if row_index % 2 == 0:  # Even rows: positions at integers (..., -1, 0, 1, ...)
                    count_in_row = 2 * int(x_max) + 1
                else:  # Odd rows: positions at half-integers (..., -0.5, 0.5, 1.5, ...)
                    count_in_row = 2 * max(0, int(x_max + 0.5))
                
                c += count_in_row
            
            print(f"  Radius {r} + {rpf}: {c} points")
            
            if c == n:
                return r, rpf, c
            
            if c > n and c < bestc:
                bestrp = rpf
                bestr = r
                bestc = c
    
    return bestr, bestrp, bestc


def getShiftedCircularArrangement(radius, adjustFactor=0.0):
    """Creates an arrangement dictionary for a shifted circular (hexagonal packing) layout.
    
    Args:
        radius: Integer, the main radius of the circle.
        adjustFactor: Float, a factor to fine-tune the circle's boundary.
        
    Returns:
        An arrangement dictionary with keys:
            - 'hex': True (indicates hexagonal packing).
            - 'rows': A list of integers representing row lengths for the circle.
            - 'type': 'shifted_circular'.
    """
    radius_squared = (radius + adjustFactor) * (radius + adjustFactor)
    
    # Vertical spacing between rows in hexagonal packing
    row_spacing = math.sqrt(3) / 2
    
    # Calculate which rows fit within the circle
    max_row_index = int(math.sqrt(radius_squared) / row_spacing) + 1
    
    # Create array to store row lengths, indexed from top to bottom
    rows_data = []
    
    for row_index in range(-max_row_index, max_row_index + 1):
        y = row_index * row_spacing
        if y * y >= radius_squared:
            rows_data.append(0)  # Empty row
            continue
        
        # Maximum x coordinate for this row
        x_max = math.sqrt(radius_squared - y * y)
        
        if row_index % 2 == 0:  # Even rows: positions at integers
            count_in_row = 2 * int(x_max) + 1
        else:  # Odd rows: positions at half-integers
            count_in_row = 2 * max(0, int(x_max + 0.5))
        
        rows_data.append(max(0, count_in_row))
    
    # Remove leading and trailing empty rows
    while rows_data and rows_data[0] == 0:
        rows_data.pop(0)
    while rows_data and rows_data[-1] == 0:
        rows_data.pop()
    
    return {'hex': True, 'rows': rows_data, 'type': 'shifted_circular'}    

def arrangement_sort(item1, item2):
    """Comparison function for sorting raster mask arrangements based on aspect ratio.

    Args:
        item1: A raster mask dictionary with 'width' and 'height' keys.
        item2: A raster mask dictionary with 'width' and 'height' keys.

    Returns:
        An integer: -1 if item1 has a better aspect ratio, 0 if equal, 1 if item2 is better.
    """
    width1 = item1['width'] if item1['width'] > 0 else 1
    height1 = item1['height'] if item1['height'] > 0 else 1
    ratio1 = min(width1, height1) / max(width1, height1)

    width2 = item2['width'] if item2['width'] > 0 else 1
    height2 = item2['height'] if item2['height'] > 0 else 1
    ratio2 = min(width2, height2) / max(width2, height2)

    if ratio1 > ratio2:
        return -1
    elif ratio1 < ratio2:
        return 1
    else:
        return 0

def proportion_sort(item1, item2):
    """Comparison function for sorting (width, height) tuples based on aspect ratio.

    Args:
        item1: A tuple (width, height).
        item2: A tuple (width, height).

    Returns:
        An integer: -1 if item1 has a better (closer to 1) aspect ratio,
                    0 if equal, 1 if item2 is better.
    """
    # Ensure width and height are not zero to avoid DivisionByZeroError
    val1 = max(item1[0], 1)
    val2 = max(item1[1], 1)
    ratio1 = min(val1, val2) / max(val1, val2)

    val1 = max(item2[0], 1)
    val2 = max(item2[1], 1)
    ratio2 = min(val1, val2) / max(val1, val2)

    # Compare aspect ratios (closer to 1 is better)
    if ratio1 > ratio2:
        return -1  # item1 has a better aspect ratio
    elif ratio1 < ratio2:
        return 1   # item2 has a better aspect ratio
    else:
        return 0   # equal aspect ratios

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
