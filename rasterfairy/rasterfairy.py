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

import numpy as np
import prime
import math

def transformPointCloud2D( points2d, target = None, autoAdjustCount = True, proportionThreshold = 0.4):
    pointCount = len(points2d)
    rasterMask = None
    
    if target is None:
        target = getRectArrangements(pointCount)[0]
        if (float(target[0]) / float(target[1])<proportionThreshold):
            width = int(math.sqrt(pointCount))
            height = int(math.ceil(float(pointCount)/float(width)))
            print "no good rectangle found for",pointCount,"points, using incomplete square",width,"*",height
            target = {'width':width,'height':height,'mask':np.zeros((height,width),dtype=int), 'count':width*height, 'hex': False}
        
    if type(target) is tuple and len(target)==2:
        #print "using rectangle target"
        if target[0] * target[1] < pointCount:
            print "ERROR: target rectangle is too small to hold data: Rect is",target[0],"*",target[1],"=",target[0] * target[1]," vs "+pointCount+" data points"
            return False
        width = target[0]
        height = target[1]
        
    elif "PIL." in str(type(target)):
        #print "using bitmap image target"
        rasterMask = getRasterMaskFromImage(target)
        width = rasterMask['width']
        height = rasterMask['height']
    elif 'mask' in target and 'count' in target and 'width' in target and 'height' in target:
        #print "using raster mask target"
        rasterMask = target
        width = rasterMask['width']
        height = rasterMask['height']
    
    if not (rasterMask is None) and rasterMask['mask'].shape[0]*rasterMask['mask'].shape[1]-np.sum( rasterMask['mask'].flat) < len(points2d):
        print "ERROR: raster mask target does not have enough grid points to hold data"
        return False
    
    if not (rasterMask is None) and (rasterMask['count']!=len(points2d)):
        mask = rasterMask['mask'].flatten()
        count = len(points2d)
        if autoAdjustCount is True:
            
            if count > rasterMask['count']:
                ones = np.nonzero(mask)[0]
                np.random.shuffle(ones)
                mask[ones[0:count-rasterMask['count']]] = 0
            elif count < rasterMask['count']:
                zeros = np.nonzero(1-mask)[0]
                np.random.shuffle(zeros)
                mask[zeros[0:rasterMask['count']-count]] = 1
        else:
            if count > rasterMask['count']:
                ones = np.nonzero(mask)[0]
                mask[ones[rasterMask['count']]-count:] = 0
            elif count < rasterMask['count']:
                zeros = np.nonzero(1-mask)[0]
                mask[zeros[rasterMask['count']-count]] = 1
                
        mask = mask.reshape((rasterMask['height'], rasterMask['width']))
        rasterMask = {'width':rasterMask['width'],'height':rasterMask['height'],'mask':mask, 'count':count, 'hex': rasterMask['hex']}
    quadrants = [{'points':points2d, 'grid':[0,0,width,height], 'indices':np.arange(pointCount)}]
    i = 0
    while i < len(quadrants) and len(quadrants) < pointCount:
        if ( len(quadrants[i]['points']) > 1 ):
            slices = sliceQuadrant(quadrants[i], mask = rasterMask)
            del quadrants[i]
            quadrants += slices
            i = 0
        else:
            i+=1

    gridPoints2d = points2d.copy()

    if not (rasterMask is None) and rasterMask['hex'] is True:
        f = math.sqrt(3.0)/2.0 
        offset = -0.5
        if np.argmin(rasterMask['mask'][0]) > np.argmin(rasterMask['mask'][1]):
            offset = 0.5
        for q in quadrants:
            if (q['grid'][1]%2==0):
                q['grid'][0]-=offset
            q['grid'][1] *= f

    for q in quadrants:
        gridPoints2d[q['indices'][0]] = np.array(q['grid'][0:2],dtype=np.float)

    return gridPoints2d, (width, height)


def sliceQuadrant( quadrant, mask = None ):
    
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
            order = np.lexsort((xy[:,1],xy[:,0]))
            sliceCount = sliceXCount
            sliceSize  = grid[2] / sliceCount
            pointsPerSlice = grid[3] * sliceSize
            gridOffset = grid[0]
        else:
            order = np.lexsort((xy[:,0],xy[:,1]))    
            sliceCount = sliceYCount
            sliceSize = grid[3] / sliceCount
            pointsPerSlice = grid[2] * sliceSize
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
        
        maskSlice = mask['mask'][grid[1]:grid[1]+grid[3],grid[0]:grid[0]+grid[2]]
        cols,rows = maskSlice.shape
        pointCountInMask = cols*rows - np.sum(maskSlice)
       
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
        
        order = np.lexsort((xy[:,1],xy[:,0]))
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
        
        order = np.lexsort((xy[:,0],xy[:,1]))
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
    img = img.convert('1')
    mask = np.array(img.getdata())&1
    return {'width':img.width,'height':img.height,'mask':mask.reshape((img.height, img.width)), 'count':len(mask)- np.sum(mask), 'hex':False}


def getCircleRasterMask( r, innerRingRadius = 0, rasterCount = None, autoAdjustCount = True  ): 
    
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
    #print "adjusted count",p.shape[0] * p.shape[1]- np.sum(p)
    return {'width':d,'height':d,'mask':p, 'count':count}
        
def getRectArrangements(n):
    p = prime.Prime()
    f = p.getPrimeFactors(n)
    f_count = len(f)
    ma = multiplyArray(f)
    arrangements = set([(1,ma)])

    if (f_count > 1):
        perms = set(p.getPermutations(f))
        for perm in perms:
            for i in range(1,f_count):
                v1 = multiplyArray(perm[0:i])
                v2 = multiplyArray(perm[i:])
                arrangements.add((min(v1, v2),max(v1, v2)))

    return sorted(list(arrangements), cmp=proportion_sort, reverse=True)

def getShiftedAlternatingRectArrangements(n):
    arrangements = set([])
    for x in range(1,n >> 1):
        v = 2 * x + 1
        if n % v == x:
            arrangements.add((x, x + 1, ((n / v) | 0) * 2 + 1))
        
    for x in range(2,1 + (n >> 1)):
        v = 2 * x - 1
        if n % v == x:
            arrangements.add((x, x - 1, ((n / v) | 0) * 2 + 1))
    
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
    hits = []
    for i in range(1, n >> 1):
        d = []
        count = n
        row = i
        d.append(row)
        while True:
            count -= row * 2
            if count == row + 1:
                if i != row:
                    d.append(row+1)
                    d+=d[0:-1][::-1]
                    hits.append({'hex':True,'rows':d,'type':'symmetric'})
                    break
            elif count <= 0:
                break
            row+=1
            d.append(row)
    return hits

def getShiftedTriangularArrangement(n):

    t = math.sqrt(8 * n + 1);
    if t != math.floor(t):
        return []
    
    arrangement = []
    i = 1
    while n>0:
        arrangement.append(i)
        n-=i
        i+=1
    
    return [{'hex':True,'rows':arrangement,'type':'triangular'}]

def getAlternatingRectArrangements(n):
    arrangements = set([])
    for x in range(1,n >> 1):
        v = 2 * x + 2
        if n % v == x:
            arrangements.add((x, x + 2, ((n / v) | 0) * 2 + 1))
        
    for x in range(2,1 + (n >> 1)):
        v = 2 * x - 2
        if n % v == x:
            arrangements.add((x, x -2, ((n / v) | 0) * 2 + 1))
    
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
    hits = []
    for i in range(1, n >> 1):
        d = []
        count = n
        row = i
        d.append(row)
        while True:
            count -= row * 2
            if count == row + 2:
                if i != row:
                    d.append(row+2)
                    d+=d[0:-1][::-1]
                    hits.append({'hex':False,'rows':d,'type':'symmetric'})
                    break
            elif count <= 0:
                break
            row+=2
            d.append(row)
    return hits

def getTriangularArrangement(n):

    t = math.sqrt(n);
    if t != math.floor(t):
        return []
    
    arrangement = []
    i = 1
    while n>0:
        arrangement.append(i)
        n-=i
        i+=2
    
    return [{'hex':False,'rows':arrangement,'type':'triangular'}]


def getArrangements(n, includeHexagonalArrangements = True,includeRectangularArrangements = True):
    r = []
    if includeHexagonalArrangements:
        r += getShiftedAlternatingRectArrangements(n)
        r += getShiftedSymmetricArrangements(n)
        r += getShiftedTriangularArrangement(n)
    if includeRectangularArrangements:
        r += getAlternatingRectArrangements(n)
        r += getSymmetricArrangements(n)
        r += getTriangularArrangement(n)
        bestr,bestrp,bestc = getBestCircularMatch(n)
        if bestc == n:
            r.append( getCircularArrangement(bestr,bestrp))
    return r

def arrangementListToRasterMasks( arrangements ):
    masks = []
    for i in range(len(arrangements)):
        masks.append(arrangementToRasterMask(arrangements[i]))
    return sorted(masks, cmp=arrangement_sort, reverse=True)

def arrangementToRasterMask( arrangement ):
    rows = np.array(arrangement['rows'])
    width = np.max(rows)
    if arrangement['hex'] is True:
        width+=1
    height = len(rows)
    mask = np.ones((height,width),dtype=int)
    for row in range(len(rows)):
        c = rows[row]
        mask[row,(width-c)>>1:((width-c)>>1)+c] = 0

    return {'width':width,'height':height,'mask':mask, 'count':np.sum(rows),'hex':arrangement['hex'],'type':arrangement['type']}

def rasterMaskToGrid( rasterMask ):
    grid = []
    mask = rasterMask['mask']
    for y in range(rasterMask['height']):
        for x in range(rasterMask['width']):
            if mask[y,x]==0:
                grid.append([x,y])
    
    grid = np.array(grid,dtype=np.float)
    if not (rasterMask is None) and rasterMask['hex'] is True:
        f = math.sqrt(3.0)/2.0 
        offset = -0.5
        if np.argmin(rasterMask['mask'][0]) > np.argmin(rasterMask['mask'][1]):
            offset = 0.5
        for i in range(len(grid)):
            if (grid[i][1]%2.0==0.0):
                grid[i][0]-=offset
            grid[i][1] *= f
    return grid
            

def getBestCircularMatch(n):
    bestc = n*2
    bestr = 0
    bestrp = 0.0
    
    minr = int(math.sqrt(n / math.pi))
    for rp in range(0,10):
        rpf = float(rp)/10.0
        for r in range(minr,minr+3):
            rlim = (r+rpf)*(r+rpf)
            c = 0
            for y in range(-r,r+1):
                yy = y*y
                for x in range(-r,r+1):
                    if x*x+yy<rlim:
                        c+=1
            if c == n:
                return r,rpf,c
               
            if c>n and c < bestc:
                bestrp = rpf
                bestr = r
                bestc = c
    return bestr,bestrp,bestc

def getCircularArrangement(radius,adjustFactor):
    rows = np.zeros( radius*2+1,dtype=int )
    
    rlim = (radius+adjustFactor)*(radius+adjustFactor)
    for y in range(-radius,radius+1):
        yy = y*y
        for x in range(-radius,radius+1):
            if x*x+yy<rlim:
                rows[radius+y]+=1
    
    return {'hex':False,'rows':rows,'type':'circular'}

def arrangement_sort(x, y):
    return int(100000000*(abs(float(min(x['width'],x['height'])) / float(max(x['width'],x['height']))) - abs(float(min(y['width'],y['height'])) / float(max(y['width'],y['height'])))))

def proportion_sort(x, y):
    return int(100000000*(abs(float(min(x[0],x[1])) / float(max(x[0],x[1]))) - abs(float(min(y[0],y[1])) / float(max(y[0],y[1])))))

def multiplyArray(a):
    f = 1
    for v in a: 
        f *= v
    return f