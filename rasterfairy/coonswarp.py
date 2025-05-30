#
# Coon Warp v1.0,
# released 23.01.2016
#
#
# Based on Paul s. Heckbert's "Bilinear coons patch image warping"
# Graphics gems IV, Pages 438-446 
# http://dl.acm.org/citation.cfm?id=180937
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
This module implements Coons warp, a technique for image warping.

It is based on Paul S. Heckbert's "Bilinear coons patch image warping"
from Graphics Gems IV, pages 438-446.
http://dl.acm.org/citation.cfm?id=180937
"""

from scipy.spatial import ConvexHull, KDTree, distance
import numpy as np
from scipy import interpolate
import math

def warpCloud( xyc, sourceGridPoints, targetGridPoints, warpQuality=9 ):
    """Warps a point cloud from source to target grid points.

    Args:
        xyc: A NumPy array of shape (N, 2) representing the input point cloud.
        sourceGridPoints: A NumPy array of shape (M, 2) representing the
            source grid points.
        targetGridPoints: A NumPy array of shape (M, 2) representing the
            target grid points.
        warpQuality: An integer specifying the number of nearest neighbors
            to consider for warping.

    Returns:
        A NumPy array of shape (N, 2) representing the warped point cloud.
    """
    sourceTree = KDTree(sourceGridPoints, leafsize=10)
    warpedXYC = []  
    for c in xyc:
        nearest_points = sourceTree.query(c,k=warpQuality)
        nx = 0.0
        ny = 0.0
        ws = 0.0
        for i in range(warpQuality):
            p = targetGridPoints[nearest_points[1][i]]
            w = nearest_points[0][i]
            if w == 0.0:
                nx = p[0]
                ny = p[1]
                ws = 1.0
                break
            else:
                w = 1.0 / w
                nx += w * p[0]
                ny += w * p[1]
                ws += w

        warpedXYC.append([nx/ws,ny/ws])

    warpedXYC = np.array(warpedXYC)
    return warpedXYC

def getCloudGrid( xyc,autoPerimeterOffset=True,autoPerimeterDensity=True,
                 width=64, height=64, perimeterSubdivisionSteps=4, paddingScale=1.05,
                 smoothing=0.001, warpQuality=9, perimeterOffset=None ):
    """Generates a Coons grid for a given point cloud.

    Args:
        xyc: A NumPy array of shape (N, 2) representing the input point cloud.
        autoPerimeterOffset: A boolean indicating whether to automatically
            determine the perimeter offset.
        autoPerimeterDensity: A boolean indicating whether to automatically
            determine the perimeter density.
        width: An integer specifying the width of the grid.
        height: An integer specifying the height of the grid.
        perimeterSubdivisionSteps: An integer specifying the number of
            subdivision steps for the perimeter.
        paddingScale: A float specifying the padding scale for the grid.
        smoothing: A float specifying the smoothing factor for the perimeter.
        warpQuality: An integer specifying the number of nearest neighbors
            to consider for warping.
        perimeterOffset: An integer or None specifying the perimeter offset.

    Returns:
        A NumPy array of shape (width * height, 2) representing the
        Coons grid points.
    """
    bounds, densities = getCloudHull(xyc, width=width, height=height, perimeterSubdivisionSteps=perimeterSubdivisionSteps,
                    smoothing=smoothing,autoPerimeterOffset=autoPerimeterOffset,
                    perimeterOffset=perimeterOffset,autoPerimeterDensity=autoPerimeterDensity)

    return getCoonsGrid(bounds, width=width, height=height,densities=densities,paddingScale=paddingScale)


def rectifyCloud(xyc,autoPerimeterOffset=True,autoPerimeterDensity=True,
                 width=64, height=64,
                 perimeterSubdivisionSteps=4, paddingScale=1.05,
                 smoothing=0.001, warpQuality=9, perimeterOffset=None ):
    """Rectifies a point cloud to a regular grid.

    Args:
        xyc: A NumPy array of shape (N, 2) representing the input point cloud.
        autoPerimeterOffset: A boolean indicating whether to automatically
            determine the perimeter offset.
        autoPerimeterDensity: A boolean indicating whether to automatically
            determine the perimeter density.
        width: An integer specifying the width of the grid.
        height: An integer specifying the height of the grid.
        perimeterSubdivisionSteps: An integer specifying the number of
            subdivision steps for the perimeter.
        paddingScale: A float specifying the padding scale for the grid.
        smoothing: A float specifying the smoothing factor for the perimeter.
        warpQuality: An integer specifying the number of nearest neighbors
            to consider for warping.
        perimeterOffset: An integer or None specifying the perimeter offset.

    Returns:
        A NumPy array of shape (N, 2) representing the rectified point cloud.
    """
    sourceGridPoints = getCloudGrid( xyc,autoPerimeterOffset=autoPerimeterOffset,autoPerimeterDensity=autoPerimeterDensity,
                 width=width, height=height,
                 perimeterSubdivisionSteps=perimeterSubdivisionSteps, paddingScale=paddingScale,
                 smoothing=smoothing, warpQuality=warpQuality, perimeterOffset=perimeterOffset)
    
    targetGridPoints = []
    for yi in range(height):
        for xi in range(width):
            targetGridPoints.append([xi,yi])

    return warpCloud( xyc, sourceGridPoints, targetGridPoints, warpQuality=warpQuality )


def getCloudHull(xyc,width=64,height=64,perimeterSubdivisionSteps=4,smoothing=0.001,
                 autoPerimeterOffset=True, perimeterOffset=None, autoPerimeterDensity=True):
    """Calculates the convex hull of a point cloud and its properties.

    Args:
        xyc: A NumPy array of shape (N, 2) representing the input point cloud.
        width: An integer specifying the width of the grid.
        height: An integer specifying the height of the grid.
        perimeterSubdivisionSteps: An integer specifying the number of
            subdivision steps for the perimeter.
        smoothing: A float specifying the smoothing factor for the perimeter splines.
        autoPerimeterOffset: A boolean indicating whether to automatically
            determine the perimeter offset.
        perimeterOffset: An integer or None specifying the perimeter offset.
        autoPerimeterDensity: A boolean indicating whether to automatically
            determine the perimeter density.

    Returns:
        A tuple containing:
            - bounds: A dictionary containing the smoothed splines for the
              top, right, bottom, and left boundaries of the hull.
            - densities: A dictionary containing the density of points along
              each boundary, or None if autoPerimeterDensity is False.
    """
    tree = KDTree(xyc, leafsize=10)

    hull = ConvexHull(xyc)
    hullPoints = []
    hullIndices = {}
    for i in range(len(hull.vertices)):
        hullIndices[hull.vertices[i]] = True
        hullPoints.append(xyc[hull.vertices[i]])

    for j in range(perimeterSubdivisionSteps):
        newPoints = []
        for i in range(len(hullPoints)):
            newPoints.append(hullPoints[i])
            midpoint = lerp(hullPoints[i], hullPoints[(i+1)%len(hullPoints)], 0.5)
            index = tree.query(midpoint)[1]
            if index not in hullIndices:
                newPoints.append(xyc[index])
                hullIndices[index] = True
        hullPoints = newPoints


    perimeterLength = 0
    for i in range(len(hullPoints)):
        perimeterLength += distance.euclidean(hullPoints[i],hullPoints[(i+1)%len(hullPoints)])

    perimeterCount = 2 * (width + height) - 4
    perimeterStep = perimeterLength / perimeterCount
    perimeterPoints = []
    perimeterDensity = np.zeros(perimeterCount)
    
    for i in range(perimeterCount):
        t = 1.0 * i / perimeterCount
        poh = getPointOnHull(hullPoints,t,perimeterLength)
        perimeterPoints.append(poh)
        perimeterDensity[i] = np.mean(tree.query(poh,k=32)[0])
    
    if autoPerimeterOffset:
        bestDensity = perimeterDensity[0] + perimeterDensity[width-1] + perimeterDensity[width+height-2] + perimeterDensity[2*width+height-3]
        perimeterOffset = 0     
        for i in range(1,width+height ):
            density = perimeterDensity[i] + perimeterDensity[(i+width-1)%perimeterCount] + perimeterDensity[(i+width+height-2)%perimeterCount] + perimeterDensity[(i+2*width+height-3)%perimeterCount]
            if density < bestDensity:
                bestDensity = density
                perimeterOffset = i
    elif perimeterOffset is None:
        perimeterOffset = 0
        corner = [np.min(xyc[:,0]),np.min(xyc[:,1])]
        d = corner-perimeterPoints[0]
        clostestDistanceToCorner = np.hypot(d[0],d[1])
        for i in range(1,perimeterCount):
            d = corner-perimeterPoints[i]
            distanceToCorner = np.hypot(d[0],d[1])
            if ( distanceToCorner < clostestDistanceToCorner):
                clostestDistanceToCorner = distanceToCorner
                perimeterOffset = i
        
    
    perimeterPoints = np.array(perimeterPoints)
    if perimeterOffset > 0:
        perimeterPoints = np.roll(perimeterPoints, -perimeterOffset, axis=0)


    perimeterPoints = np.append(perimeterPoints,[perimeterPoints[0]],axis=0)

    bounds = {'top':perimeterPoints[0:width],
              'right':perimeterPoints[width-1:width+height-1],
              'bottom':perimeterPoints[width+height-2:2*width+height-2],
              'left':perimeterPoints[2*width+height-3:]}

    bounds['s_top'],u = interpolate.splprep([bounds['top'][:,0], bounds['top'][:,1]],s=smoothing)
    bounds['s_right'],u = interpolate.splprep([bounds['right'][:,0],bounds['right'][:,1]],s=smoothing)
    bounds['s_bottom'],u = interpolate.splprep([bounds['bottom'][:,0],bounds['bottom'][:,1]],s=smoothing)
    bounds['s_left'],u = interpolate.splprep([bounds['left'][:,0],bounds['left'][:,1]],s=smoothing)
    
    
    densities = None
    if autoPerimeterDensity:
        densities = {}
    
        density_top = np.zeros(len(bounds['top']))
        for i in range(len(density_top)):
            t = 1.0 * i / len(density_top)
            density_top[i] = np.mean(tree.query( np.array(interpolate.splev( t,bounds['s_top'])).flatten(),k=64)[0])
        density_top /= np.sum(density_top)
        
        density_right = np.zeros(len(bounds['right']))
        for i in range(len(density_right)):
            t = 1.0 * i / len(density_right)
            density_right[i] = np.mean(tree.query( np.array(interpolate.splev( t,bounds['s_right'])).flatten(),k=64)[0])
        density_right /= np.sum(density_right)
        
        density_bottom = np.zeros(len(bounds['bottom']))
        for i in range(len(density_bottom)):
            t = 1.0 * i / len(density_bottom)
            density_bottom[i] = np.mean(tree.query( np.array(interpolate.splev( t,bounds['s_bottom'])).flatten(),k=64)[0])
        density_bottom /= np.sum(density_bottom)
        
        density_left = np.zeros(len(bounds['left']))
        for i in range(len(density_left)):
            t = 1.0 * i / len(density_left)
            density_left[i] = np.mean(tree.query( np.array(interpolate.splev( t,bounds['s_left'])).flatten(),k=64)[0])
        density_left /= np.sum(density_left)
        
        densities = {'top':density_top,'right':density_right,'bottom':density_bottom,'left':density_left}
    
    
    return bounds, densities

def getCircularGrid( fitCloud=None, width=64, height=64, paddingScale=1.0):
    """Generates a circular Coons grid.

    Args:
        fitCloud: A NumPy array of shape (N, 2) representing a point cloud
            to fit the circular grid to, or None to create a default circular grid.
        width: An integer specifying the width of the grid.
        height: An integer specifying the height of the grid.
        paddingScale: A float specifying the padding scale for the grid.

    Returns:
        A NumPy array of shape (width * height, 2) representing the
        circular Coons grid points.
    """
    return getCoonsGrid(getCircularBounds(fitCloud=fitCloud,width=width,height=height),width=width,height=height, paddingScale=paddingScale)

def getCircularBounds(fitCloud=None,width=64,height=64,smoothing=0.01):
    """Calculates the boundary splines for a circular grid.

    Args:
        fitCloud: A NumPy array of shape (N, 2) representing a point cloud
            to fit the circular bounds to, or None for default circular bounds.
        width: An integer specifying the width of the grid.
        height: An integer specifying the height of the grid.
        smoothing: A float specifying the smoothing factor for the boundary splines.

    Returns:
        A dictionary containing the smoothed splines for the top, right,
        bottom, and left boundaries of the circular grid.
    """
    circumference = 2*(width+height)

    if not fitCloud is None:
        cx = np.mean(fitCloud[:,0])
        cy = np.mean(fitCloud[:,1])
        r = 0.5* max( np.max(fitCloud[:,0])- np.min(fitCloud[:,0]),np.max(fitCloud[:,1])- np.min(fitCloud[:,1]))
    else:
        r = circumference /(2.0*math.pi)
        cx = cy = r
    perimeterPoints = np.zeros((circumference,2),dtype=float)
    for i in range(circumference):
        angle = (2.0*math.pi)*float(i) / circumference - math.pi * 0.5 
        perimeterPoints[i][0] = cx + r * math.cos(angle)
        perimeterPoints[i][1] = cy + r * math.sin(angle)
        
        
    bounds = {'top':perimeterPoints[0:width],
              'right':perimeterPoints[width-1:width+height-1],
              'bottom':perimeterPoints[width+height-2:2*width+height-2],
              'left':perimeterPoints[2*width+height-3:]}
    
    bounds['s_top'],u = interpolate.splprep([bounds['top'][:,0], bounds['top'][:,1]],s=smoothing)
    bounds['s_right'],u = interpolate.splprep([bounds['right'][:,0],bounds['right'][:,1]],s=smoothing)
    bounds['s_bottom'],u = interpolate.splprep([bounds['bottom'][:,0],bounds['bottom'][:,1]],s=smoothing)
    bounds['s_left'],u = interpolate.splprep([bounds['left'][:,0],bounds['left'][:,1]],s=smoothing)
   
    
    return bounds


def getCoonsGrid( bounds, width=64, height=64, densities=None, paddingScale=1.0):
    """Generates a Coons grid from given boundary splines.

    Args:
        bounds: A dictionary containing the smoothed splines for the
            top, right, bottom, and left boundaries.
        width: An integer specifying the width of the grid.
        height: An integer specifying the height of the grid.
        densities: A dictionary containing the density of points along
            each boundary, or None for uniform density.
        paddingScale: A float specifying the padding scale for the grid.

    Returns:
        A NumPy array of shape (width * height, 2) representing the
        Coons grid points.
    """
    targets = []
    for yi in range(height):
        for xi in range(width):
            targets.append(getCoonsPatchPointBez(bounds,xi,yi,width,height,densities=densities))

    targets = np.array(targets)
    tmean = [np.mean(targets[:,0]),np.mean(targets[:,1])]
    targets -= tmean
    targets *= paddingScale
    targets += tmean
    
    return targets


def getCoonsPatchPointBez(bounds,x,y,width,height, densities = None):
    """Calculates a point on a Coons patch using Bezier interpolation.

    Args:
        bounds: A dictionary containing the smoothed splines for the
            top, right, bottom, and left boundaries.
        x: An integer representing the x-coordinate on the grid.
        y: An integer representing the y-coordinate on the grid.
        width: An integer specifying the width of the grid.
        height: An integer specifying the height of the grid.
        densities: A dictionary containing the density of points along
            each boundary, or None for uniform density.

    Returns:
        A NumPy array of shape (2,) representing the coordinates of the
        point on the Coons patch.
    """
    p00 = np.array(interpolate.splev( 0.0,bounds['s_top'])).flatten()
    p10 = np.array(interpolate.splev( 1.0,bounds['s_top'])).flatten()
    p11 = np.array(interpolate.splev( 0.0,bounds['s_bottom'])).flatten()
    p01 = np.array(interpolate.splev( 1.0,bounds['s_bottom'])).flatten()
    
    u = 1.0 * x / (width-1)
    v = 1.0 * y / (height-1)
    iu = 1.0 - u
    iv = 1.0 - v
    if densities is None:
        pu0 = np.array(interpolate.splev( u,bounds['s_top'])).flatten()
        pu1 = np.array(interpolate.splev(iu,bounds['s_bottom'])).flatten()
        pv0 = np.array(interpolate.splev(iv,bounds['s_left'])).flatten()
        pv1 = np.array(interpolate.splev( v,bounds['s_right'])).flatten()
    else:
        ut = 0.0
        ub = 0.0
        for i in range(x):
            ut+=densities['top'][i]
            ub+=densities['bottom'][i]
        vl = 0.0
        vr = 0.0
        for i in range(y):
            vl+=densities['left'][i]
            vr+=densities['right'][i]
        
        pu0 = np.array(interpolate.splev( ut,bounds['s_top'])).flatten()
        pu1 = np.array(interpolate.splev(1.0-ub,bounds['s_bottom'])).flatten()
        pv0 = np.array(interpolate.splev(1.0-vl,bounds['s_left'])).flatten()
        pv1 = np.array(interpolate.splev( vr,bounds['s_right'])).flatten()   
        
    return iv * pu0 + v * pu1 + iu * pv0 + u * pv1 - iu * iv * p00 - u * iv * p10 - iu * v * p01 - u * v * p11


def lerp( p1, p2, t):
    """Performs linear interpolation between two points.

    Args:
        p1: A NumPy array representing the first point.
        p2: A NumPy array representing the second point.
        t: A float representing the interpolation factor (between 0 and 1).

    Returns:
        A NumPy array representing the interpolated point.
    """
    return (1.0-t)*p1+t*p2

def leftOrRight(p,l1,l2):
    """Determines if a point is to the left or right of a line segment.

    Args:
        p: A NumPy array representing the point to check.
        l1: A NumPy array representing the start point of the line segment.
        l2: A NumPy array representing the end point of the line segment.

    Returns:
        An integer:
            - 1 if p is to the left of the line segment.
            - -1 if p is to the right of the line segment.
            - 0 if p is collinear with the line segment.
    """
    return np.sign((l2[0] - l1[0]) * (p[1] - l1[1]) - (l2[1] - l1[1]) * (p[0] - l1[0]))

def getPointOnHull( hullPoints,t, totalLength ):
    """Gets a point on the convex hull at a given normalized distance.

    Args:
        hullPoints: A list of NumPy arrays representing the points of the
            convex hull in order.
        t: A float representing the normalized distance along the hull
            perimeter (between 0 and 1).
        totalLength: A float representing the total length of the hull perimeter.

    Returns:
        A NumPy array representing the point on the hull.
    """
    lh = len(hullPoints)
    for j in range(lh+1):
        sideLength = distance.euclidean(hullPoints[j%lh],hullPoints[(j+1)%lh])
        t_sub = sideLength / totalLength;
        if t > t_sub:
            t-= t_sub
        else:
            return lerp(hullPoints[j%lh],hullPoints[(j+1)%lh], t / t_sub )

    return hullPoints[-1]             
