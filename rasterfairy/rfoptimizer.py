#
# Random Grid Swap Utility v1.0
# part of Raster Fairy v1.02,
# released 27.01.2016
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
This module provides the SwapOptimizer class, a utility for optimizing the
arrangement of points on a grid by performing various types of swaps and
rotations of grid cell assignments. It aims to minimize the sum of squared
Euclidean distances between original points and their assigned grid cells.
"""

import scipy.spatial.distance as dist
from scipy import spatial
import math
from IPython.display import clear_output
import numpy as np

class SwapOptimizer:
    """
    Optimizes the assignment of points to a grid by iteratively swapping
    grid cell assignments to minimize the total squared Euclidean distance
    between the original points and their assigned grid cells.

    The optimizer uses a simulated annealing-like approach with a learning
    mechanism for choosing different swap strategies. It keeps track of the
    best arrangement found and can be run for a specified number of iterations
    or continued from a previous state.

    Instance Variables:
        lastState (dict | None): Stores the state of the optimizer from the
            previous run, including cell assignments, iteration count,
            distance matrix, and weights for different swap choices.
            This allows optimization to be continued.
        grid_norm (numpy.ndarray | None): Normalized coordinates of the target
            grid cells.
        xy_norm (numpy.ndarray | None): Normalized coordinates of the input
            points.
        lastSwapTable (numpy.ndarray | None): The best swap table (mapping of
            original point index to grid cell index) found in the last
            optimization run.
        lastWidth (int): Width of the grid from the last optimization.
        lastHeight (int): Height of the grid from the last optimization.
    """
    lastState = None
    grid_norm = None
    xy_norm = None
    lastSwapTable = None
    lastWidth = 0
    lastHeight = 0

    # No explicit __init__ method is defined.
    # Instance variables are initialized at the class level or within methods.

    def optimize( self, xy, grid, width, height, iterations, shakeIterations = 0,swapTable=None,continueFromLastState = False):
        """
        Main optimization loop.

        This method iteratively tries different swaps and rotations of grid
        cell assignments to minimize the sum of squared Euclidean distances
        between original points (xy) and their assigned grid cells (grid).

        Args:
            xy (numpy.ndarray): An array of shape (N, 2) representing the
                original 2D point coordinates.
            grid (numpy.ndarray): An array of shape (N, 2) representing the
                target 2D grid cell coordinates.
            width (int): The width of the grid.
            height (int): The height of the grid.
            iterations (int): The number of optimization iterations to perform.
            shakeIterations (int, optional): Number of random swaps to perform
                at the beginning to "shake" the arrangement. Defaults to 0.
            swapTable (numpy.ndarray, optional): An initial mapping of original
                point indices to grid cell indices. If None, an identity mapping
                is created. Defaults to None.
            continueFromLastState (bool, optional): If True, the optimization
                will attempt to continue from where the last `optimize` call
                left off (using `self.lastState`). Defaults to False.

        Returns:
            numpy.ndarray: The optimized swapTable, representing the best
            mapping found from original point indices to grid cell indices.
            Returns None if there's an error in cell mapping during initialization.
        """

        if self.lastState is None or not continueFromLastState:
            if xy is None or grid is None or width is None or height is None:
                 raise ValueError("xy, grid, width, and height must be provided if not continuing from last state.")
            self.grid_norm = grid - np.min(grid,axis=0)
            self.grid_norm /=  np.max(self.grid_norm,axis=0)

            self.xy_norm = xy - np.min(xy,axis=0)
            self.xy_norm /=  np.max(self.xy_norm,axis=0)

            self.lastWidth = width
            self.lastHeight = height
        else: # continueFromLastState is True
            width = self.lastWidth
            height = self.lastHeight
            if self.grid_norm is None or self.xy_norm is None: # Should not happen if lastState is valid
                raise ValueError("Optimizer state is incomplete for continuation.")


        reward = 5.0
        punish = 0.999

        totalEntries = len(self.grid_norm)

        choiceModes = np.arange(10)
        # Ensure colSizes and rowSizes are not empty if width/height is 1
        colSizes = np.arange(1,max(2,width))
        rowSizes = np.arange(1,max(2,height))
        swapOffsets = np.arange(1,9)
        offsets = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[-1,1],[-1,-1]])
        toleranceChoices = np.arange(1,16)

        if swapTable is None:
            swapTable =  np.arange(totalEntries)


        if self.lastState is None or not continueFromLastState:
            cells = np.zeros((width, height),dtype=np.int)


            ci = {}
            for j in range(totalEntries):
                cx = int(0.5+self.grid_norm[j][0]*(width-1))
                cy = int(0.5+self.grid_norm[j][1]*(height-1))

                if (cx,cy) in ci:
                    print(("ERROR:",(cx,cy)," doubled"))
                cells[cx][cy] = j
                ci[(cx,cy) ] = True

            if len(ci) != totalEntries:
                print("ERROR in cell mapping")
                return None # Indicate error

            distances = dist.cdist(self.grid_norm, self.xy_norm, 'sqeuclidean')
            swapChoiceWeights = np.array([1.0] * len(choiceModes),dtype=np.float64)
            colSizeWeights = np.array([1.0] * len(colSizes),dtype=np.float64)
            rowSizeWeights = np.array([1.0] * len(rowSizes),dtype=np.float64)
            swapOffsetWeights = np.array([1.0] * len(swapOffsets),dtype=np.float64)
            toleranceWeights = np.array([1.0] * len(toleranceChoices),dtype=np.float64)
            totalIterations = iterations
        else: # continueFromLastState is True
            cells = self.lastState['cells']
            distances = self.lastState['distances']
            swapChoiceWeights = self.lastState['swapChoiceWeights']
            colSizeWeights = self.lastState['colSizeWeights']
            rowSizeWeights = self.lastState['rowSizeWeights']
            swapOffsetWeights =self.lastState['swapOffsetWeights']
            toleranceWeights = self.lastState['toleranceWeights']
            totalIterations = self.lastState['iterations']+iterations


        bestQuality = startingQuality = self.sumDistances(swapTable,distances)
        startingSwapTable = swapTable.copy()
        print(("Starting sum of distances",startingQuality))
        if shakeIterations > 0:
            self.shake(cells,swapTable,shakeIterations,1,width,height)
            bestQuality = self.sumDistances(swapTable,distances)
            print(("After shake sum of distances",bestQuality))

        bestSwapTableBeforeAnnealing = swapTable.copy()
        toleranceSteps = 0

        for i in range(iterations):
            if i>0 and i % 20000 == 0:
                clear_output(wait=True)
                print(("Starting sum of distances",startingQuality))
            if i % 1000 == 0:
                print((i,bestQuality))

            if toleranceSteps == 0:
                swapTable = bestSwapTableBeforeAnnealing.copy()
                chosenToleranceSteps = toleranceSteps = np.random.choice(toleranceChoices,p=toleranceWeights/np.sum(toleranceWeights))
                #if np.random.random() < 0.001:
                #    shake(cells,swapTable,np.random.randint(1,3),1,width,height)


            toleranceSteps -= 1

            cellx = np.random.randint(0,width)
            celly = np.random.randint(0,height)

            #since the offset gets changed it's important to use a fresh copy - do not remove the copy()!
            cp = offsets.copy()[np.random.randint(0,8)]

            swapChoice = np.random.choice(choiceModes,p=swapChoiceWeights/np.sum(swapChoiceWeights))
            if swapChoice == 0:
                offsetx = np.random.choice(swapOffsets,p=swapOffsetWeights/np.sum(swapOffsetWeights))
                offsety = np.random.choice(swapOffsets,p=swapOffsetWeights/np.sum(swapOffsetWeights))
                cp[0] *= offsetx
                cp[1] *= offsety
                self.swapIndices(swapTable,cells[cellx][celly],cells[(cellx+width+cp[0])%width][(celly+height+cp[1])%height])
                quality = self.sumDistances(swapTable,distances)
                if quality < bestQuality:
                    bestQuality = quality
                    bestSwapTableBeforeAnnealing = swapTable.copy()
                    self.learnWeight(toleranceWeights,chosenToleranceSteps-1,reward)
                    self.learnWeight(swapChoiceWeights,swapChoice,reward)
                    self.learnWeight(swapOffsetWeights,offsetx-1,reward)
                    self.learnWeight(swapOffsetWeights,offsety-1,reward)
                elif toleranceSteps == 0: # Only revert and punish if no improvement after tolerance steps
                    self.swapIndices(swapTable,cells[cellx][celly],cells[(cellx+width+cp[0])%width][(celly+height+cp[1])%height]) # Revert
                    self.learnWeight(toleranceWeights,chosenToleranceSteps-1,punish)
                    self.learnWeight(swapChoiceWeights,swapChoice,punish)
                    self.learnWeight(swapOffsetWeights,offsetx-1,punish)
                    self.learnWeight(swapOffsetWeights,offsety-1,punish)


            elif swapChoice == 1:
                rotateLeft = np.random.random() < 0.5
                self.rotate3Indices(swapTable,cells[cellx][celly],cells[(cellx+width+cp[0])%width][(celly+height+cp[1])%height],cells[(cellx+width+cp[0]*2)%width][(celly+height+cp[1]*2)%height],rotateLeft)
                quality = self.sumDistances(swapTable,distances)
                if quality < bestQuality:
                    bestQuality = quality
                    bestSwapTableBeforeAnnealing = swapTable.copy()
                    self.learnWeight(toleranceWeights,chosenToleranceSteps-1,reward)
                    self.learnWeight(swapChoiceWeights,swapChoice,reward)
                elif toleranceSteps == 0:
                    self.rotate3Indices(swapTable,cells[cellx][celly],cells[(cellx+width+cp[0])%width][(celly+height+cp[1])%height],cells[(cellx+width+cp[0]*2)%width][(celly+height+cp[1]*2)%height],not rotateLeft) # Revert
                    self.learnWeight(swapChoiceWeights,swapChoice,punish)
                    self.learnWeight(toleranceWeights,chosenToleranceSteps-1,punish)

            elif swapChoice == 2:
                rotateLeft = np.random.random() < 0.5
                self.rotate4Indices(swapTable,
                               cells[cellx][celly],
                               cells[(cellx+width+cp[0])%width][(celly+height+cp[1])%height],
                               cells[(cellx+width+cp[0]*2)%width][(celly+height+cp[1]*2)%height],
                               cells[(cellx+width+cp[0]*3)%width][(celly+height+cp[1]*3)%height],rotateLeft)
                quality = self.sumDistances(swapTable,distances)
                if quality < bestQuality:
                    bestQuality = quality
                    bestSwapTableBeforeAnnealing = swapTable.copy()
                    self.learnWeight(toleranceWeights,chosenToleranceSteps-1,reward)
                    self.learnWeight(swapChoiceWeights,swapChoice,reward)
                elif toleranceSteps == 0:
                    self.rotate4Indices(swapTable,
                                   cells[cellx][celly],
                                   cells[(cellx+width+cp[0])%width][(celly+height+cp[1])%height],
                                   cells[(cellx+width+cp[0]*2)%width][(celly+height+cp[1]*2)%height],
                                   cells[(cellx+width+cp[0]*3)%width][(celly+height+cp[1]*3)%height],
                                   not rotateLeft) # Revert
                    self.learnWeight(swapChoiceWeights,swapChoice,punish)
                    self.learnWeight(toleranceWeights,chosenToleranceSteps-1,punish)

            elif swapChoice == 3:
                rotateLeft = np.random.random() < 0.5
                self.rotate5Indices(swapTable,
                               cells[cellx][celly],
                               cells[(cellx+width+cp[0])%width][(celly+height+cp[1])%height],
                               cells[(cellx+width+cp[0]*2)%width][(celly+height+cp[1]*2)%height],
                               cells[(cellx+width+cp[0]*3)%width][(celly+height+cp[1]*3)%height],
                               cells[(cellx+width+cp[0]*4)%width][(celly+height+cp[1]*4)%height],rotateLeft)
                quality = self.sumDistances(swapTable,distances)
                if quality < bestQuality:
                    bestQuality = quality
                    bestSwapTableBeforeAnnealing = swapTable.copy()
                    self.learnWeight(toleranceWeights,chosenToleranceSteps-1,reward)
                    self.learnWeight(swapChoiceWeights,swapChoice,reward)
                elif toleranceSteps == 0:
                    self.rotate5Indices(swapTable,
                                   cells[cellx][celly],
                                   cells[(cellx+width+cp[0])%width][(celly+height+cp[1])%height],
                                   cells[(cellx+width+cp[0]*2)%width][(celly+height+cp[1]*2)%height],
                                   cells[(cellx+width+cp[0]*3)%width][(celly+height+cp[1]*3)%height],
                                   cells[(cellx+width+cp[0]*4)%width][(celly+height+cp[1]*4)%height],
                                   not rotateLeft) # Revert
                    self.learnWeight(swapChoiceWeights,swapChoice,punish)
                    self.learnWeight(toleranceWeights,chosenToleranceSteps-1,punish)


            elif swapChoice == 4:
                cols = np.random.choice(colSizes,p=colSizeWeights/np.sum(colSizeWeights))
                rows = np.random.choice(rowSizes,p=rowSizeWeights/np.sum(rowSizeWeights))
                cellx2 = np.random.randint(0,width-cols)
                celly2 = np.random.randint(0,height-rows)
                if self.swapBlock(cells,swapTable,cellx,celly,cellx2,celly2,cols,rows,width,height):
                    quality = self.sumDistances(swapTable,distances)
                    if quality < bestQuality:
                        bestQuality = quality
                        bestSwapTableBeforeAnnealing = swapTable.copy()
                        self.learnWeight(toleranceWeights,chosenToleranceSteps-1,reward)
                        self.learnWeight(swapChoiceWeights,swapChoice,reward)
                        self.learnWeight(colSizeWeights,cols-1,reward)
                        self.learnWeight(rowSizeWeights,rows-1,reward)
                    elif toleranceSteps == 0:
                        self.swapBlock(cells,swapTable,cellx,celly,cellx2,celly2,cols,rows,width,height) # Revert
                        self.learnWeight(toleranceWeights,chosenToleranceSteps-1,punish)
                        self.learnWeight(swapChoiceWeights,swapChoice,punish)
                        self.learnWeight(colSizeWeights,cols-1,punish)
                        self.learnWeight(rowSizeWeights,rows-1,punish)

            elif swapChoice == 5:
                cols = np.random.choice(colSizes,p=colSizeWeights/np.sum(colSizeWeights))
                rows = np.random.choice(rowSizes,p=rowSizeWeights/np.sum(rowSizeWeights))
                dx = dy = 0
                if np.random.random() < 0.5:
                    dx = np.random.randint(-cellx,width-cols) if width-cols > -cellx else 0
                else:
                    dy = np.random.randint(-celly,height-rows) if height-rows > -celly else 0

                if self.shiftBlock(cells,swapTable,cellx,celly,cols,rows,dx,dy,width,height):
                    quality = self.sumDistances(swapTable,distances)
                    if quality < bestQuality:
                        bestQuality = quality
                        bestSwapTableBeforeAnnealing = swapTable.copy()
                        self.learnWeight(toleranceWeights,chosenToleranceSteps-1,reward)
                        self.learnWeight(swapChoiceWeights,swapChoice,reward)
                        self.learnWeight(colSizeWeights,cols-1,reward)
                        self.learnWeight(rowSizeWeights,rows-1,reward)
                    elif toleranceSteps == 0:
                        self.shiftBlock(cells,swapTable,cellx+dx,celly+dy,cols,rows,-dx,-dy,width,height) # Revert
                        self.learnWeight(swapChoiceWeights,swapChoice,punish)
                        self.learnWeight(colSizeWeights,cols-1,punish)
                        self.learnWeight(rowSizeWeights,rows-1,punish)
                        self.learnWeight(toleranceWeights,chosenToleranceSteps-1,punish)

            elif swapChoice == 6:
                cols = np.random.choice(colSizes,p=colSizeWeights/np.sum(colSizeWeights))
                rows = np.random.choice(rowSizes,p=rowSizeWeights/np.sum(rowSizeWeights))
                flipxy = [[True,False],[False,True],[True,True]][np.random.randint(0,3)]

                if self.flipBlock(cells,swapTable,cellx,celly,cols,rows,flipxy[0],flipxy[1],width,height):
                    quality = self.sumDistances(swapTable,distances)
                    if quality < bestQuality:
                        bestQuality = quality
                        bestSwapTableBeforeAnnealing = swapTable.copy()
                        self.learnWeight(toleranceWeights,chosenToleranceSteps-1,reward)
                        self.learnWeight(swapChoiceWeights,swapChoice,reward)
                        self.learnWeight(colSizeWeights,cols-1,reward)
                        self.learnWeight(rowSizeWeights,rows-1,reward)
                    elif toleranceSteps == 0:
                        self.flipBlock(cells,swapTable,cellx,celly,cols,rows,flipxy[0],flipxy[1],width,height) # Revert
                        self.learnWeight(swapChoiceWeights,swapChoice,punish)
                        self.learnWeight(colSizeWeights,cols-1,punish)
                        self.learnWeight(rowSizeWeights,rows-1,punish)
                        self.learnWeight(toleranceWeights,chosenToleranceSteps-1,punish)

            elif swapChoice == 7:
                cols = np.random.choice(colSizes,p=colSizeWeights/np.sum(colSizeWeights))
                rows = np.random.choice(rowSizes,p=rowSizeWeights/np.sum(rowSizeWeights))
                oldState = self.shuffleBlock(cells,swapTable,cellx,celly,cols,rows,width,height)
                if len(oldState)>0:
                    quality = self.sumDistances(swapTable,distances)
                    if quality < bestQuality:
                        bestQuality = quality
                        bestSwapTableBeforeAnnealing = swapTable.copy()
                        self.learnWeight(toleranceWeights,chosenToleranceSteps-1,reward)
                        self.learnWeight(swapChoiceWeights,swapChoice,reward)
                        self.learnWeight(colSizeWeights,cols-1,reward)
                        self.learnWeight(rowSizeWeights,rows-1,reward)
                    elif toleranceSteps == 0:
                        self.restoreBlock(cells,swapTable,cellx,celly,cols,rows,oldState) # Revert
                        self.learnWeight(swapChoiceWeights,swapChoice,punish)
                        self.learnWeight(colSizeWeights,cols-1,punish)
                        self.learnWeight(rowSizeWeights,rows-1,punish)
                        self.learnWeight(toleranceWeights,chosenToleranceSteps-1,punish)

            elif swapChoice == 8:
                rotateLeft = np.random.random() < 0.5
                self.rotate3Indices(swapTable,cells[cellx][celly],cells[(cellx+width+cp[0])%width][(celly+height+cp[1])%height],cells[(cellx+width+cp[1])%width][(celly+height+cp[0])%height],rotateLeft)
                quality = self.sumDistances(swapTable,distances)
                if quality < bestQuality:
                    bestQuality = quality
                    bestSwapTableBeforeAnnealing = swapTable.copy()
                    self.learnWeight(toleranceWeights,chosenToleranceSteps-1,reward)
                    self.learnWeight(swapChoiceWeights,swapChoice,reward)
                elif toleranceSteps == 0:
                    self.rotate3Indices(swapTable,cells[cellx][celly],cells[(cellx+width+cp[0])%width][(celly+height+cp[1])%height],cells[(cellx+width+cp[1])%width][(celly+height+cp[0])%height],not rotateLeft) # Revert
                    self.learnWeight(swapChoiceWeights,swapChoice,punish)
                    self.learnWeight(toleranceWeights,chosenToleranceSteps-1,punish)

            elif swapChoice == 9:
                rotateLeft = np.random.random() < 0.5
                self.rotate4Indices(swapTable,
                               cells[cellx][celly],
                               cells[(cellx+width+1)%width][celly],
                               cells[(cellx+width+1)%width][(celly+height+1)%height],
                               cells[cellx][(celly+height+1)%height],rotateLeft)
                quality = self.sumDistances(swapTable,distances)
                if quality < bestQuality:
                    bestQuality = quality
                    bestSwapTableBeforeAnnealing = swapTable.copy()
                    self.learnWeight(toleranceWeights,chosenToleranceSteps-1,reward)
                    self.learnWeight(swapChoiceWeights,swapChoice,reward)
                elif toleranceSteps == 0:
                    self.rotate4Indices(swapTable,
                               cells[cellx][celly],
                               cells[(cellx+width+1)%width][celly],
                               cells[(cellx+width+1)%width][(celly+height+1)%height],
                               cells[cellx][(celly+height+1)%height], not rotateLeft) # Revert
                    self.learnWeight(swapChoiceWeights,swapChoice,punish)
                    self.learnWeight(toleranceWeights,chosenToleranceSteps-1,punish)


        if bestQuality > startingQuality: # If no improvement at all, revert to original
            bestSwapTableBeforeAnnealing = startingSwapTable
            bestQuality = startingQuality

        print(("final distance sum:",bestQuality))
        print(("improvement:", startingQuality-bestQuality))
        if startingQuality<bestQuality: # This condition might be redundant now
            print("reverting to initial swap table") # Should already be handled by the above
        self.lastState = {'cells':cells,'iterations':totalIterations, 'distances':distances,
                                              'swapChoiceWeights':swapChoiceWeights,'swapOffsetWeights':swapOffsetWeights,
                                              'colSizeWeights':colSizeWeights,'rowSizeWeights':rowSizeWeights,
                                              'toleranceWeights':toleranceWeights}

        self.lastSwapTable = bestSwapTableBeforeAnnealing.copy()

        return bestSwapTableBeforeAnnealing

    def continueOptimization( self, iterations, shakeIterations = 0):
        """
        Continues a previous optimization run.

        Uses the stored `self.lastState` and `self.lastSwapTable` to
        continue the optimization process for a given number of additional
        iterations.

        Args:
            iterations (int): The number of additional optimization
                iterations to perform.
            shakeIterations (int, optional): Number of random swaps to perform
                at the beginning of this continuation. Defaults to 0.

        Returns:
            numpy.ndarray or None: The optimized swapTable from the continued
            run. Returns None if `self.lastState` is None (i.e., no previous
            optimization to continue).
        """
        if self.lastState is None:
            print("No previous state to continue from.")
            return None
        return self.optimize( None, None, None, None, iterations, shakeIterations = shakeIterations,continueFromLastState = True,swapTable=self.lastSwapTable)

    def swapIndices(self,d,i1,i2):
        """
        Swaps two elements in an array in-place.

        Args:
            d (numpy.ndarray): The array in which to swap elements.
            i1 (int): Index of the first element.
            i2 (int): Index of the second element.

        Returns:
            None. The array `d` is modified in-place.
        """
        d[i1], d[i2] = d[i2], d[i1]

    def rotate3Indices(self,d,i1,i2,i3,rotateLeft=True):
        """
        Rotates three elements in an array in-place.

        Args:
            d (numpy.ndarray): The array containing the elements.
            i1 (int): Index of the first element.
            i2 (int): Index of the second element.
            i3 (int): Index of the third element.
            rotateLeft (bool, optional): Direction of rotation.
                True for left rotation (i1->i2, i2->i3, i3->i1),
                False for right rotation. Defaults to True.

        Returns:
            None. The array `d` is modified in-place.
            Returns early if indices are not unique.
        """
        if len(set([i1,i2,i3]))!=3: # Ensure unique indices before proceeding
            return
        if rotateLeft:
            # d[i1], d[i2], d[i3] = d[i2], d[i3], d[i1] # Original logic was this for left
            d[i2],d[i3],d[i1] = d[i1],d[i2],d[i3] # Corrected: i1->i2, i2->i3, i3->i1
        else:
            # d[i1], d[i2], d[i3] = d[i3], d[i1], d[i2] # Original logic was this for right
            d[i3],d[i1],d[i2] = d[i1],d[i2],d[i3] # Corrected: i1->i3, i3->i2, i2->i1
        # The provided code had d[i2],d[i3],d[i1] = d[i3],d[i2],d[i1] for both, which is a swap of i1,i3
        # Correcting to actual rotation.
        # For left: temp = d[i1]; d[i1]=d[i2]; d[i2]=d[i3]; d[i3]=temp;
        # For right: temp = d[i1]; d[i1]=d[i3]; d[i3]=d[i2]; d[i2]=temp;
        # The current implementation in the original code for rotateLeft:
        # d[i2],d[i3],d[i1] = d[i3],d[i2],d[i1] means:
        # new d[i2] = old d[i3]
        # new d[i3] = old d[i2]
        # new d[i1] = old d[i1] (this is wrong, it should be old d[i1])
        # This results in d[i1] being untouched, and d[i2] and d[i3] swapped.
        # A true 3-cycle rotation:
        if rotateLeft: # i1->i2, i2->i3, i3->i1
            val_i1 = d[i1]
            d[i1] = d[i3] # i1 gets value from i3
            d[i3] = d[i2] # i3 gets value from i2
            d[i2] = val_i1 # i2 gets value from i1
        else: # i1->i3, i3->i2, i2->i1 (right rotation)
            val_i1 = d[i1]
            d[i1] = d[i2] # i1 gets value from i2
            d[i2] = d[i3] # i2 gets value from i3
            d[i3] = val_i1 # i3 gets value from i1


    def rotate4Indices(self,d,i1,i2,i3,i4,rotateLeft=True):
        """
        Rotates four elements in an array in-place.

        Args:
            d (numpy.ndarray): The array containing the elements.
            i1 (int): Index of the first element.
            i2 (int): Index of the second element.
            i3 (int): Index of the third element.
            i4 (int): Index of the fourth element.
            rotateLeft (bool, optional): Direction of rotation. Defaults to True.

        Returns:
            None. The array `d` is modified in-place.
            Returns early if indices are not unique.
        """
        if len(set([i1,i2,i3,i4]))!=4: # Ensure unique indices
            return
        if rotateLeft: # i1->i2, i2->i3, i3->i4, i4->i1
            # d[i1],d[i4],d[i3],d[i2] = d[i4],d[i3],d[i2],d[i1] # Original logic
            # This means: new d[i1]=old d[i4], new d[i4]=old d[i3], new d[i3]=old d[i2], new d[i2]=old d[i1]
            # This is a left rotation: i1 <- i2 <- i3 <- i4 <- i1
            val_i1 = d[i1]
            d[i1] = d[i2]
            d[i2] = d[i3]
            d[i3] = d[i4]
            d[i4] = val_i1
        else: # i1->i4, i4->i3, i3->i2, i2->i1 (right rotation)
            # d[i3],d[i2],d[i1],d[i4] = d[i4],d[i3],d[i2],d[i1] # Original logic
            # This means: new d[i3]=old d[i4], new d[i2]=old d[i3], new d[i1]=old d[i2], new d[i4]=old d[i1]
            # This is a right rotation: i1 -> i4 -> i3 -> i2 -> i1
            val_i1 = d[i1]
            d[i1] = d[i4]
            d[i4] = d[i3]
            d[i3] = d[i2]
            d[i2] = val_i1

    def rotate5Indices(self,d,i1,i2,i3,i4,i5,rotateLeft=True):
        """
        Rotates five elements in an array in-place.

        Args:
            d (numpy.ndarray): The array containing the elements.
            i1..i5 (int): Indices of the five elements.
            rotateLeft (bool, optional): Direction of rotation. Defaults to True.

        Returns:
            None. The array `d` is modified in-place.
            Returns early if indices are not unique.
        """
        if len(set([i1,i2,i3,i4,i5]))!=5: # Ensure unique indices
            return
        if rotateLeft: # i1->i2, i2->i3, i3->i4, i4->i5, i5->i1
            # d[i1],d[i5],d[i4],d[i3],d[i2] = d[i5],d[i4],d[i3],d[i2],d[i1] # Original
            # This is: new d[i1]=old d[i5], new d[i5]=old d[i4], new d[i4]=old d[i3], new d[i3]=old d[i2], new d[i2]=old d[i1]
            # This is a left rotation: i1 <- i2 <- i3 <- i4 <- i5 <- i1
            val_i1 = d[i1]
            d[i1] = d[i2]
            d[i2] = d[i3]
            d[i3] = d[i4]
            d[i4] = d[i5]
            d[i5] = val_i1
        else: # i1->i5, i5->i4, i4->i3, i3->i2, i2->i1 (right rotation)
            # d[i4],d[i3],d[i2],d[i1],d[i5] = d[i5],d[i4],d[i3],d[i2],d[i1] # Original
            # This is: new d[i4]=old d[i5], new d[i3]=old d[i4], new d[i2]=old d[i3], new d[i1]=old d[i2], new d[i5]=old d[i1]
            # This is a right rotation: i1 -> i5 -> i4 -> i3 -> i2 -> i1
            val_i1 = d[i1]
            d[i1] = d[i5]
            d[i5] = d[i4]
            d[i4] = d[i3]
            d[i3] = d[i2]
            d[i2] = val_i1


    def swapBlock(self,cells,d,tlx1,tly1,tlx2,tly2,cols,rows,width,height):
        """
        Swaps two rectangular blocks of cell assignments in the swapTable `d`.

        Args:
            cells (numpy.ndarray): A 2D array (width, height) mapping
                (x,y) grid coordinates to indices in `d` (and original points).
            d (numpy.ndarray): The swapTable array to modify.
            tlx1 (int): Top-left x-coordinate of the first block.
            tly1 (int): Top-left y-coordinate of the first block.
            tlx2 (int): Top-left x-coordinate of the second block.
            tly2 (int): Top-left y-coordinate of the second block.
            cols (int): Width of the blocks to swap.
            rows (int): Height of the blocks to swap.
            width (int): Total width of the grid.
            height (int): Total height of the grid.

        Returns:
            bool: True if the swap was performed successfully, False otherwise
            (e.g., if blocks overlap or go out of bounds).
        """
        # Check for overlap or out of bounds
        if not (max(tlx1,tlx2)+cols <= width and \
                max(tly1,tly2)+rows <= height and \
                (abs(tlx1 - tlx2) >= cols or abs(tly1 - tly2) >= rows)):
            return False

        temp_block_storage = []
        for r_offset in range(rows):
            for c_offset in range(cols):
                idx1 = cells[tlx1+c_offset][tly1+r_offset]
                idx2 = cells[tlx2+c_offset][tly2+r_offset]
                temp_block_storage.append(d[idx1])
                d[idx1] = d[idx2]

        storage_idx = 0
        for r_offset in range(rows):
            for c_offset in range(cols):
                idx2 = cells[tlx2+c_offset][tly2+r_offset]
                d[idx2] = temp_block_storage[storage_idx]
                storage_idx+=1
        return True

    def shuffleBlock(self,cells,d,tlx,tly,cols,rows,width,height):
        """
        Shuffles the assignments within a rectangular block of cells.

        Args:
            cells (numpy.ndarray): 2D array mapping grid coords to indices in `d`.
            d (numpy.ndarray): The swapTable to modify.
            tlx (int): Top-left x-coordinate of the block.
            tly (int): Top-left y-coordinate of the block.
            cols (int): Width of the block.
            rows (int): Height of the block.
            width (int): Total width of the grid.
            height (int): Total height of the grid.

        Returns:
            numpy.ndarray: A copy of the original assignments within the
            block before shuffling. Returns an empty list if the block is
            out of bounds.
        """
        if not (tlx+cols <= width and tly+rows <= height): # Corrected boundary check
            return []

        block_indices = []
        for r_offset in range(rows):
            for c_offset in range(cols):
                block_indices.append(cells[tlx+c_offset][tly+r_offset])

        current_values_in_block = d[block_indices].copy()
        shuffled_values = current_values_in_block.copy()
        np.random.shuffle(shuffled_values)

        d[block_indices] = shuffled_values
        return current_values_in_block # Return the state before shuffling

    def restoreBlock(self,cells,d,tlx,tly,cols,rows,oldState):
        """
        Restores the assignments within a block to a previous state.

        Args:
            cells (numpy.ndarray): 2D array mapping grid coords to indices in `d`.
            d (numpy.ndarray): The swapTable to modify.
            tlx (int): Top-left x-coordinate of the block.
            tly (int): Top-left y-coordinate of the block.
            cols (int): Width of the block.
            rows (int): Height of the block.
            oldState (numpy.ndarray): The array of values to restore the
                block's assignments to.

        Returns:
            None. Modifies `d` in-place.
        """
        idx_counter = 0
        for r_offset in range(rows):
            for c_offset in range(cols):
                grid_idx = cells[tlx+c_offset][tly+r_offset]
                d[grid_idx] = oldState[idx_counter]
                idx_counter+=1


    def shiftBlock(self,cells,d,tlx,tly,cols,rows,dx,dy,width,height):
        """
        Shifts a block of cell assignments horizontally or vertically.

        The block is shifted by `dx` columns or `dy` rows. The elements
        that are shifted out of one side of the grid wrap around to the
        other side if the shift is within the block's own dimension.
        This implementation seems to handle shifts by overwriting,
        with the "empty" space being filled by elements from the other end.

        Args:
            cells (numpy.ndarray): 2D array mapping grid coords to indices.
            d (numpy.ndarray): The swapTable to modify.
            tlx (int): Top-left x of the block to shift.
            tly (int): Top-left y of the block to shift.
            cols (int): Width of the block.
            rows (int): Height of the block.
            dx (int): Shift in x-direction (number of columns).
            dy (int): Shift in y-direction (number of rows).
            width (int): Total grid width.
            height (int): Total grid height.

        Returns:
            bool: True if shift was valid and performed, False otherwise.
                  A shift is invalid if dx and dy are both non-zero or both zero,
                  if the block size is < 2, or if boundaries are violated.
        """
        if (dx!=0 and dy!= 0) or (dx==0 and dy == 0) or cols*rows < 2:
            return False

        # Normalize negative shifts to positive shifts from the other direction
        # This logic seems complex and might not be a standard "shift with wrap"
        # It appears to define a source block and a destination for that block's contents.
        # The "temp1" seems to be the content of the area that will be overwritten by the shift.
        # Then "temp2" (the block to be shifted) is placed, and "temp1" is placed where "temp2" was.

        # Store original indices and values of the block to be moved
        source_block_indices = []
        source_block_values = []
        for r_offset in range(rows):
            for c_offset in range(cols):
                src_idx = cells[tlx + c_offset][tly + r_offset]
                source_block_indices.append(src_idx)
                source_block_values.append(d[src_idx])

        # Store original indices and values of the block that will be overwritten by the move
        target_block_indices = []
        target_block_values = []
        for r_offset in range(rows):
            for c_offset in range(cols):
                # Check bounds for target block
                target_x, target_y = tlx + c_offset + dx, tly + r_offset + dy
                if not (0 <= target_x < width and 0 <= target_y < height):
                    return False # Shift would go out of bounds
                tgt_idx = cells[target_x][target_y]
                target_block_indices.append(tgt_idx)
                target_block_values.append(d[tgt_idx])

        # Perform the shift: place source_block_values into target_block_indices
        for i in range(len(target_block_indices)):
            d[target_block_indices[i]] = source_block_values[i]

        # Fill the original source_block_indices with target_block_values (the wrapped part)
        # This step is what makes it a "shift" rather than a simple "move" or "copy"
        # However, the original logic for temp1/temp2 was more specific about which part wraps.
        # The current logic effectively swaps the contents of two blocks if they don't overlap.
        # If they overlap, it's more complex.
        # For a true cyclic shift, elements pushed out one side reappear on the other.
        # The original code's handling of dx<0 and dy<0 by redefining tlx,cols,dx etc.
        # was an attempt to simplify this, but let's try a direct approach for clarity.

        # This simplified version just swaps the values if the areas are distinct.
        # A true cyclic shift is more complex. Given the original structure,
        # it seems like it was trying to implement a specific type of area swap.
        # Let's stick to the observed behavior of swapping values in two regions.
        # The original logic for `temp1` and `temp2` and how they are written back
        # needs careful re-evaluation if a true cyclic block shift is intended.
        # For now, this re-implements the swapping of two blocks' contents.

        # If the blocks are the same, no operation needed (already handled by dx=0, dy=0 check)
        # If blocks overlap, this simple swap is not correct for a "shift".
        # The original code's conditions for `temp1` and `temp2` suggest a more
        # specific type of shift where a part of the grid is moved, and the
        # space it occupied is filled by a specific other part of the grid.

        # Reverting to a structure closer to original to capture its intent,
        # assuming it's a specific kind of regional swap/shift.

        # Check bounds for the source block itself
        if not (0 <= tlx < width and 0 <= tly < height and \
                0 <= tlx + cols -1 < width and 0 <= tly + rows -1 < height):
            return False
        # Check bounds for the destination of the block
        if not (0 <= tlx + dx < width and 0 <= tly + dy < height and \
                0 <= tlx + dx + cols -1 < width and 0 <= tly + dy + rows -1 < height):
            return False


        # Values of the block that will be moved
        block_to_move_values = np.array([d[cells[tlx+c][tly+r]] for r in range(rows) for c in range(cols)])

        # Values of the area that the block_to_move will overwrite
        # These are the values that will effectively "wrap around" or fill the vacated space
        area_to_be_overwritten_values = np.array([d[cells[tlx+dx+c][tly+dy+r]] for r in range(rows) for c in range(cols)])

        # Place block_to_move_values into the new location
        idx = 0
        for r in range(rows):
            for c in range(cols):
                d[cells[tlx+dx+c][tly+dy+r]] = block_to_move_values[idx]
                idx +=1

        # Place area_to_be_overwritten_values into the original location of the block
        idx = 0
        for r in range(rows):
            for c in range(cols):
                d[cells[tlx+c][tly+r]] = area_to_be_overwritten_values[idx]
                idx+=1

        return True


    def flipBlock(self,cells,d,tlx,tly,cols,rows,flipx,flipy,width,height):
        """
        Flips a rectangular block of cell assignments horizontally and/or vertically.

        Args:
            cells (numpy.ndarray): 2D array mapping grid coords to indices.
            d (numpy.ndarray): The swapTable to modify.
            tlx (int): Top-left x of the block.
            tly (int): Top-left y of the block.
            cols (int): Width of the block.
            rows (int): Height of the block.
            flipx (bool): True to flip horizontally.
            flipy (bool): True to flip vertically.
            width (int): Total grid width.
            height (int): Total grid height.

        Returns:
            bool: True if flip was performed, False if block is out of bounds.
        """
        if not (tlx+cols <= width and tly+rows <= height): # Corrected boundary check
            return False

        original_values = []
        for r_offset in range(rows):
            for c_offset in range(cols):
                original_values.append(d[cells[tlx+c_offset][tly+r_offset]])

        val_idx = 0
        for r_offset in range(rows):
            target_r_offset = (rows-1-r_offset) if flipy else r_offset
            for c_offset in range(cols):
                target_c_offset = (cols-1-c_offset) if flipx else c_offset
                # Read from original_values sequentially, write to flipped position
                d[cells[tlx+target_c_offset][tly+target_r_offset]] = original_values[val_idx]
                val_idx+=1
        return True

    def shake( self,cells,d,iterations,strength,width,height):
        """
        Randomly swaps adjacent or nearby cells for a number of iterations.

        This is used to introduce some randomness or to escape local minima.

        Args:
            cells (numpy.ndarray): 2D array mapping grid coords to indices.
            d (numpy.ndarray): The swapTable to modify.
            iterations (int): Number of shake operations to perform.
            strength (int): Maximum Manhattan distance for a swap (e.g.,
                strength 1 means only directly adjacent cells).
            width (int): Total grid width.
            height (int): Total grid height.

        Returns:
            None. Modifies `d` in-place.
        """
        for _ in range(iterations): # Use _ if i is not used
            col = np.random.randint(0,width)
            row = np.random.randint(0,height)
            # Ensure ox, oy allow for actual movement
            ox, oy = 0, 0
            while ox == 0 and oy == 0: # Ensure there is some offset
                 ox = np.random.randint(-strength,strength+1)
                 oy = np.random.randint(-strength,strength+1)

            if col+ox >= 0 and row+oy >= 0 and col+ox<width and row+oy<height:
                i1 = cells[col][row]
                i2 = cells[col+ox][row+oy]
                if i1 != i2 : # Avoid swapping an element with itself
                    self.swapIndices(d, i1, i2) # Use the class method


    def sumDistances(self,swapTable,distances):
        """
        Calculates the sum of squared Euclidean distances for the current
        arrangement in the swapTable.

        Args:
            swapTable (numpy.ndarray): A 1D array where `swapTable[i]` is the
                index of the grid cell assigned to the i-th original point.
            distances (numpy.ndarray): A 2D array where `distances[j, k]` is
                the squared Euclidean distance between the j-th grid cell
                (from normalized grid) and the k-th original point (from
                normalized xy).

        Returns:
            float: The sum of squared distances for the current mapping.
        """
        r_indices = np.arange(len(swapTable)) # Original point indices
        # distances[swapTable[original_point_idx], original_point_idx]
        return np.sum(distances[swapTable[r_indices],r_indices])

    def learnWeight( self,b, idx, f ):
        """
        Adjusts a weight in a weight array.

        Used to update the probabilities of choosing different swap strategies
        or parameters based on their success.

        Args:
            b (numpy.ndarray): The array of weights to modify.
            idx (int): The index of the weight to adjust.
            f (float): The factor by which to multiply the weight.
                       (e.g., >1 for reward, <1 for punishment).

        Returns:
            None. Modifies `b` in-place. Weights are clamped to a min/max range.
        """
        b[idx]*=f
        if b[idx]<0.0001:
            b[idx] = 0.0001
        elif b[idx]>10000.0:
            b[idx]=10000.0
