#
# Random Grid Swap Utility v1.0
# part of Raster Fairy v1.1.0,
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
from scipy import spatial, ndimage
import math
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
    
    def __init__(self):
        self.lastState = None
        self.grid_norm = None
        self.xy_norm = None
        self.lastSwapTable = None
        self.lastWidth = 0
        self.lastHeight = 0
        self.strategy_success_history = {}
        self.recent_improvements = []
        self.stagnation_counter = 0

    def optimize( self, xy, grid, width, height, iterations, shakeIterations = 0, swapTable=None, continueFromLastState = False):
        """
        Main optimization loop with enhanced learning and adaptive strategies (optimized version).
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
        else: 
            width = self.lastWidth
            height = self.lastHeight
            if self.grid_norm is None or self.xy_norm is None:
                raise ValueError("Optimizer state is incomplete for continuation.")

        reward = 1.2
        punish = 0.99

        totalEntries = len(self.grid_norm)

        choiceModes = np.arange(10)
        colSizes = np.arange(1,max(2,width))
        rowSizes = np.arange(1,max(2,height))
        swapOffsets = np.arange(1,9)
        offsets = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[1,-1],[-1,-1]])
        toleranceChoices = np.arange(1,16)

        if swapTable is None:
            swapTable =  np.arange(totalEntries)

        if self.lastState is None or not continueFromLastState:
            cells = np.zeros((width, height),dtype=int)
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
                return None

            distances = dist.cdist(self.grid_norm, self.xy_norm, 'sqeuclidean')
            swapChoiceWeights = np.array([1.0] * len(choiceModes),dtype=np.float64)
            colSizeWeights = np.array([1.0] * len(colSizes),dtype=np.float64)
            rowSizeWeights = np.array([1.0] * len(rowSizes),dtype=np.float64)
            swapOffsetWeights = np.array([1.0] * len(swapOffsets),dtype=np.float64)
            toleranceWeights = np.array([1.0] * len(toleranceChoices),dtype=np.float64)
            totalIterations = iterations
            
            # OPTIMIZED: Reduce update frequencies and pre-compute expensive operations
            heat_map_update_frequency = max(500, iterations // 20)
            disorder_check_frequency = max(200, iterations // 50)
            smart_learning_frequency = max(100, iterations // 100)
            
            heat_map = None
            cached_disorder_level = 0.5
            last_quality_for_disorder = 0
            
            # Pre-compute some expensive values
            min_distances_per_point = np.min(distances, axis=0)
            theoretical_min_quality = np.sum(min_distances_per_point)
            
            # Initialize strategy success tracking
            if not hasattr(self, 'strategy_success_history'):
                self.strategy_success_history = {}
                
        else: 
            cells = self.lastState['cells']
            distances = self.lastState['distances']
            swapChoiceWeights = self.lastState['swapChoiceWeights']
            colSizeWeights = self.lastState['colSizeWeights']
            rowSizeWeights = self.lastState['rowSizeWeights']
            swapOffsetWeights =self.lastState['swapOffsetWeights']
            toleranceWeights = self.lastState['toleranceWeights']
            totalIterations = self.lastState['iterations']+iterations
            
            # Restore optimized variables
            heat_map_update_frequency = self.lastState.get('heat_map_update_frequency', max(500, iterations // 20))
            disorder_check_frequency = max(200, iterations // 50)
            smart_learning_frequency = max(100, iterations // 100)
            heat_map = None
            cached_disorder_level = self.lastState.get('cached_disorder_level', 0.5)
            
            min_distances_per_point = np.min(distances, axis=0)
            theoretical_min_quality = np.sum(min_distances_per_point)

        bestQuality = startingQuality = self.sumDistances(swapTable,distances)
        startingSwapTable = swapTable.copy()
        last_quality_for_disorder = bestQuality
        print(("Starting sum of distances",startingQuality))
        
        if shakeIterations > 0:
            self.shake(cells,swapTable,shakeIterations,1,width,height)
            bestQuality = self.sumDistances(swapTable,distances)
            print(("After shake sum of distances",bestQuality))

        bestSwapTableBeforeAnnealing = swapTable.copy()
        toleranceSteps = 0

        for i in range(iterations):
            if i>0 and i % 20000 == 0:
                print(("Starting sum of distances",startingQuality))
                
            if i % 1000 == 0:
                print((i,bestQuality))

            current_quality = self.sumDistances(swapTable, distances)
            
            # Update disorder level only periodically
            if i % disorder_check_frequency == 0:
                cached_disorder_level = self.measureDisorderFast(current_quality, theoretical_min_quality, startingQuality)
                last_quality_for_disorder = current_quality
                print(f"Disorder Level: {cached_disorder_level}")

            # Update heat map less frequently
            if i % heat_map_update_frequency == 0:
                heat_map = self.buildHeatMapFast(swapTable, distances, width, height)
            
            # Create context with cached values
            current_context = {'disorder': cached_disorder_level, 'iteration': i}
            
            # Use smart weights only periodically, otherwise use simple adaptive weights
            if i % smart_learning_frequency == 0:
                base_adaptive_weights = self.getAdaptiveWeights(swapChoiceWeights, cached_disorder_level, 'swapChoice')
                smart_swap_weights = self.getSmartWeights(base_adaptive_weights, current_context)
                adaptive_col_weights = self.getAdaptiveWeights(colSizeWeights, cached_disorder_level, 'blockSize')
                adaptive_row_weights = self.getAdaptiveWeights(rowSizeWeights, cached_disorder_level, 'blockSize')
            else:
                # Use simpler adaptive weights most of the time
                smart_swap_weights = self.getAdaptiveWeights(swapChoiceWeights, cached_disorder_level, 'swapChoice')
                adaptive_col_weights = colSizeWeights
                adaptive_row_weights = rowSizeWeights

            if toleranceSteps == 0:
                swapTable = bestSwapTableBeforeAnnealing.copy()
                chosenToleranceSteps = toleranceSteps = np.random.choice(toleranceChoices,p=toleranceWeights/np.sum(toleranceWeights))

            toleranceSteps -= 1

            if heat_map is not None and i % 3 == 0:
                cellx, celly = self.selectHotCellFast(heat_map, width, height)
            else:
                cellx = np.random.randint(0,width)
                celly = np.random.randint(0,height)

            cp = offsets.copy()[np.random.randint(0,8)]

            # Use the (possibly simplified) smart weights
            swapChoice = np.random.choice(choiceModes, p=smart_swap_weights/np.sum(smart_swap_weights))
            
            # Store quality before operation for learning (but only track every few iterations)
            quality_before_operation = current_quality
            should_track = (i % 10 == 0)
            
            if swapChoice == 0:
                offsetx = np.random.choice(swapOffsets,p=swapOffsetWeights/np.sum(swapOffsetWeights))
                offsety = np.random.choice(swapOffsets,p=swapOffsetWeights/np.sum(swapOffsetWeights))
                cp[0] *= offsetx
                cp[1] *= offsety
                self.swapIndices(swapTable,cells[cellx][celly],cells[(cellx+width+cp[0])%width][(celly+height+cp[1])%height])
                quality = self.sumDistances(swapTable,distances)
                
                if should_track:
                    improvement = quality_before_operation - quality if quality < quality_before_operation else 0
                    self.trackStrategySuccess(swapChoice, improvement, current_context)
                
                if quality < bestQuality:
                    bestQuality = quality
                    bestSwapTableBeforeAnnealing = swapTable.copy()
                    self.learnWeight(toleranceWeights,chosenToleranceSteps-1,reward)
                    self.learnWeight(swapChoiceWeights,swapChoice,reward)
                    self.learnWeight(swapOffsetWeights,offsetx-1,reward)
                    self.learnWeight(swapOffsetWeights,offsety-1,reward)
                elif toleranceSteps == 0:
                    self.swapIndices(swapTable,cells[cellx][celly],cells[(cellx+width+cp[0])%width][(celly+height+cp[1])%height])
                    self.learnWeight(toleranceWeights,chosenToleranceSteps-1,punish)
                    self.learnWeight(swapChoiceWeights,swapChoice,punish)
                    self.learnWeight(swapOffsetWeights,offsetx-1,punish)
                    self.learnWeight(swapOffsetWeights,offsety-1,punish)

            elif swapChoice == 1:
                rotateLeft = np.random.random() < 0.5
                self.rotate3Indices(swapTable,cells[cellx][celly],cells[(cellx+width+cp[0])%width][(celly+height+cp[1])%height],cells[(cellx+width+cp[0]*2)%width][(celly+height+cp[1]*2)%height],rotateLeft)
                quality = self.sumDistances(swapTable,distances)
                
                if should_track:
                    improvement = quality_before_operation - quality if quality < quality_before_operation else 0
                    self.trackStrategySuccess(swapChoice, improvement, current_context)
                
                if quality < bestQuality:
                    bestQuality = quality
                    bestSwapTableBeforeAnnealing = swapTable.copy()
                    self.learnWeight(toleranceWeights,chosenToleranceSteps-1,reward)
                    self.learnWeight(swapChoiceWeights,swapChoice,reward)
                elif toleranceSteps == 0:
                    self.rotate3Indices(swapTable,cells[cellx][celly],cells[(cellx+width+cp[0])%width][(celly+height+cp[1])%height],cells[(cellx+width+cp[0]*2)%width][(celly+height+cp[1]*2)%height],not rotateLeft)
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
                
                if should_track:
                    improvement = quality_before_operation - quality if quality < quality_before_operation else 0
                    self.trackStrategySuccess(swapChoice, improvement, current_context)
                
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
                                   not rotateLeft)
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
                
                if should_track:
                    improvement = quality_before_operation - quality if quality < quality_before_operation else 0
                    self.trackStrategySuccess(swapChoice, improvement, current_context)
                
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
                                   not rotateLeft)
                    self.learnWeight(swapChoiceWeights,swapChoice,punish)
                    self.learnWeight(toleranceWeights,chosenToleranceSteps-1,punish)

            elif swapChoice == 4:
                cols = np.random.choice(colSizes, p=adaptive_col_weights/np.sum(adaptive_col_weights))
                rows = np.random.choice(rowSizes, p=adaptive_row_weights/np.sum(adaptive_row_weights))
                cellx2 = np.random.randint(0,width-cols)
                celly2 = np.random.randint(0,height-rows)
                if self.swapBlock(cells,swapTable,cellx,celly,cellx2,celly2,cols,rows,width,height):
                    quality = self.sumDistances(swapTable,distances)
                    
                    if should_track:
                        improvement = quality_before_operation - quality if quality < quality_before_operation else 0
                        self.trackStrategySuccess(swapChoice, improvement, current_context)
                    
                    if quality < bestQuality:
                        bestQuality = quality
                        bestSwapTableBeforeAnnealing = swapTable.copy()
                        self.learnWeight(toleranceWeights,chosenToleranceSteps-1,reward)
                        self.learnWeight(swapChoiceWeights,swapChoice,reward)
                        self.learnWeight(colSizeWeights,cols-1,reward)
                        self.learnWeight(rowSizeWeights,rows-1,reward)
                    elif toleranceSteps == 0:
                        self.swapBlock(cells,swapTable,cellx,celly,cellx2,celly2,cols,rows,width,height)
                        self.learnWeight(toleranceWeights,chosenToleranceSteps-1,punish)
                        self.learnWeight(swapChoiceWeights,swapChoice,punish)
                        self.learnWeight(colSizeWeights,cols-1,punish)
                        self.learnWeight(rowSizeWeights,rows-1,punish)

            elif swapChoice == 5:
                cols = np.random.choice(colSizes, p=adaptive_col_weights/np.sum(adaptive_col_weights))
                rows = np.random.choice(rowSizes, p=adaptive_row_weights/np.sum(adaptive_row_weights))
                dx = dy = 0
                if np.random.random() < 0.5:
                    dx = np.random.randint(-cellx,width-cellx-cols) if width-cellx-cols > -cellx else 0
                else:
                    dy = np.random.randint(-celly,height-celly-rows) if height-celly-rows > -celly else 0

                if self.shiftBlock(cells,swapTable,cellx,celly,cols,rows,dx,dy,width,height):
                    quality = self.sumDistances(swapTable,distances)
                    
                    if should_track:
                        improvement = quality_before_operation - quality if quality < quality_before_operation else 0
                        self.trackStrategySuccess(swapChoice, improvement, current_context)
                    
                    if quality < bestQuality:
                        bestQuality = quality
                        bestSwapTableBeforeAnnealing = swapTable.copy()
                        self.learnWeight(toleranceWeights,chosenToleranceSteps-1,reward)
                        self.learnWeight(swapChoiceWeights,swapChoice,reward)
                        self.learnWeight(colSizeWeights,cols-1,reward)
                        self.learnWeight(rowSizeWeights,rows-1,reward)
                    elif toleranceSteps == 0:
                        self.shiftBlock(cells,swapTable,cellx+dx,celly+dy,cols,rows,-dx,-dy,width,height)
                        self.learnWeight(swapChoiceWeights,swapChoice,punish)
                        self.learnWeight(colSizeWeights,cols-1,punish)
                        self.learnWeight(rowSizeWeights,rows-1,punish)
                        self.learnWeight(toleranceWeights,chosenToleranceSteps-1,punish)

            elif swapChoice == 6:
                cols = np.random.choice(colSizes, p=adaptive_col_weights/np.sum(adaptive_col_weights))
                rows = np.random.choice(rowSizes, p=adaptive_row_weights/np.sum(adaptive_row_weights))
                flipxy = [[True,False],[False,True],[True,True]][np.random.randint(0,3)]

                if self.flipBlock(cells,swapTable,cellx,celly,cols,rows,flipxy[0],flipxy[1],width,height):
                    quality = self.sumDistances(swapTable,distances)
                    
                    if should_track:
                        improvement = quality_before_operation - quality if quality < quality_before_operation else 0
                        self.trackStrategySuccess(swapChoice, improvement, current_context)
                    
                    if quality < bestQuality:
                        bestQuality = quality
                        bestSwapTableBeforeAnnealing = swapTable.copy()
                        self.learnWeight(toleranceWeights,chosenToleranceSteps-1,reward)
                        self.learnWeight(swapChoiceWeights,swapChoice,reward)
                        self.learnWeight(colSizeWeights,cols-1,reward)
                        self.learnWeight(rowSizeWeights,rows-1,reward)
                    elif toleranceSteps == 0:
                        self.flipBlock(cells,swapTable,cellx,celly,cols,rows,flipxy[0],flipxy[1],width,height)
                        self.learnWeight(swapChoiceWeights,swapChoice,punish)
                        self.learnWeight(colSizeWeights,cols-1,punish)
                        self.learnWeight(rowSizeWeights,rows-1,punish)
                        self.learnWeight(toleranceWeights,chosenToleranceSteps-1,punish)

            elif swapChoice == 7:
                cols = np.random.choice(colSizes, p=adaptive_col_weights/np.sum(adaptive_col_weights))
                rows = np.random.choice(rowSizes, p=adaptive_row_weights/np.sum(adaptive_row_weights))
                oldState = self.shuffleBlock(cells,swapTable,cellx,celly,cols,rows,width,height)
                if len(oldState)>0:
                    quality = self.sumDistances(swapTable,distances)
                    
                    if should_track:
                        improvement = quality_before_operation - quality if quality < quality_before_operation else 0
                        self.trackStrategySuccess(swapChoice, improvement, current_context)
                    
                    if quality < bestQuality:
                        bestQuality = quality
                        bestSwapTableBeforeAnnealing = swapTable.copy()
                        self.learnWeight(toleranceWeights,chosenToleranceSteps-1,reward)
                        self.learnWeight(swapChoiceWeights,swapChoice,reward)
                        self.learnWeight(colSizeWeights,cols-1,reward)
                        self.learnWeight(rowSizeWeights,rows-1,reward)
                    elif toleranceSteps == 0 and len(oldState) > 0:
                        self.restoreBlock(cells,swapTable,cellx,celly,cols,rows,oldState)
                        self.learnWeight(swapChoiceWeights,swapChoice,punish)
                        self.learnWeight(colSizeWeights,cols-1,punish)
                        self.learnWeight(rowSizeWeights,rows-1,punish)
                        self.learnWeight(toleranceWeights,chosenToleranceSteps-1,punish)
                        
            elif swapChoice == 8:
                rotateLeft = np.random.random() < 0.5
                self.rotate3Indices(swapTable,cells[cellx][celly],cells[(cellx+width+cp[0])%width][(celly+height+cp[1])%height],cells[(cellx+width+cp[1])%width][(celly+height+cp[0])%height],rotateLeft)
                quality = self.sumDistances(swapTable,distances)
                
                if should_track:
                    improvement = quality_before_operation - quality if quality < quality_before_operation else 0
                    self.trackStrategySuccess(swapChoice, improvement, current_context)
                
                if quality < bestQuality:
                    bestQuality = quality
                    bestSwapTableBeforeAnnealing = swapTable.copy()
                    self.learnWeight(toleranceWeights,chosenToleranceSteps-1,reward)
                    self.learnWeight(swapChoiceWeights,swapChoice,reward)
                elif toleranceSteps == 0:
                    self.rotate3Indices(swapTable,cells[cellx][celly],cells[(cellx+width+cp[0])%width][(celly+height+cp[1])%height],cells[(cellx+width+cp[1])%width][(celly+height+cp[0])%height],not rotateLeft)
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
                
                if should_track:
                    improvement = quality_before_operation - quality if quality < quality_before_operation else 0
                    self.trackStrategySuccess(swapChoice, improvement, current_context)
                
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
                               cells[cellx][(celly+height+1)%height], not rotateLeft)
                    self.learnWeight(swapChoiceWeights,swapChoice,punish)
                    self.learnWeight(toleranceWeights,chosenToleranceSteps-1,punish)

        if bestQuality > startingQuality:
            bestSwapTableBeforeAnnealing = startingSwapTable
            bestQuality = startingQuality

        print(("final distance sum:",bestQuality))
        print(("improvement:", startingQuality-bestQuality))
       
        self.lastState = {'cells':cells,'iterations':totalIterations, 'distances':distances,
                          'swapChoiceWeights':swapChoiceWeights,'swapOffsetWeights':swapOffsetWeights,
                          'colSizeWeights':colSizeWeights,'rowSizeWeights':rowSizeWeights,
                          'toleranceWeights':toleranceWeights,
                          'heat_map_update_frequency': heat_map_update_frequency,
                          'cached_disorder_level': cached_disorder_level}

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
            val_i1 = d[i1]
            d[i1] = d[i2]
            d[i2] = d[i3]
            d[i3] = val_i1
           
        else: 
            val_i1 = d[i1]
            d[i1] = d[i3]
            d[i3] = d[i2]
            d[i2] = val_i1
        

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
        if rotateLeft: 
            val_i1 = d[i1]
            d[i1] = d[i2]
            d[i2] = d[i3]
            d[i3] = d[i4]
            d[i4] = val_i1
        else: 
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
        if rotateLeft: 
            val_i1 = d[i1]
            d[i1] = d[i2]
            d[i2] = d[i3]
            d[i3] = d[i4]
            d[i4] = d[i5]
            d[i5] = val_i1
        else: 
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
        Shifts a rectangular block of cell assignments horizontally or vertically.
        """
        if (dx!=0 and dy!= 0) or (dx==0 and dy == 0) or cols*rows < 2:
            return False

        # Check bounds for the source block itself
        if not (0 <= tlx and 0 <= tly and \
                tlx + cols <= width and tly + rows <= height):
            return False
        
        # Check bounds for the destination of the block
        if not (0 <= tlx + dx and 0 <= tly + dy and \
                tlx + dx + cols <= width and tly + dy + rows <= height):
            return False

        # NEW: Check for overlap between source and destination
        # Source block: (tlx, tly) to (tlx+cols-1, tly+rows-1)
        # Dest block: (tlx+dx, tly+dy) to (tlx+dx+cols-1, tly+dy+rows-1)
        if not (tlx + cols <= tlx + dx or tlx + dx + cols <= tlx or \
                tly + rows <= tly + dy or tly + dy + rows <= tly):
            # Blocks overlap, which would cause data corruption
            return False

        # Values of the block that will be moved
        block_to_move_values = []
        for r in range(rows):
            for c in range(cols):
                block_to_move_values.append(d[cells[tlx+c][tly+r]])

        # Values of the area that the block_to_move will overwrite
        area_to_be_overwritten_values = []
        for r in range(rows):
            for c in range(cols):
                area_to_be_overwritten_values.append(d[cells[tlx+dx+c][tly+dy+r]])

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
            while ox == 0 and oy == 0 and strength>0: # Ensure there is some offset
                 ox = np.random.randint(-strength,strength+1)
                 oy = np.random.randint(-strength,strength+1)

            if col+ox >= 0 and row+oy >= 0 and col+ox<width and row+oy<height:
                i1 = cells[col][row]
                i2 = cells[col+ox][row+oy]
                if i1 != i2 : # Avoid swapping an element with itself
                    self.swapIndices(d, i1, i2) # Use the class method


    def sumDistances(self,swapTable, distances):
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
            
    
    
    def measureDisorderFast(self, current_quality, theoretical_min, starting_quality):
        """
        Fast disorder measurement using pre-computed values.
        """
        # Use starting quality as max estimate instead of computing random samples
        if starting_quality <= theoretical_min:
            return 0.0
        
        disorder_level = (current_quality - theoretical_min) / (starting_quality - theoretical_min)
        return np.clip(disorder_level, 0, 1)
        
    def getAdaptiveWeights(self, base_weights, disorder_level, strategy_type):
        """
        Adjusts strategy weights based on current disorder level.
        """
        if strategy_type == 'swapChoice':
            # High disorder: favor larger operations (blocks, shuffles)
            # Low disorder: favor smaller operations (swaps, rotations)
            disorder_bias = np.array([
                0.5,  # choice 0: basic swaps - always useful
                0.3,  # choice 1: 3-rotations - more useful when ordered
                0.3,  # choice 2: 4-rotations - more useful when ordered  
                0.2,  # choice 3: 5-rotations - more useful when ordered
                1.5,  # choice 4: block swaps - more useful when disordered
                1.0,  # choice 5: block shifts - useful throughout
                0.8,  # choice 6: block flips - moderately useful when disordered
                2.0,  # choice 7: shuffles - very useful when disordered
                0.4,  # choice 8: triangle rotations - fine-tuning
                0.4   # choice 9: square rotations - fine-tuning
            ])
            
            # Interpolate between conservative (low disorder) and aggressive (high disorder)
            adaptive_multiplier = disorder_bias * disorder_level + (1 - disorder_level)
            return base_weights * adaptive_multiplier
            
        elif strategy_type == 'blockSize':
            # High disorder: favor larger blocks
            size_bias = 1 + disorder_level * 2  # Scale from 1x to 3x for larger sizes
            return base_weights * (np.arange(len(base_weights)) * size_bias + 1)
            
        return base_weights        
        
    
    def buildHeatMapFast(self, swapTable, distances, width, height):
        """
        Faster heat map building with simplified calculations.
        """
        heat_map = np.zeros((width, height))
        
        # Sample only every Nth point for speed
        sample_step = max(1, len(swapTable) // 1000)  # Sample at most 1000 points
        
        for i in range(0, len(swapTable), sample_step):
            grid_idx = swapTable[i]
            gx = int(0.5 + self.grid_norm[grid_idx][0] * (width-1))
            gy = int(0.5 + self.grid_norm[grid_idx][1] * (height-1))
            
            # Simplified heat calculation - just use current distance without finding optimal
            current_distance = distances[grid_idx, i]
            heat_map[gx, gy] += current_distance
        
        # Skip Gaussian filter for speed
        # Normalize to probabilities
        if np.max(heat_map) > 0:
            heat_map = heat_map / np.max(heat_map)
        
        return heat_map 
        
   
    
    def selectHotCellFast(self, heat_map, width, height):
        """
        Faster hot cell selection using simplified probability.
        """
        # Simple approach: find the hottest regions and pick randomly among top 20%
        flat_heat = heat_map.flatten()
        if np.max(flat_heat) == 0:
            return np.random.randint(0, width), np.random.randint(0, height)
        
        # Find top 20% threshold
        threshold = np.percentile(flat_heat, 80)
        hot_indices = np.where(flat_heat >= threshold)[0]
        
        if len(hot_indices) == 0:
            return np.random.randint(0, width), np.random.randint(0, height)
        
        selected_idx = np.random.choice(hot_indices)
        cellx = selected_idx % width
        celly = selected_idx // width
        
        return cellx, celly
        
    def trackStrategySuccess(self, strategy_id, improvement, context):
        """
        Track detailed success metrics for each strategy.
        """
        if strategy_id not in self.strategy_success_history:
            self.strategy_success_history[strategy_id] = {
                'attempts': 0,
                'successes': 0,
                'total_improvement': 0,
                'context_performance': {}  # Performance in different contexts
            }
        
        history = self.strategy_success_history[strategy_id]
        history['attempts'] += 1
        
        # Define context_key early so it's always available
        context_key = f"disorder_{int(context['disorder']*10)}"
        
        # Initialize context performance tracking if needed
        if context_key not in history['context_performance']:
            history['context_performance'][context_key] = {'attempts': 0, 'successes': 0}
        
        # Always increment attempts for this context
        history['context_performance'][context_key]['attempts'] += 1
        
        # Only track successes if there was improvement
        if improvement > 0:
            history['successes'] += 1
            history['total_improvement'] += improvement
            history['context_performance'][context_key]['successes'] += 1  
            
    def getSmartWeights(self, base_weights, current_context):
        """
        Calculate weights based on historical performance in similar contexts.
        """
        smart_weights = base_weights.copy()
        
        for i, weight in enumerate(base_weights):
            if i in self.strategy_success_history:
                history = self.strategy_success_history[i]
                
                # Base success rate
                if history['attempts'] > 10:  # Only adjust if we have enough data
                    success_rate = history['successes'] / history['attempts']
                    smart_weights[i] *= (0.5 + success_rate)  # Scale between 0.5x and 1.5x
                    
                # Context-specific adjustment
                context_key = f"disorder_{int(current_context['disorder']*10)}"
                if context_key in history['context_performance']:
                    ctx_perf = history['context_performance'][context_key]
                    if ctx_perf['attempts'] > 5:
                        ctx_success_rate = ctx_perf['successes'] / ctx_perf['attempts']
                        smart_weights[i] *= (0.7 + 0.6 * ctx_success_rate)
        
        return smart_weights        
        
    def detectConvergence(self, recent_improvements, window_size=1000):
        """
        Detect if optimization has converged and suggest phase changes.
        """
        if len(recent_improvements) < window_size:
            return False, "early"
        
        recent_window = recent_improvements[-window_size:]
        improvement_rate = np.mean(recent_window)
        improvement_std = np.std(recent_window)
        
        if improvement_rate < 1e-8 and improvement_std < 1e-8:
            return True, "converged"
        elif improvement_rate < 1e-6:
            return False, "fine_tuning"
        else:
            return False, "active"                           
