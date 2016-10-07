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

import scipy.spatial.distance as dist
from scipy import spatial
import math
from IPython.display import clear_output
import numpy as np

class SwapOptimizer:
    
    lastState = None
    grid_norm = None
    xy_norm = None
    lastSwapTable = None
    lastWidth = 0
    lastHeight = 0
    
    def optimize( self, xy, grid, width, height, iterations, shakeIterations = 0,swapTable=None,continueFromLastState = False):     

        if self.lastState is None:
            self.grid_norm = grid - np.min(grid,axis=0)
            self.grid_norm /=  np.max(self.grid_norm,axis=0)

            self.xy_norm = xy - np.min(xy,axis=0)
            self.xy_norm /=  np.max(self.xy_norm,axis=0)
            
            self.lastWidth = width
            self.lastHeight = height
        else:
            width = self.lastWidth
            height = self.lastHeight

        reward = 5.0
        punish = 0.999

        totalEntries = len(self.grid_norm)

        choiceModes = np.arange(10)
        colSizes = np.arange(1,width)
        rowSizes = np.arange(1,height)
        swapOffsets = np.arange(1,9)
        offsets = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[-1,1],[-1,-1]])
        toleranceChoices = np.arange(1,16)

        if swapTable is None:
            swapTable =  np.arange(totalEntries)


        if self.lastState is None:
            cells = np.zeros((width, height),dtype=np.int)


            ci = {}
            for j in range(totalEntries):
                cx = int(0.5+self.grid_norm[j][0]*(width-1))
                cy = int(0.5+self.grid_norm[j][1]*(height-1))

                if (cx,cy) in ci:
                    print "ERROR:",(cx,cy)," doubled"
                cells[cx][cy] = j
                ci[(cx,cy) ] = True

            if len(ci) != totalEntries:
                print "ERROR in cell mapping"
                return

            distances = dist.cdist(self.grid_norm, self.xy_norm, 'sqeuclidean')
            swapChoiceWeights = np.array([1.0] * len(choiceModes),dtype=np.float64)
            colSizeWeights = np.array([1.0] * len(colSizes),dtype=np.float64)
            rowSizeWeights = np.array([1.0] * len(rowSizes),dtype=np.float64)
            swapOffsetWeights = np.array([1.0] * len(swapOffsets),dtype=np.float64)
            toleranceWeights = np.array([1.0] * len(toleranceChoices),dtype=np.float64)
            totalIterations = iterations
        else:
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
        print "Starting sum of distances",startingQuality
        if shakeIterations > 0:
            self.shake(cells,swapTable,shakeIterations,1,width,height)
            bestQuality = self.sumDistances(swapTable,distances)
            print "After shake sum of distances",bestQuality

        bestSwapTableBeforeAnnealing = swapTable.copy()
        toleranceSteps = 0

        for i in range(iterations):
            if i>0 and i % 20000 == 0:
                clear_output(wait=True)
                print "Starting sum of distances",startingQuality
            if i % 1000 == 0:
                print i,bestQuality

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
                        self.swapBlock(cells,swapTable,cellx,celly,cellx2,celly2,cols,rows,width,height)
                        self.learnWeight(toleranceWeights,chosenToleranceSteps-1,punish)
                        self.learnWeight(swapChoiceWeights,swapChoice,punish)
                        self.learnWeight(colSizeWeights,cols-1,punish)
                        self.learnWeight(rowSizeWeights,rows-1,punish)

            elif swapChoice == 5:   
                cols = np.random.choice(colSizes,p=colSizeWeights/np.sum(colSizeWeights))
                rows = np.random.choice(rowSizes,p=rowSizeWeights/np.sum(rowSizeWeights))
                dx = dy = 0
                if np.random.random() < 0.5:
                    dx = np.random.randint(-cellx,width-cols)
                else:
                    dy = np.random.randint(-celly,height-rows)   

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
                        self.shiftBlock(cells,swapTable,cellx+dx,celly+dy,cols,rows,-dx,-dy,width,height)
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
                        self.flipBlock(cells,swapTable,cellx,celly,cols,rows,flipxy[0],flipxy[1],width,height)
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
                        self.restoreBlock(cells,swapTable,cellx,celly,cols,rows,oldState)
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
        print "final distance sum:",bestQuality
        print "improvement:", startingQuality-bestQuality 
        if startingQuality<bestQuality:
            print "reverting to initial swap table"
        self.lastState = {'cells':cells,'iterations':totalIterations, 'distances':distances,
                                              'swapChoiceWeights':swapChoiceWeights,'swapOffsetWeights':swapOffsetWeights,
                                              'colSizeWeights':colSizeWeights,'rowSizeWeights':rowSizeWeights,
                                              'toleranceWeights':toleranceWeights}
        
        self.lastSwapTable = bestSwapTableBeforeAnnealing.copy()
        
        return bestSwapTableBeforeAnnealing 
    
    def continueOptimization( self, iterations, shakeIterations = 0):     
        if self.lastState is None:
            return
        return self.optimize( None, None, None, None, iterations, shakeIterations = shakeIterations,continueFromLastState = True,swapTable=self.lastSwapTable)
    
    def swapIndices(self,d,i1,i2):
        d[i1], d[i2] = d[i2], d[i1]

    def rotate3Indices(self,d,i1,i2,i3,rotateLeft=True):
        if len(set([i1,i2,i3]))!=3:
            return
        if rotateLeft:
            d[i2],d[i3],d[i1] = d[i3],d[i2],d[i1] 
        else:
            d[i2],d[i3],d[i1] = d[i3],d[i2],d[i1] 

    def rotate4Indices(self,d,i1,i2,i3,i4,rotateLeft=True):
        if len(set([i1,i2,i3,i4]))!=4:
            return
        if rotateLeft:
            d[i1],d[i4],d[i3],d[i2] = d[i4],d[i3],d[i2],d[i1] 
        else:
            d[i3],d[i2],d[i1],d[i4] = d[i4],d[i3],d[i2],d[i1] 

    def rotate5Indices(self,d,i1,i2,i3,i4,i5,rotateLeft=True):
        if rotateLeft:
            d[i1],d[i5],d[i4],d[i3],d[i2] = d[i5],d[i4],d[i3],d[i2],d[i1] 
        else:
            d[i4],d[i3],d[i2],d[i1],d[i5] = d[i5],d[i4],d[i3],d[i2],d[i1] 


    def swapBlock(self,cells,d,tlx1,tly1,tlx2,tly2,cols,rows,width,height):
        if max(tlx1,tlx2)+cols < width and max(tly1,tly2)+rows < height and (max(tlx1,tlx2) - min(tlx1,tlx2) >= cols or max(tly1,tly2) - min(tly1,tly2) >= rows):
            temp = []
            for row in range( rows):
                for col in range( cols):
                    temp.append(d[cells[tlx1+col][tly1+row]])
                    d[cells[tlx1+col][tly1+row]] = d[cells[tlx2+col][tly2+row]]
            i = 0
            for row in range( rows):
                for col in range( cols):
                    d[cells[tlx2+col][tly2+row]] = temp[i]
                    i+=1
            return True
        else:
            return False

    def shuffleBlock(self,cells,d,tlx,tly,cols,rows,width,height):
        if tlx+cols < width and tly+rows < height:
            temp = []
            for row in range( rows):
                for col in range( cols):
                    temp.append(d[cells[tlx+col][tly+row]])
            temp = np.array(temp)
            oldState = temp.copy()
            np.random.shuffle(temp)
            i = 0
            for row in range( rows):
                for col in range( cols):
                    d[cells[tlx+col][tly+row]] = temp[i]
                    i+=1
            return oldState
        else:
            return []

    def restoreBlock(self,cells,d,tlx,tly,cols,rows,oldState):
        i = 0
        for row in range( rows):
            for col in range( cols):
                d[cells[tlx+col][tly+row]] = oldState[i]
                i+=1


    def shiftBlock(self,cells,d,tlx,tly,cols,rows,dx,dy,width,height):
        if (dx!=0 and dy!= 0) or (dx==0 and dy == 0) or cols*rows < 2:
            return False

        if dx < 0:
            tlx += dx
            cd = cols
            cols = -dx
            dx = cd

        if dy < 0:
            tly += dy
            cd = rows 
            rows = -dy
            dy = cd

        if tlx+cols >= width or tlx+cols+dx >= width or tly+rows >= height or tly+rows+dy >= height:
            return False
        if tlx < 0 or tlx+dx < 0 or tly < 0 or tly+dy < 0:
            return False

        temp2 = []
        for row in range( rows):
            for col in range( cols):
                temp2.append(d[cells[tlx+col][tly+row]])

        temp1 = []
        if dy==0 and dx > 0:
            for row in range( rows ):
                for col in range( dx):
                    temp1.append(d[cells[tlx+cols+col][tly+row]])
        elif dx == 0 and dy > 0:
            for row in range( dy):
                for col in range( cols ):
                    temp1.append(d[cells[tlx+col][tly+rows+row]])


        i = 0
        for row in range( rows):
            for col in range( cols):
                d[cells[tlx+col+dx][tly+row+dy]] = temp2[i]
                i+=1

        i = 0
        if dy==0 and dx > 0:
            for row in range( rows ):
                for col in range( dx):
                    d[cells[tlx+col][tly+row]] = temp1[i]
                    i+=1
        elif dx == 0 and dy > 0:
            for row in range( dy):
                for col in range( cols ):
                    d[cells[tlx+col][tly+row]]= temp1[i]
                    i+=1
        return True


    def flipBlock(self,cells,d,tlx,tly,cols,rows,flipx,flipy,width,height):

        if tlx+cols >= width or tly+rows>=height:
            return False
        temp = []
        for row in range( rows):
            for col in range( cols):
                temp.append(d[cells[tlx+col][tly+row]])

        i = 0
        for r in range( rows):
            if flipy:
                row = rows-r-1
            else:
                row = r
            for c in range( cols):
                if flipx:
                    col = cols-c-1
                else:
                    col = c
                d[cells[tlx+col][tly+row]] = temp[i]
                i+=1
        return True

    def shake( self,cells,d,iterations,strength,width,height):
        for i in range(iterations):
            col = np.random.randint(0,width)
            row = np.random.randint(0,height)
            ox = np.random.randint(-strength,strength+1)
            oy = np.random.randint(-strength,strength+1)
            if (ox!=0 or oy!=0) and col+ox > -1 and row+oy > -1 and col+ox<width and row+oy<height:
                i1 = cells[col][row]
                i2 = cells[col+ox][row+oy]
                d[i1], d[i2] = d[i2], d[i1]


    def sumDistances(self,swapTable,distances):
        r = np.arange(len(swapTable))
        return np.sum(distances[swapTable[r],r])

    def learnWeight( self,b, idx, f ):
        b[idx]*=f
        if b[idx]<0.0001:
            b[idx] = 0.0001
        elif b[idx]>10000.0:
            b[idx]=10000.0
    