import numpy as np
import pyvista as pv
import multiprocessing 
import time
import os
import sys

from itertools import repeat
import traceback


def MeinhardtVoronoi(Matrix, RegionNeighboursIndices, RegionEdgeLengths, RegionDistances, kAB, kC, kDE, DiffG, DiffS, TimeDelta):

    MatrixG1, MatrixG2, MatrixR, MatrixS1, MatrixS2 = Matrix[:, 0], Matrix[:, 1], Matrix[:, 2], Matrix[:, 3], Matrix[:, 4]
    LaplacianParameters = ((MatrixG1, RegionNeighboursIndices, RegionEdgeLengths, RegionDistances), (MatrixG2, RegionNeighboursIndices, RegionEdgeLengths, RegionDistances), (MatrixS1, RegionNeighboursIndices, RegionEdgeLengths, RegionDistances), (MatrixS2, RegionNeighboursIndices, RegionEdgeLengths, RegionDistances)) 
    LaplaceG1 = ApplyLaplacianVoronoi(LaplacianParameters[0])
    LaplaceG2 = ApplyLaplacianVoronoi(LaplacianParameters[1])
    LaplaceS1 = ApplyLaplacianVoronoi(LaplacianParameters[2])
    LaplaceS2 = ApplyLaplacianVoronoi(LaplacianParameters[3])

    #Calculate change in matricies
    DeltaG1 = DiffG * LaplaceG1 + 0.01 * MatrixG1 * MatrixG1 * MatrixS2 / MatrixR - MatrixG1 * kAB
    DeltaG2 = DiffG * LaplaceG2 + 0.01 * MatrixG2 * MatrixG2 * MatrixS1 / MatrixR - MatrixG2 * kAB
    DeltaR = 0.01 * MatrixG1 * MatrixG1 * MatrixS2 + 0.01 * MatrixG2 * MatrixG2 * MatrixS1 - MatrixR * kC
    DeltaS1 = DiffS * LaplaceS1 + (MatrixG1 - MatrixS1) * kDE
    DeltaS2 = DiffS * LaplaceS2 + (MatrixG2 - MatrixS2) * kDE

    #Apply change
    MatrixG1 += DeltaG1 * TimeDelta
    MatrixG2 += DeltaG2 * TimeDelta
    MatrixR += DeltaR * TimeDelta
    MatrixS1 += DeltaS1 * TimeDelta
    MatrixS2 += DeltaS2 * TimeDelta
    
    MatrixNext = np.stack((MatrixG1, MatrixG2, MatrixR, MatrixS1, MatrixS2), axis=1)
    return MatrixNext

def Meinhardt(Model, MatrixG1, MatrixG2, MatrixR, MatrixS1, MatrixS2, kAB, kC, kDE, DiffG, DiffS, TimeDelta):

    if Model == 'Meinhardt':
        #Get laplacians
        LaplaceG1 = ApplyLaplacian(MatrixG1)
        LaplaceG2 = ApplyLaplacian(MatrixG2)
        LaplaceS1 = ApplyLaplacian(MatrixS1)
        LaplaceS2 = ApplyLaplacian(MatrixS2)

        #Calculate change in matricies
        DeltaG1 = DiffG * LaplaceG1 + 0.01 * MatrixG1 * MatrixG1 * MatrixS2 / MatrixR - MatrixG1 * kAB
        DeltaG2 = DiffG * LaplaceG2 + 0.01 * MatrixG2 * MatrixG2 * MatrixS1 / MatrixR - MatrixG2 * kAB
        DeltaR = 0.01 * MatrixG1 * MatrixG1 * MatrixS2 + 0.01 * MatrixG2 * MatrixG2 * MatrixS1 - MatrixR * kC
        DeltaS1 = DiffS * LaplaceS1 + (MatrixG1 - MatrixS1) * kDE
        DeltaS2 = DiffS * LaplaceS2 + (MatrixG2 - MatrixS2) * kDE

        #Apply change
        MatrixG1 += DeltaG1 * TimeDelta
        MatrixG2 += DeltaG2 * TimeDelta
        MatrixR += DeltaR * TimeDelta
        MatrixS1 += DeltaS1 * TimeDelta
        MatrixS2 += DeltaS2 * TimeDelta
        
        return (MatrixG1, MatrixG2, MatrixR, MatrixS1, MatrixS2)
    
    else:
        return -1

#Laplacian Operator
def ApplyLaplacian(Matrix):
    LaplacianMatrix = Matrix * -4
    LaplacianMatrix += np.roll(Matrix, (0,-1), (0,1))
    LaplacianMatrix += np.roll(Matrix, (0,+1), (0,1))
    LaplacianMatrix += np.roll(Matrix, (-1,0), (0,1))
    LaplacianMatrix += np.roll(Matrix, (+1,0), (0,1))
    
    return LaplacianMatrix


def ApplyLaplacianVoronoi(Args):
    Matrix, RegionNeighbours, RegionEdgeLengths, RegionDistances = Args
    LaplacianValues = []
    count = 0

    for Value, Neighbours, RegionEdgeLength, Distances in zip(Matrix, RegionNeighbours, RegionEdgeLengths, RegionDistances):
        #Calculate Weights (1/Distance^2)
        WeightedSum = 0.0
        WeightsSummed = sum(1/(Dist ** 2) for Dist in Distances)

        for Neighbour, Edge, Dist in zip(Neighbours, RegionEdgeLength, Distances):
            if Edge > 1:
                Edge = 1

            NeighbourValue = Matrix[Neighbour]
            WeightedNeighbour = 1 / (Dist ** 2)
            WeightedSum += WeightedNeighbour * (NeighbourValue - Value) * Edge
            # print(NeighbourValue,Dist, WeightedNeighbour, WeightedSum, Edge)
            # input()
        
        Laplace = 4 * WeightedSum / WeightsSummed
        LaplacianValues.append(Laplace)
        count += 1
        
        # if count == 100:
        #     break

    return np.array(LaplacianValues)

def ApplyLaplacianVoronoiTEST(Args):
    Matrix, RegionNeighbours, RegionEdgeLengths, RegionDistances = Args
    NumRegions = len(Matrix)
    LaplacianValues = np.zeros(NumRegions)

    count = 0
    for Value, Neighbours, Distances in zip(Matrix, RegionNeighbours, RegionDistances):
        Weights = 1.0/(Distances**2)
        WeightsSummed = np.sum(Weights)

        NeighbourValues = Matrix[Neighbours]
        WeightedSum = np.sum(Weights * (NeighbourValues - Value))

        Laplace = 4 * WeightedSum/WeightsSummed
        LaplacianValues[count] = Laplace
        count += 1

    return LaplacianValues


def LaplaicianVoronoiWorker(args):
    Matrix, ThreadArgs = args
    Value, Neighbours, RegionEdgeLengths, Distances = ThreadArgs
    LaplacianValues = []

    try:
        #Check Num Neighbours vs Length
        assert len(Neighbours) == len(Distances) 
        
        WeightedSum = 0.0
        WeightsSummed = sum(1/(Dist ** 2) for Dist in Distances)
        
        for Neighbour, Dist in zip(Neighbours, Distances):
            NeighbourValue = Matrix[Neighbour]
            WeightedNeighbour = 1 / (Dist ** 2)

            # print(f"-   {NeighbourValue}\n-   {WeightedNeighbour}\n")

            WeightedSum += WeightedNeighbour * (NeighbourValue - Value)

        Laplace = 4 * WeightedSum / WeightsSummed
        LaplacianValues.append(Laplace)
        
        return np.array(LaplacianValues)
    except:
        print(f"////////////////////////////////////////////////\nRan into error running Laplacian Voronoi Working Multi-thread\nRegion Neighbours: {Neighbours, type(Neighbours)}\nRegion Edge Lengths: {RegionEdgeLengths, type(RegionEdgeLengths)}\nRegion Distances: {Distances, type(Distances)}\n\n{traceback.print_exc()}\n////////////////////////////////////////////////")
        sys.exit()

#https://www.geeksforgeeks.org/how-to-normalize-an-array-in-numpy-in-python/
def Normalize(Array, MinValue = 0, MaxValue = 1):
    NormalizedArray = []
    Diff = MaxValue - MinValue
    DiffArray = max(Array) - min(Array)
    for i in Array:
        temp = (((i - min(Array))*Diff)/DiffArray) + MinValue
        NormalizedArray.append(temp)

    return NormalizedArray

def NormalizeMatrix(Matrix):
    MatrixG1 = Matrix[:,0]
    NormalizedG1 = np.array(Normalize(MatrixG1))
    MatrixG2 = Matrix[:,1]
    NormalizedG2 = np.array(Normalize(MatrixG2))
    MatrixR = Matrix[:,2]
    NormalizedR = np.array(Normalize(MatrixR))
    MatrixS1 = Matrix[:,3]
    NormalizedS1 = np.array(Normalize(MatrixS1))
    MatrixS2 = Matrix[:,4]
    NormalizedS2 = np.array(Normalize(MatrixS2))

    MatrixNormalized = np.stack((MatrixG1, MatrixG2, MatrixR, MatrixS1, MatrixS2), axis=1)
    return MatrixNormalized
 
def RenderMeinhardt(Faces, Points, PyvistaFaces, VoronoiRegions, Matrix, RegionNeighboursIndices, RegionEdgeLengths, RegionDistances):
    VorPoints = np.array(VoronoiRegions)
    StackedPoints = np.concatenate((Points, VorPoints), axis=0)
    RGBArray = []

    #Get Normalized Matrix
    print(Matrix[:,1])
    Matrix[:,1] = Normalize(Matrix[:,1], 0 , 255)

    ChemicalValues = np.floor(Matrix[:,1])
    RGBValues = np.column_stack((ChemicalValues,ChemicalValues,ChemicalValues)).astype(np.uint8)
    # print(RGBValues)
    # print(type(RGBValues[0]))
    # print(type(RGBValues[0,0]))

    # #Create Mesh
    Mesh = pv.PolyData()
    Mesh.points = StackedPoints
    Mesh.faces = PyvistaFaces
    Mesh.save("mesh.ply")
    MeshCloud = pv.PolyData(VorPoints)
    MeshCloud.save("PointCloud.ply", texture = RGBValues)

    #Create Plotter
    p = pv.Plotter()
    p.add_mesh(Mesh, show_edges = True)

    #Plot Points
    for Region, Concentrations in zip(VoronoiRegions, Matrix):
        # print(Region)
        G2 = int(Concentrations[1])
        # print(f"G2 Concentration: {G2}")

        if len(Region) != 0:
            p.add_points(np.array(Region), color = [G2, G2, G2])

    p.show()



    # # LAPLACE VISUALISATION
    # MatrixG1 = Matrix[:,1]
    # Lap = ApplyLaplacianVoronoi((MatrixG1, RegionNeighboursIndices, RegionEdgeLengths, RegionDistances))
    # LAPLACIANNORMAL = Normalize(Lap, 0 , 255)
    # LAPLACIANNORMAL = np.floor(LAPLACIANNORMAL)
    # RGBValues = np.column_stack((LAPLACIANNORMAL,LAPLACIANNORMAL,LAPLACIANNORMAL)).astype(np.uint8)

    # p = pv.Plotter()
    # p.add_mesh(Mesh, show_edges = True)

    # #Plot Points
    # for Region, Concentrations in zip(VoronoiRegions, RGBValues):
    #     print(Region)
    #     print(np.asarray(Concentrations))
    #     # print(f"G2 Concentration: {G2}")

    #     if len(Region) != 0:
    #         p.add_points(np.array(Region), color = np.asarray(Concentrations))

    # p.show()
