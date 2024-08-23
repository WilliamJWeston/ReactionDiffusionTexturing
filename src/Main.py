import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import traceback
import time
import statistics

from ReactionDiffusion import *
from itertools import groupby
from matplotlib.colors import Normalize
from Map import Map
from Methods import *
from Relaxtion import *
from ObjReader import ParseOBJ

import pyvista as pv
import sys

# np.set_printoptions(threshold=sys.maxsize)

#MatplotLib Animation Update
def UpdateFigure(Frame, UpdatesPerFrame, *Arguments):
    #https://www.karlsims.com/rd.html - Gray Scott Model of Reaction-Diffusion
    # print(f"Argumemts: {Arguments} ({len(Arguments)})")
    Type = Arguments[0]

    match Type:
        case "GrayScott":
            Method = GrayScott
            MatrixA, MatrixB = Arguments[1], Arguments[2]
            DA, DB, Feed, Kill, TimeDelta = Arguments[3:]

            for update in range(UpdatesPerFrame):
                MatrixA, MatrixB = Method(*Arguments)
                Arguments = (Type, MatrixA, MatrixB, DA, DB, Feed, Kill, TimeDelta)

            ImageA, ImageB = Images[0], Images[1]

            ImageA.set_array(MatrixA)
            ImageB.set_array(MatrixB)
            ImageA.set_norm(Normalize(vmin=np.amin(MatrixA),vmax=np.amax(MatrixA)))
            ImageB.set_norm(Normalize(vmin=np.amin(MatrixB),vmax=np.amax(MatrixB)))

            return ImageA, ImageB
        
        case "Meinhardt":
            Method = Meinhardt
            MatrixA, MatrixB, MatrixC, MatrixD, MatrixE = Arguments[1], Arguments[2], Arguments[3], Arguments[4], Arguments[5]
            kAB, kC, kDE, Diff1, Diff2, DeltaTime = Arguments[6:]

            for update in range(UpdatesPerFrame):
                MatrixA, MatrixB, MatrixC, MatrixD, MatrixE = Method(*Arguments)
                Arguments = (Type, MatrixA, MatrixB, MatrixC, MatrixD, MatrixE, kAB, kC, kDE, Diff1, Diff2, DeltaTime)

            ImageA, ImageB, ImageC, ImageD, ImageE = Images[0], Images[1], Images[2], Images[3], Images[4]

            ImageA.set_array(MatrixA)
            ImageB.set_array(MatrixB)
            ImageC.set_array(MatrixC)
            ImageD.set_array(MatrixD)
            ImageE.set_array(MatrixE)
            print(f"=====================================================\n{MatrixA}\n{Normalize(vmin=np.amin(MatrixA),vmax=np.amax(MatrixA))}")
            ImageA.set_norm(Normalize(vmin=np.amin(MatrixA),vmax=np.amax(MatrixA)))
            ImageB.set_norm(Normalize(vmin=np.amin(MatrixB),vmax=np.amax(MatrixB)))
            ImageC.set_norm(Normalize(vmin=np.amin(MatrixC),vmax=np.amax(MatrixC)))
            ImageD.set_norm(Normalize(vmin=np.amin(MatrixD),vmax=np.amax(MatrixD)))
            ImageE.set_norm(Normalize(vmin=np.amin(MatrixE),vmax=np.amax(MatrixE)))

            return ImageA, ImageB, ImageC, ImageD, ImageE
        
        case "Voronoi Meinhardt":
            Method = MeinhardtVoronoi

            for update in range(UpdatesPerFrame):
                MatrixA, MatrixB, MatrixC, MatrixD, MatrixE = Method(*Arguments)

            ImageA, ImageB, ImageC, ImageD, ImageE = Images[0], Images[1], Images[2], Images[3], Images[4]

            ImageA.set_array(MatrixA)
            ImageB.set_array(MatrixB)
            ImageC.set_array(MatrixC)
            ImageD.set_array(MatrixD)
            ImageE.set_array(MatrixE)
            ImageA.set_norm(Normalize(vmin=np.amin(MatrixA),vmax=np.amax(MatrixA)))
            ImageB.set_norm(Normalize(vmin=np.amin(MatrixB),vmax=np.amax(MatrixB)))
            ImageC.set_norm(Normalize(vmin=np.amin(MatrixC),vmax=np.amax(MatrixC)))
            ImageD.set_norm(Normalize(vmin=np.amin(MatrixD),vmax=np.amax(MatrixD)))
            ImageE.set_norm(Normalize(vmin=np.amin(MatrixE),vmax=np.amax(MatrixE)))

            return ImageA, ImageB, ImageC, ImageD, ImageE

    return Images

#Main
if __name__ == '__main__':

    Choice = "Relaxation"
    # Choice = "ReactionDiffusion"
    # Choice = "RenderMeinhardt"

    #OBJ Filepaths
    # OBJFilepath = f"OBJs/BambooShark/bamboo shark.obj"
    OBJFilepath = f"OBJs/Horse/SHIRE_01.obj"
    OBJFilepath = f"OBJs/Horse/HORSELOWPOLY.obj"
    # OBJFilepath = f"OBJs/Horse/Horse_OBJ.obj"
    # OBJFilepath = f"OBJs/Horse/HorseArmor.obj"
    # OBJFilepath = f"OBJs/Horse/Horse.obj"
    #OBJFilepath = f"OBJs/Wolf/WOLF.OBJ"

    #OBJ File Data
    if Choice == "Relaxation":
        Data = ParseOBJ(Filename = OBJFilepath)
        NumSamples = 250000
        Relaxation(MeshData = Data, NumSamples = NumSamples, IterationCount = 6, PlottingSample = 4000, ScalingFactor = 1)
        sys.exit()

    Selection = "None"
    # Selection = "Meinhardt"
    Selection = "GrayScott"

    #Animation Parameters
    match Selection:
        case 'Meinhardt':
            kAB = 0.04
            kC = 0.06
            kDE = 0.04
            DiffG = 0.009
            DiffS = 0.2
            DeltaTime = 1.0
            UpdatesPerFrame = 10

            Matrix = Map(Model = 'Meinhardt', Depth = 5, Width = 200, Height = 200, Centre = [(50,50),(100,150)], Radius = 12, Shape = 'Circle', Randomness = 0.5, SimulationArguments = (kAB, kC, kDE, DiffG, DiffS, DeltaTime), UpdatesPerFrame = UpdatesPerFrame)
        
        case 'GrayScott':
            #rolls -> DA = 0.16 DB = 0.08 Feed = 0.06 Kill = 0.062 DeltaTime = 1.0
            DA = 0.1
            DB = 0.05 
            Feed = 0.058
            Kill = 0.063
            DeltaTime = 1.0
            UpdatesPerFrame = 10

            Radius = 70
            Centre = [(100,100)]

            Matrix = Map(Model = 'GrayScott', Depth = 2, Width = 200, Height = 200, Centre = Centre,Radius = Radius, Shape = 'Circle', Randomness = 0.5, SimulationArguments = (DA, DB, Feed, Kill, DeltaTime), UpdatesPerFrame = UpdatesPerFrame)

        
        case 'None':
            pass

    #Animation Values
    if Selection != "None":
        Figure = Matrix.Figure
        Images = Matrix.Images
        AnimationArgs = Matrix.getAnimationArguements()
        Animation = animation.FuncAnimation(Figure, UpdateFigure, fargs=AnimationArgs, interval=1, blit=True)
        plt.show()
        plt.close()

    if Choice == "ReactionDiffusion":
        kAB = 0.04
        kC = 0.06
        kDE = 0.04
        DiffG = 0.009
        DiffS = 0.2
        DeltaTime = 1.0
        UpdatesPerFrame = 10

        #Load Mesh Data
        Faces = loadData('StoredData/Faces')
        PyvistaFaces = loadData('StoredData/PyvistaFaces')
        Points = loadData('StoredData/Points')
        Vertexes = loadData('StoredData/Vertexes')

        #Load Voronoi and Relaxation Data
        SamplePoints = loadData('StoredData/SamplePoints')
        SampleFaces = loadData('StoredData/SampleFaces')
        Regions = loadData('StoredData/Regions')
        RegionNeighbours = loadData('StoredData/RegionNeighbours')
        RegionNeighboursValues = loadData('StoredData/RegionNeighboursValues')
        RegionNeighboursIndices = loadData('StoredData/RegionNeighbourIndices')

        RegionEdgeLengths = loadData('StoredData/RegionEdgeLengths')
        RegionDistances = loadData('StoredData/RegionDistances')

        # PlotVoronoi(Faces, Points, PyvistaFaces, Regions, SamplePoints = SamplePoints, Matrix = None)

        Labels = [f"Label {i}" for i in range(len(SamplePoints))]
        plotter = pv.Plotter()
        plotter.add_point_labels(SamplePoints, Labels, point_size=3, font_size=10)
        plotter.show()
        v1,v2,v3,v4 = 7369, 6893, 6932, 7399
        #7369, 6893, 6932, 7399
        #249, 598, 602, 545
        Matrix = np.zeros((len(Regions), 5))
        Matrix[:,:] = 0.75
        Noise = np.random.normal(0, 0.01, Matrix.shape)
        Matrix[:,0][v1] = Matrix[:,0][v1] + Noise[:,0][v1] + 0.05
        Matrix[:,0][v2] = Matrix[:,0][v2] + Noise[:,0][v2] + 0.05
        Matrix[:,0][v3] = Matrix[:,0][v3] + Noise[:,0][v3] + 0.05
        Matrix[:,0][v4] = Matrix[:,0][v4] + Noise[:,0][v4] + 0.05
        Matrix[:,2] = 0.140625

        TotalFrames = 15000

        MatrixG1, MatrixG2, MatrixR, MatrixS1, MatrixS2 = Matrix[:, 0], Matrix[:, 1], Matrix[:, 2], Matrix[:, 3], Matrix[:, 4]
        

        #Get Current Time
        OldSystemTime = time.time()
        for Frame in range(0, TotalFrames):
            
            #Calculate Next Matrix
            MatrixNext = MeinhardtVoronoi(Matrix, RegionNeighboursIndices, RegionEdgeLengths, RegionDistances, kAB, kC, kDE, DiffG, DiffS, DeltaTime)
            
            #Update Var
            Matrix = MatrixNext

            if Frame % 200 == 0:
                UpdatedSystemTime = time.time()

                ElapsedSystemTime = time.time() - OldSystemTime
                print(f"Frame: {Frame} ({ElapsedSystemTime} seconds since last update)")
                print(Matrix)
                OldSystemTime = time.time()

        print(f"Simulation Complete")
        storeData('StoredData/Matrix', Matrix)

    if Choice == "RenderMeinhardt":
        Faces = loadData('StoredData/Faces')
        PyvistaFaces = loadData('StoredData/PyvistaFaces')
        Points = loadData('StoredData/Points')
        SamplePoints = loadData('StoredData/SamplePoints')
        Matrix = loadData('StoredData/Matrix')

        RegionNeighboursValues = loadData('StoredData/RegionNeighboursValues')
        RegionNeighboursIndices = loadData('StoredData/RegionNeighbourIndices')
        RegionEdgeLengths = loadData('StoredData/RegionEdgeLengths')
        RegionDistances = loadData('StoredData/RegionDistances')

        RenderMeinhardt(Faces, Points, PyvistaFaces, SamplePoints, Matrix, RegionNeighboursIndices, RegionEdgeLengths, RegionDistances)
