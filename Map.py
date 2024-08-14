import numpy as np
import matplotlib.pyplot as pl
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import random

class Map:
    #Constructor
    def __init__(self, Model, Width, Height, Depth, Radius, Shape, Randomness, SimulationArguments, UpdatesPerFrame, Matrix = None, Filename = None, Centre = None):

        #If there is a filename read the file
        if Filename != None:
            Matrix = np.genfromtxt(Filename, delimiter=',')

        #Else create the Map
        else:
            if Matrix == None:

                match Model:
                    case 'Meinhardt':
                        Matrix = np.zeros((Width, Height, Depth))

                        #Default to middle of the circle
                        if Centre is None:
                            Centre = [(int(Width/2), int(Height/2))]
                        CentrePoints = len(Centre)

                        #Default radius to 1/16th
                        if Radius is None: 
                            Radius = min(Centre[0], Centre[1], Width-Centre[0], Height-Centre[1])/16

                        Matrix[:,:,:] = 0.75
                        Matrix[:,:,2] = 0.140625

                        for CentrePoint in Centre:
                            Y, X = np.ogrid[:Height, :Width]
                            DistanceFromCentre = np.sqrt((X - CentrePoint[0])**2 + (Y-CentrePoint[1])**2)
                            
                            # print(DistanceFromCentre)
                            # import matplotlib.pyplot as plt
                            # plt.imshow(DistanceFromCentre)
                            # plt.colorbar()
                            # plt.show()

                            Mask = DistanceFromCentre <= Radius

                            MatrixA = Matrix[:,:,0]
                            Noise = np.random.normal(0, 0.01, MatrixA.shape)
                            MatrixA[Mask] = MatrixA[Mask] + Noise[Mask] + 0.05

                        #Simulation Parameters
                        self.kAB, self.kC, self.kDE, self.Diff1, self.Diff2, self.DeltaTime = SimulationArguments
                        self.UpdatesPerFrame = UpdatesPerFrame

                    case 'GrayScott':
                        #Create 2 Matricies
                        MatrixA = (1-Randomness) * np.ones((Width, Height))
                        MatrixB = np.zeros((Width, Height))
                    
                        #Add Randomness
                        MatrixA += Randomness * np.random.random((Width, Height))
                        MatrixB += Randomness * np.random.random((Width, Height))

                        #Create Initial Values
                        CenterWidth = Width//2
                        CenterHeight = Height//2

                        if Shape == 'Circle':
                            #Default to middle of the circle
                            if Centre is None:
                                Centre = (int(Width/2), int(Height/2))

                            #Default to 1/8th
                            if Radius is None: 
                                Radius = min(Centre[0], Centre[1], Width-Centre[0], Height-Centre[1])/8

                            for CentrePoint in Centre:
                                Y, X = np.ogrid[:Height, :Width]
                                DistanceFromCentre = np.sqrt((X - CentrePoint[0])**2 + (Y-CentrePoint[1])**2)
                                Mask = DistanceFromCentre <= Radius

                                MatrixA[Mask] = 0.5
                                MatrixB[Mask] = 0.2

                        #Square
                        elif Shape == 'Square':
                            MatrixA[CenterWidth-Radius:CenterWidth+Radius, CenterHeight-Radius:CenterHeight+Radius] = 0.5
                            MatrixB[CenterWidth-Radius:CenterWidth+Radius, CenterHeight-Radius:CenterHeight+Radius] = 0.25

                        Matrix = np.dstack((MatrixA,MatrixB))

                        #Simulation Parameters
                        self.DA, self.DB, self.Feed , self.Kill, self.DeltaTime = SimulationArguments
                        self.UpdatesPerFrame = UpdatesPerFrame

        #Figure Work
        Figure, Axis = pl.subplots(1,Depth)
        Images = []

        for DepthIndex in range(0,Depth):
            Images.append(Axis[DepthIndex].imshow(Matrix[:,:,DepthIndex], animated=True, cmap='Greys'))
            Axis[DepthIndex].axis('off')
            Axis[DepthIndex].set_title(f'Chemical {chr(DepthIndex + 65)}')

        #Declare Matrix, Figure, and Images
        self.Model = Model
        self.Matrix = Matrix
        self.Figure = Figure
        self.Images = Images

        #Matrxi Dimensions
        self.Width = Width
        self.Height = Height
        self.Depth = Depth

    def __str__(self):
        return f"{self.Width}x{self.Height}x{self.Depth} Matrix\n{self.Matrix})"

    #Map
    @property
    def M(self):
        print("Getter")
        return self.Matrix
    @M.setter
    def M(self, value):
        self.Matrix = value
    @M.deleter
    def M(self):
        del self.Matrix

    def getMatrixA(self):
        return self.Matrix[:,:,0]

    def getMatrixB(self):
        return self.Matrix[:,:,1]

    def getHeight(self):
        return self.Height
    
    def getWidth(self):
        return self.Width
    
    def getFigure(self):
        return self.Figure
    
    def getAnimationArguements(self):
        match self.Model:
            case 'GrayScott':
                return(self.UpdatesPerFrame, self.Model, self.getMatrixA(), self.getMatrixB(), self.DA, self.DB, self.Feed, self.Kill, self.DeltaTime)
            
            case 'Meinhardt':
                return(self.UpdatesPerFrame, self.Model, self.Matrix[:,:,0], self.Matrix[:,:,1], self.Matrix[:,:,2], self.Matrix[:,:,3], self.Matrix[:,:,4], self.kAB, self.kC, self.kDE, self.Diff1, self.Diff2, self.DeltaTime)