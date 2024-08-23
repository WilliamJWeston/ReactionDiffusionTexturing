from Methods import *
from ObjReader import ParseOBJ
from Voronoi import *
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import traceback
import sys

from numba import jit, cuda 

class IterationHolder(object):
    def __init__(self):

        #Create arrays to store relaxation repulsion values
        self.ResultantSample = []
        self.ResultantFaces = []

        #Dictionaries to store all values for each given sample
        self.FaceSamples = {}
        self.NearbySamples = []
        self.NearbySampsIndices = []
        self.NearbySampleDistances = {}
        self.BarycentricWeights = []

    def getNearbySamplesAndWeights(self):
        return(self.NearbySamples, self.BarycentricWeights)

    def setNearbySamplesAndWeights(self, NearbySamples, NearbySampsIndices, BarycentricWeights):
        self.NearbySamples = NearbySamples
        self.NearbySampsIndices = NearbySampsIndices
        self.BarycentricWeights = BarycentricWeights


    def getNearbySampleIndices(self):
        return(self.NearbySampsIndices)

    def getNearbySamples(self):
        return(self.NearbySamples)


    def getMappedPointAndFace(self):
        return(self.ResultantSample, self.ResultantFaces)
    
    def setMappedPointAndFace(self, a, b):
        self.ResultantSample.append(a)
        self.ResultantFaces.append(b)

#Verticies List - VertexList Object
#Faces List - FaceList Object
#Number of Samples - Number of samples to take from Mesh
#Replace - Bool: If to apply replace to random.choice, determining wether a value can be chose twice
#https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
def SampleFacesMethod(Verticies, Faces, NumSamples, Replace = False):
    TotalArea = Faces.TotalArea
    FaceSurfaceAreas = Faces.SurfaceAreas/TotalArea

    SamplesPerFace = np.ceil(NumSamples * FaceSurfaceAreas).astype(int)
    Overflow = np.sum(SamplesPerFace) - NumSamples

    #Remove Overflow
    if Overflow > 0:
        Indices = np.where(SamplesPerFace > 0)[0]
        FloorIndicies = np.random.choice(Indices, Overflow, replace = Replace)
        SamplesPerFace[FloorIndicies] -= 1

    NumSamples = np.sum(SamplesPerFace)
    SampleFaceIndicies = np.zeros((NumSamples), dtype=int)
    Accumulator = 0

    #Create Two Random Variables per Sample
    RandomVariables = np.random.rand(NumSamples, 2)
    Coordinates = Faces.VertexValues

    #Create Arary of Indicies, with one indicie pointing back to the original Face array per sample
    for FaceIndex, FaceNumberSamples in enumerate(SamplesPerFace):
        SampleFaceIndicies[Accumulator: Accumulator + FaceNumberSamples] = FaceIndex
        Accumulator += FaceNumberSamples
    
    #Initalise Arrays ABC, which will contain the respective 1st, 2nd, and 3rd Vertex of each face
    A = []
    B = []
    C = []

    for SampleIndex in SampleFaceIndicies:
        SampleCoordinate = Coordinates[SampleIndex]
        A.append(SampleCoordinate[0])
        B.append(SampleCoordinate[1])
        C.append(SampleCoordinate[2])

    #Sample Points
    SampledPointsA = (1 - np.sqrt(RandomVariables[:,0:1])) * A
    SampledPointsB = np.sqrt(RandomVariables[:,0:1]) * (1 - RandomVariables[:,1:]) * B 
    SampledPointsC = np.sqrt(RandomVariables[:,0:1]) * RandomVariables[:,1:] * C
    SampledPoints = SampledPointsA + SampledPointsB + SampledPointsC

    return SampledPoints, SampleFaceIndicies

def Relaxation_Setup(IterationObject, Samples, SampleFaces, FaceNeighbours, FaceValues, GroupedSamplePointsByFace, Faces, RepulsionRadius, Points, PyvistaFaces):
    #Create dicts to store values
    FaceSamples = {}
    BarycentricWeights = []
    NearbySamps = []
    NearbySampsIndices = []

    #For each point P on the Surface
    print("\n===== Mapping Neighbouring Verticies onto Face =====")
    for SamplePoint, Face, Index in zip(Samples, SampleFaces, range(0,len(Samples))):
        BarycentricWeights.append(CartesianToBarycentric(FaceValues[Face], SamplePoint))

        #Create empty dict to hold all samples that are mapped and within replusion radius
        temp_NearbySamps = []
        temp_NearbySampsIndicies = []

        #Get current faces samples, and append to nearby if within replusion radius
        AllFaceSamples = GroupedSamplePointsByFace[Face]
        FaceSamplesInd = [x for x in AllFaceSamples if x != Index]
        FaceSampleVals = [Samples[x] for x in FaceSamplesInd]
        SetKey(FaceSamples, Index, FaceSampleVals)

        #Determine wether face samples on the face are near the sample
        for FaceSampleValue, FaceSampleIndex in zip(FaceSampleVals, FaceSamplesInd):
            InterSampleDistance = PointToPointDistance(FaceSampleValue, SamplePoint)
            if InterSampleDistance < RepulsionRadius:
                temp_NearbySamps.append(FaceSampleValue)
                temp_NearbySampsIndicies.append(FaceSampleIndex)

        #Loop through the Neighbours
        for Neighbour in FaceNeighbours[Face]:
            if Neighbour != None:

                #Try and get Neighbours sampled points if it exists
                try:
                    TempSamples = GroupedSamplePointsByFace[Neighbour]
                    TempSamplevalues = [Samples[x] for x in TempSamples]

                    #Map each neighbours sample point back onto the plane
                    for NeighbourSample, NeighbourSampleIndex in zip(TempSamplevalues, TempSamples):
                        MappedNeigbourSample, _ = MapToPlane(Faces.VertexValues[Face], NeighbourSample)

                        #If within Replusion Radius, add to dict
                        if PointToPointDistance(MappedNeigbourSample, SamplePoint) < RepulsionRadius:
                            temp_NearbySamps.append(MappedNeigbourSample)
                            temp_NearbySampsIndicies.append(NeighbourSampleIndex)

                except KeyError:
                    pass

        NearbySamps.append(temp_NearbySamps)
        NearbySampsIndices.append(temp_NearbySampsIndicies)
        IterationObject.setNearbySamplesAndWeights(NearbySamps, NearbySampsIndices, BarycentricWeights)

    return IterationObject

def Relaxation_Repulsion(PreviousIterationObject, IterationObject, Samples, SampleFaces, FaceNeighbours, Mesh, ScalingFactor, Faces, Points, PyvistaFaces, RepulsionRadius):
    #Retrieve Setup Values
    NearbySamples, BarycentricWeights = PreviousIterationObject.getNearbySamplesAndWeights()
    PackedVar = (Faces, Points, PyvistaFaces, FaceNeighbours)

    #Compute and store repulsive forces for each sample point, apply them, and add them to an array
    for SamplePoint, Face, Index in zip(Samples, SampleFaces, range(0,len(Samples))):

        #Convert samplepoint to numpy, default newpoint to the samplepoint
        NewPoint = SamplePoint
        SamplePoint = np.array(SamplePoint)
        ResultantReplusiveForce = np.zeros(3)
    
        try:
            MappedNearbySample = NearbySamples[Index]
        except KeyError:
            # print("//////////////////////////////////////////")
            # traceback.print_exc()
            # print("//////////////////////////////////////////")
            pass
        except AttributeError:
            print("//////////////////////////////////////////")
            # traceback.print_exc()
            # print("//////////////////////////////////////////")
            pass

        #Calculate Influences
        Influences, Diffs = CalculateInfluence(SamplePoint, MappedNearbySample, RepulsionRadius, 5)
        
        #If there are no nearby sample points
        if Influences is None and Diffs is None:
            print("*No nearby sample points*")
            IterationObject.setMappedPointAndFace(SamplePoint, Face)
        
        else: 
            WeightedDiffs = Diffs * Influences[:, np.newaxis]
            ResultantReplusiveForce = WeightedDiffs.sum(axis=0) * ScalingFactor
            ResultantPoint = np.add(ResultantReplusiveForce, SamplePoint)

            NewPoint, NewFace, ResultantBarycentricWeights, EdgeValues = MapResultant(Face, ResultantPoint, SamplePoint, PackedVar)
            IterationObject.setMappedPointAndFace(NewPoint, NewFace)

    return IterationObject

def Relaxation(MeshData, NumSamples, IterationCount, PlottingSample = None, ScalingFactor = 2):
    #Mesh Vertex Values
    Vertexes = MeshData.Vertexes
    Points = Vertexes.Value

    #Mesh Face Values
    Faces = MeshData.Faces
    FaceIndicies = Faces.VertexIndiciesZeroBased
    FaceValues = Faces.VertexValues
    FacesTotalArea = Faces.TotalArea
    PyvistaFaces = Faces.PyVista

    #Store Mesh Data
    storeData('StoredData/Faces', Faces)
    storeData('StoredData/PyvistaFaces', PyvistaFaces)
    storeData('StoredData/Points', Points)
    storeData('StoredData/Vertexes', Vertexes)
    storeData('StoredData/FaceIndicies', FaceIndicies)
    storeData('StoredData/FaceValues', FaceValues)
    storeData('StoredData/FacesTotalArea', FacesTotalArea)

    #Calculate Average Face Area
    AverageFaceArea = FacesTotalArea/len(FaceValues)

    #PyVista Initalisation
    Mesh = pv.PolyData()
    Mesh.points = Points
    Mesh.faces = PyvistaFaces

    #Sample Points on the Mesh, and Group all the sampled points by their associated Face
    Samples, SampleFaces = SampleFacesMethod(Vertexes, Faces, NumSamples)
    print("Samples Done")

    GroupedSamplePointsByFace = {key: [item[0] for item in group] for key, group in groupby(sorted(enumerate(SampleFaces), key=lambda x: x[1]), lambda x: x[1])}
    print("Grouping Samples Done")

    RepulsionRadius = math.sqrt(FacesTotalArea/NumSamples)*2
    print(f"Average Face Area: {AverageFaceArea}")
    print(f"Repulsion Radius: {RepulsionRadius}")

    #Get all face neighbours and store into dict
    FaceNeighbours = {}
    for Face in range(0,len(FaceIndicies)):
        FaceNeighbours[Face] = (GetNeighbouringFacesByEdge(Face, Faces))
    #List to store each iterations values !!!!!!!!!!!!!!!!!!!!
    # IterationValueList = []

    #Calculate initial setup values for first iteration
    InitalSetupObject = IterationHolder()
    InitialRelaxationSetup = Relaxation_Setup(InitalSetupObject, Samples, SampleFaces, FaceNeighbours, FaceValues, GroupedSamplePointsByFace, Faces, RepulsionRadius, Points, PyvistaFaces)
    print("Relaxation Setup Done")

    for Iteration in range(0, IterationCount):
        print(f"\n/////\nIteration {Iteration}\n/////")
        
        #Create new class item to store iterations contents
        IterationObject = IterationHolder()

        #Get previous iterations values to iterate on
        if Iteration > 0:
            MappedNewPoint, NewFace = PreviousIteration.getMappedPointAndFace()
            Samples = np.array(MappedNewPoint)
            SampleFaces = NewFace
            GroupedSamplePointsByFace = {key: [item[0] for item in group] for key, group in groupby(sorted(enumerate(SampleFaces), key=lambda x: x[1]), lambda x: x[1])}

        else:
            PreviousIteration = InitialRelaxationSetup

        #Apply relaxation repulsion process
        RepulsionIterationObject = Relaxation_Repulsion(PreviousIteration, IterationObject, Samples, SampleFaces, FaceNeighbours, Mesh, ScalingFactor, Faces, Points, PyvistaFaces, RepulsionRadius)
        
        #Gather values
        IterationObject = Relaxation_Setup(RepulsionIterationObject, Samples, SampleFaces, FaceNeighbours, FaceValues, GroupedSamplePointsByFace, Faces, RepulsionRadius, Points, PyvistaFaces)
    
        #Add values to iteration object
        PreviousIteration = IterationObject

        #Reduce Scaling Factor to converge
        ScalingFactor = ScalingFactor * 0.8

    storeData('StoredData/PreviousIteration', PreviousIteration)
    MappedNewPoint, NewFace = PreviousIteration.getMappedPointAndFace()
    storeData('StoredData/SamplePoints', MappedNewPoint)
    storeData('StoredData/SampleFaces', NewFace)

    print("\n===== Calculating Voronoi Regions =====") #Pass last iteration object into function
    PlotSample(Faces, Points, PyvistaFaces, FaceNeighbours = FaceNeighbours, IterationObject = IterationObject, PlottingSample = None)
    GroupedSamplePointsByFace = {key: [item[0] for item in group] for key, group in groupby(sorted(enumerate(SampleFaces), key=lambda x: x[1]), lambda x: x[1])}
    
    Voronroi(Faces, Points, PyvistaFaces, FaceNeighbours, IterationObject, GroupedSamplePointsByFace)
      
    print("\n===== Saving Mesh =====")
    # SaveMesh(Mesh, 'RelaxaedMesh')


    print("\n===== Plotting Sample =====")
    try:
        # PlotSample(Faces, Points, PyvistaFaces, FaceNeighbours = FaceNeighbours, IterationObjectList = IterationValueList, PlottingSample = PlottingSample)
        pass
    except:
        traceback.print_exc()

    return

#Maps sample points onto FaceB's Plane
#https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d
def MapToPlane(FaceCoordinates, Point):
    #Calculate Plane Unit Normal
    PlaneUnitNormal = np.array(UnitNormal(FaceCoordinates[0],FaceCoordinates[1],FaceCoordinates[2]))
    nx, ny, nz = PlaneUnitNormal

    PointPlaneVector = Point - FaceCoordinates[0]
    vx, vy, vz = PointPlaneVector

    Dist = (nx*vx + ny*vy + nz*vz)
    NewPoint = Point - (Dist*PlaneUnitNormal)

    #Sanity Check
    Distance = PointPlaneDistance(NewPoint, FaceCoordinates)
    if Distance != 0:
        print("Mapping Problem: Distance < 0.05")

    return NewPoint.tolist(), Distance

def DetermineFace(CurrentFace, Sample, Faces):
    #Get all faces
    FaceIndicesList = Faces.VertexIndicies
    FaceValuesList = Faces.VertexValues

    #Determine current  faces coordinates
    FaceCoordinates = FaceValuesList[CurrentFace]
    FaceIndices = FaceIndicesList[CurrentFace]

    ResultantBarycentricWeights = CartesianToBarycentric(FaceCoordinates, Sample)
    Edge, EdgeValues = BarycentricClosestEdge(FaceCoordinates, FaceIndices, ResultantBarycentricWeights)

    return Edge, EdgeValues, ResultantBarycentricWeights
  
def MapResultant(Face, Point, OldPoint, PackedVar):
    #Unpack
    Faces, Points, PyvistaFaces, FaceNeighbours = PackedVar

    #Determine current faces coordinates
    FaceIndicesList = Faces.VertexIndicies
    FaceValuesList = Faces.VertexValues
    FaceCoordinates = FaceValuesList[Face]
    CurrentFaceNeighbours = FaceNeighbours[Face]

    #Determine New Face (if there is one)
    Edge, EdgeValues, ResultantBarycentricWeights = DetermineFace(Face, Point, Faces)
    PointPlaneDist = PointPlaneDistance(Point, FaceCoordinates)

    #Point remains on current face
    if Edge is None:
        if PointPlaneDist != 0:
            Point, _ = MapToPlane(FaceValuesList[Face], Point)
        return Point, Face, ResultantBarycentricWeights, EdgeValues
    
    #Point no longer remains on the current face
    MapFlag = False
    for Neighbour in CurrentFaceNeighbours:
        #Check if neighnour is valid (mappable)
        if Neighbour != None:
            NeighbourVertices = FaceIndicesList[Neighbour]
            
            #Check if Neighbourhood Vertices match Edge Vertices
            if Edge[0] in NeighbourVertices and Edge[1] in NeighbourVertices:
                
                #Map point to neighbours plane
                NewPoint, _ = MapToPlane(FaceValuesList[Neighbour], Point)
                NewFace = Neighbour
                MapFlag = True
                break

        else:
            NoneFlag = True

    #Check if new mapped point is on a face
    if MapFlag == True:
        UpdatedEdge, UpdatedEdgeValues, UpdatedResultantBarycentricWeights = DetermineFace(NewFace, NewPoint, Faces)
        PointPlaneDist = PointPlaneDistance(NewPoint, FaceValuesList[NewFace])
    
    #Check if NewPoint remains on the NewFace
    if UpdatedEdge is None and PointPlaneDist == 0:
        return NewPoint, NewFace, UpdatedResultantBarycentricWeights, UpdatedEdgeValues

    #NewPoint doesnt lie on NewFace
    else:
        Count = 0
        Flag = False
        while Flag != True:
            Count += 1
            if Count > 5:
                    # print(f"New Point {NewPoint} ({NewFace})\n  UpdatedEdge: {UpdatedEdge} - {UpdatedEdgeValues} - {UpdatedResultantBarycentricWeights}")
                    # SimplePlot(Faces, Points, PyvistaFaces, NewPoint, Point, Face, NewFace, Neighbours = None, Edge = UpdatedEdgeValues)

                    ContigencyPoint = CloestPointOnSegment(UpdatedEdgeValues[0], UpdatedEdgeValues[1], NewPoint)
                    ContigencyFace = NewFace
                    ContingencyEdge, ContingencyEdgeValues, ContingencyResultantBarycentricWeights = DetermineFace(ContigencyFace, ContigencyPoint, Faces)
                    ContingencyPointPlaneDist = PointPlaneDistance(ContigencyPoint, FaceValuesList[ContigencyFace])
                    return ContigencyPoint, ContigencyFace, ContingencyResultantBarycentricWeights, ContingencyEdgeValues   
            
            #Check new neighbours edges
            NewFaceNeighbours = FaceNeighbours[NewFace]
            for Neighbour in NewFaceNeighbours:
                if Neighbour != None:
                    NeighbourVertices = Faces.VertexIndicies[Neighbour]
                    
                    if UpdatedEdge[0] in NeighbourVertices and UpdatedEdge[1] in NeighbourVertices:
                        #Map point to neighbours plane
                        NewPoint, _ = MapToPlane(FaceValuesList[Neighbour], NewPoint)
                        NewFace = Neighbour
                        MapFlag = True
                        break

            #Check if new mapped point is on a face
            if MapFlag == True:
                UpdatedEdge, UpdatedEdgeValues, UpdatedResultantBarycentricWeights = DetermineFace(NewFace, NewPoint, Faces)
                PointPlaneDist = PointPlaneDistance(NewPoint, FaceValuesList[NewFace])
                    
            #NewPoint doesnt leave (iterative) NewFace
            if UpdatedEdge is None and PointPlaneDist == 0:
                return NewPoint, NewFace, UpdatedResultantBarycentricWeights, UpdatedEdgeValues            

def toUV(Triangle, Coordinates, NPArray = True, PreserveDict = False):
    # ### The original computation of the plane equation ###
    # Given points p1 and p2, the vector through them is W = (p2 - p1) We want the plane equation Ax + By + Cz + d = 0, and to make the plane prepandicular to the vector, we set (A, B, C) = W
    p1 = np.array(Triangle[0])
    p2 = np.array(Triangle[1])
    p3 = np.array(Triangle[2])
    A, B, C = W = p2 - p1
    D = -1 * np.dot(W, p1)

    # ### Normalizing W ###
    magnitude = np.linalg.norm(W)
    normal = W / magnitude

    # We take a vector U that we know that is perpendicular to W, but we also need to make sure it's not zero.
    if A != 0:
        u_not_normalized = np.array([B, -A, 0])
    else:
        # If A is 0, then either B or C have to be nonzero
        u_not_normalized = np.array([0, B, -C])
    u_magnitude = np.linalg.norm(u_not_normalized)

    # ### Normalizing W ###
    U = u_not_normalized / u_magnitude
    V = np.cross(normal, U)

    # Now, for a point p3 = (x3, y3, z3) it's (u, v) coordinates would be computed relative to our reference point (p1)
    p3_u = np.dot(U, p3 - p1)
    p3_v = np.dot(V, p3 - p1)

    if isinstance(Coordinates, np.ndarray):
        U_Coord = np.dot(U, Coordinates - p1)
        V_coord = np.dot(V, Coordinates - p1)
        
        if NPArray:
            return np.array([U_Coord, V_coord])
        else:
            return [U_Coord, V_coord]
    
    if not Coordinates:
        return []
    
    CoordinatesType = type(Coordinates[0])
    if PreserveDict:
        UVCoordinateDict = {}
        
        if CoordinatesType == dict:
            DictValues = NestedDictValues(Coordinates)
            for SampleIndex, Sample in DictValues:
                U_Coord = np.dot(U, Sample - p1)
                V_coord = np.dot(V, Sample - p1)
                SetKey(UVCoordinateDict, SampleIndex, [U_Coord, V_coord])
        
        if CoordinatesType == list:

            MergedDict = {}
            for d in Coordinates:
                MergedDict.update(d)
            # print(f"MergedDict: {MergedDict} - Type: {type(MergedDict)}")

            for SampleIndex, SampleValue in MergedDict.items():
                U_Coord = np.dot(U, SampleValue - p1)
                V_coord = np.dot(V, SampleValue - p1)
                SetKey(UVCoordinateDict, SampleIndex, [U_Coord, V_coord])

        return UVCoordinateDict

    if not PreserveDict:
        CoordinatesArray = []

        if CoordinatesType == dict:
            DictValues = GetDictItems(Coordinates, True)
            # print(f"DICT VALUES: {DictValues}")
            for Sample in DictValues:
                U_Coord = np.dot(U, Sample - p1)
                V_coord = np.dot(V, Sample - p1)
                CoordinatesArray.append([U_Coord, V_coord])

        if CoordinatesType == list:
            for Sample in Coordinates:
                U_Coord = np.dot(U, Sample - p1)
                V_coord = np.dot(V, Sample - p1)
                CoordinatesArray.append([U_Coord, V_coord])

        if NPArray:
            return np.array(CoordinatesArray)
        else:
            return CoordinatesArray

def fromUV(Triangle, Coordinates, NPArray = True):
    # ### The original computation of the plane equation ###
    # Given points p1 and p2, the vector through them is W = (p2 - p1) We want the plane equation Ax + By + Cz + d = 0, and to make the plane prepandicular to the vector, we set (A, B, C) = W
    p1 = np.array(Triangle[0])
    p2 = np.array(Triangle[1])
    p3 = np.array(Triangle[2])
    A, B, C = W = p2 - p1
    D = -1 * np.dot(W, p1)

    # ### Normalizing W ###
    magnitude = np.linalg.norm(W)
    normal = W / magnitude

    # We take a vector U that we know that is perpendicular to W, but we also need to make sure it's not zero.
    if A != 0:
        u_not_normalized = np.array([B, -A, 0])
    else:
        # If A is 0, then either B or C have to be nonzero
        u_not_normalized = np.array([0, B, -C])
    u_magnitude = np.linalg.norm(u_not_normalized)

    # ### Normalizing W ###
    U = u_not_normalized / u_magnitude
    V = np.cross(normal, U)

    # And to convert the point back to 3D, we just use the same reference point and multiply U and V by the coordinates
    # ConvertedPoint = p1 + p3_u * U + p3_v * V

    if isinstance(Coordinates, np.ndarray):
        U_Coord = Coordinates[0]
        V_coord = Coordinates[1]
        ConvertedPoint = p1 + U_Coord * U + V_coord * V
        if NPArray:
            return np.array(ConvertedPoint)
        else:
            return ConvertedPoint
    
    if not Coordinates:
        return []
    
    NewCoordinates = []
    for Coord in Coordinates:
        U_Coord = Coord[0]
        V_coord = Coord[1]
        ConvertedPoint = p1 + U_Coord * U + V_coord * V   
        NewCoordinates.append(ConvertedPoint)
        
    if NPArray:
        return np.array(NewCoordinates)
    else:
        return NewCoordinates

def GetPlaneUV(Triangle):
    a,b,c,d = PlaneEquation(Triangle)
    
    PlaneUnitNormal = np.array(UnitNormal(Triangle[0],Triangle[1],Triangle[2]))
    nx, ny, nz = PlaneUnitNormal
    N = [nx,ny,nz]

    U = NormaliseVector([b, -a, 0])
    V = np.array(VectorCrossProduct(N, U))

    return (U,V), Triangle[0]

def CalculateInfluence(SamplePoint, NearbyPoints, RepulsionRadius, DecayConstant = 1, Sigma = 0.4):
    
    #Initailly check if NearbyPoints is not empty
    if len(NearbyPoints) == 0:
        return None, None
    
    #Unpack the Variable
    x0, y0, z0 = SamplePoint
    NearbyPoints = np.array(NearbyPoints)

    #Calculate Diffs
    Diffs = SamplePoint - NearbyPoints

    #Calculate Distances
    Distances = np.sqrt((NearbyPoints[:,0] - x0)**2 + (NearbyPoints[:,1] - y0)**2 + (NearbyPoints[:,2] - z0)**2)

    #Normalise Distances
    NormalisedDistances = Distances/RepulsionRadius

    #Calculate Weights
    NormalisedWeights = np.exp(-DecayConstant * NormalisedDistances)

    return NormalisedWeights, Diffs

#https://github.com/nortikin/sverchok/blob/master/utils/voronoi3d.py
#https://stackoverflow.com/questions/85275/how-do-i-derive-a-voronoi-diagram-given-its-point-set-and-its-delaunay-triangula
#https://stackoverflow.com/questions/53696900/render-voronoi-diagram-to-numpy-array
#https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647
#use circumference of circle going through points
def Voronroi(Faces, Points, PyvistaFaces, FaceNeighboursList, IterationObject, GroupedSamplePointsByFace):
    LastObject = IterationObject
    NearbySamples = LastObject.getNearbySamples()
    NearbySampleIndices = LastObject.getNearbySampleIndices()

    MappedNewPoint, NewFace = LastObject.getMappedPointAndFace()

    #Create Voronoi Sample Arrays
    Regions = []
    RegionNeighbours = []
    RegionNeighboursValues = []
    RegionNeighbourIndices = []
    RegionEdgeLengths = []
    RegionDistances = []
    FailedVoronoiCount = 0

    #Loop through each sample point
    for Point, Face, Index in zip(MappedNewPoint, NewFace, range(0,len(MappedNewPoint))):
        FaceCoordinates = Faces.VertexValues[Face]

        NearbySample = NearbySamples[Index]
        NearbyIndices = NearbySampleIndices[Index]

        #Convert to Barycentric
        try:
            Point_Bary = CartesianToBarycentric(FaceCoordinates, Point)
            NearbySamples_Bary = CartesianToBarycentric(FaceCoordinates, NearbySample, True)
        except:
            print(f"///////////////////////////////////////////////////////////////")
            traceback.print_exc()
            print(f"========================\nNearbySamples: {NearbySample} - Type{type(NearbySample)}")
            print(f"///////////////////////////////////////////////////////////////")

        #Determine Voronoi Adjacency
        try:
            BaryValues = MergeUVValues(Point_Bary, NearbySamples_Bary)
            BarySamplePointRegion, SamplePointNeighbours, SampleNeighbourValues, SampleNeighbourIndices, SampleEdgeLengths, SampleDistances = DetermineVoronoiAdjacency(BaryValues, NearbyIndices, False)
        except:
            # print(f"///////////////////////////////////////////////////////////////")
            # traceback.print_exc()
            # print(f"========================\nBaryValues: {BaryValues}-{type(BaryValues)}\n")
            # print(f"========================\nPoint_Bary: {Point_Bary}-{type(Point_Bary)}\n")
            # print(f"========================\nMappedNeighbours_Bary: {MappedNeighbours_Bary}-{type(MappedNeighbours_Bary)}\n")
            # print(f"========================\nFaceSamples_Bary: {FaceSamples_Bary}-{type(FaceSamples_Bary)}\n")
            # print(f"========================\nNearbySamples: {NearbySamples_Bary}-{type(NearbySamples_Bary)}\n")
            # print(f"///////////////////////////////////////////////////////////////")
            FailedVoronoiCount += 1
            BarySamplePointRegion = []

        #Convert back to Cartesian
        VoronoiRegion, VoronoiFaces = VoronoiMapping(BarySamplePointRegion, Face, Faces, FaceNeighboursList)
        SampleNeighbourSamples = BarycentricToCartesian(SampleNeighbourValues, FaceCoordinates, True)
        

        # SimplePlot(Faces, Points, PyvistaFaces, SampleNeighbourSamples, Point, Face, FaceNeighboursList, None, Var = VoronoiRegion)
        Regions.append(VoronoiRegion)
        RegionNeighbours.append(SamplePointNeighbours)
        RegionNeighboursValues.append(SampleNeighbourSamples)
        RegionNeighbourIndices.append(SampleNeighbourIndices)
        RegionEdgeLengths.append(SampleEdgeLengths)
        RegionDistances.append(SampleDistances)

    #Plot Voronoi
    print(f"Failed Voronoi Count: {FailedVoronoiCount}")


    storeData('StoredData/Regions', Regions)
    storeData('StoredData/RegionNeighbours', RegionNeighbours)
    storeData('StoredData/RegionNeighboursValues', RegionNeighboursValues)
    storeData('StoredData/RegionNeighbourIndices', RegionNeighbourIndices)

    storeData('StoredData/RegionEdgeLengths', RegionEdgeLengths)
    storeData('StoredData/RegionDistances', RegionDistances)

    PlotVoronoi(Faces, Points, PyvistaFaces, Regions)

def VoronoiMapping(VoronoiRegion, Face, Faces, FaceNeighbours):
    # print(f"==== Voronoi Mapping ====")
    # print(VoronoiRegion)

    VoronoiVertices = []
    VoronoiFaces = []

    FaceValuesList = Faces.VertexValues
    FaceIndicesList = Faces.VertexIndicies

    FaceCoordinates = FaceValuesList[Face]
    CurrentFaceNeighbours = FaceNeighbours[Face]

    for BarycentricVertex in VoronoiRegion:
        #Convert Barycentric Coordinates to Cartesian
        Vertex = BarycentricToCartesian(BarycentricVertex, FaceCoordinates)
        # print(f"Vertex: {BarycentricVertex} -> {Vertex}")

        #Determine if Voronoi Vertex lies on current face, and its distance
        Edge, EdgeValues, ResultantBarycentricWeights = DetermineFace(Face, Vertex, Faces)
        PointPlaneDist = PointPlaneDistance(Vertex, FaceCoordinates)

        #Point remains on current face
        if Edge is None:
            if PointPlaneDist != 0:
                Vertex, _ = MapToPlane(FaceValuesList[Face], Vertex)
            VoronoiVertices.append(Vertex)
            VoronoiFaces.append(Face)
            continue

        #Point no longer remains on the current face  
        MapFlag = False
        for Neighbour in CurrentFaceNeighbours:  
            #Check if neighnour is valid (mappable)
            if Neighbour != None:
                NeighbourVertices = FaceIndicesList[Neighbour]

                #Check if Neighbourhood Vertices match Edge Vertices
                if Edge[0] in NeighbourVertices and Edge[1] in NeighbourVertices:
                    
                    #Map point to neighbours plane
                    NewVertex, _ = MapToPlane(FaceValuesList[Neighbour], Vertex)
                    NewFace = Neighbour
                    MapFlag = True
                    break

            else:
                NoneFlag = True

        #Check if new mapped point is on a face
        if MapFlag == True:
            UpdatedEdge, UpdatedEdgeValues, UpdatedResultantBarycentricWeights = DetermineFace(NewFace, NewVertex, Faces)
            PointPlaneDist = PointPlaneDistance(NewVertex, FaceValuesList[NewFace])

        # print(f"Critical: {UpdatedEdge}-{PointPlaneDist}")

        #Check if NewPoint remains on the NewFace
        if UpdatedEdge is None and PointPlaneDist == 0:
            VoronoiVertices.append(NewVertex)
            VoronoiFaces.append(NewFace)
            continue
        
        #NewPoint doesnt lie on NewFace
        else:
            Count = 0
            Flag = False
            while Flag != True:
                Count += 1
                MapFlag = False

                if Count > 5:
                    # print(f"New Point {NewVertex} ({NewFace}) ({PointPlaneDist})\n  UpdatedEdge: {UpdatedEdge} - {UpdatedEdgeValues} - {UpdatedResultantBarycentricWeights}")
                    # SimplePlot(Faces, Points, PyvistaFaces, NewPoint, Point, Face, NewFace, Neighbours = None, Edge = UpdatedEdgeValues)

                    ContigencyPoint = CloestPointOnSegment(UpdatedEdgeValues[0], UpdatedEdgeValues[1], NewVertex)
                    ContigencyFace = NewFace

                    # ContingencyEdge, ContingencyEdgeValues, ContingencyResultantBarycentricWeights = DetermineFace(ContigencyFace, ContigencyPoint, Faces)
                    # ContingencyPointPlaneDist = PointPlaneDistance(ContigencyPoint, FaceValuesList[ContigencyFace])

                    VoronoiVertices.append(ContigencyPoint)
                    VoronoiFaces.append(ContigencyFace)
                    Flag = True
                    continue    
                
                #Check new neighbours edges
                NewFaceNeighbours = FaceNeighbours[NewFace]
                for Neighbour in NewFaceNeighbours:
                    
                    if Neighbour != None:
                        NeighbourVertices = Faces.VertexIndicies[Neighbour]
                        
                        if UpdatedEdge[0] in NeighbourVertices and UpdatedEdge[1] in NeighbourVertices:
                            
                            #Map point to neighbours plane
                            NewVertex, _ = MapToPlane(FaceValuesList[Neighbour], NewVertex)
                            NewFace = Neighbour
                            MapFlag = True
                            break
                
                #Check if new mapped point is on a face
                if MapFlag == True:
                    UpdatedEdge, UpdatedEdgeValues, UpdatedResultantBarycentricWeights = DetermineFace(NewFace, NewVertex, Faces)
                    PointPlaneDist = PointPlaneDistance(NewVertex, FaceValuesList[NewFace])

                #NewPoint doesnt leave (iterative) NewFace
                if UpdatedEdge is None and PointPlaneDist == 0:
                    VoronoiVertices.append(NewVertex)
                    VoronoiFaces.append(NewFace)
                    Flag = True 
                
    return VoronoiVertices, VoronoiFaces
