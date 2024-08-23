import math
import pickle
import numpy as np
from matplotlib.colors import Normalize
from itertools import groupby
import pyvista as pv
import traceback
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

#https://faculty.cc.gatech.edu/~turk/
#https://github.com/gollygang/ready

class IterationHolder(object):
    def __init__(self):
        #Create dicts to store values
        self.ResultantSamples = []
        self.MappedResultantSample = []
        self.ResultantFaces = []
        self.ResultantForces = []
        self.ResultantBarycentric = []
        self.ResultantEdge = []

        #Dictionaries to store all values for each given sample
        self.FaceSampleValues = {}
        self.NeighbourSamples = {}
        self.NeighbourSampleValues = {}
        self.MappedNeighbourSampleValues = {}
        self.NearbySamples = {}
        self.NearbySampleDistances = {}
        self.BarycentricWeights = {}

    def setSampleValues(self, FaceSampleValues, NeighbourSamples, NeighbourSampleValues, MappedNeighbourSampleValues, NearbySamples, NearbySampleDistances, BarycentricWeights):
        self.FaceSampleValues = FaceSampleValues
        self.NeighbourSamples = NeighbourSamples
        self.NeighbourSampleValues = NeighbourSampleValues
        self.MappedNeighbourSampleValues = MappedNeighbourSampleValues
        self.NearbySamples = NearbySamples
        self.NearbySampleDistances = NearbySampleDistances
        self.BarycentricWeights = BarycentricWeights

    def getSampleValues(self):
        return self.FaceSampleValues, self.NeighbourSamples, self.NeighbourSampleValues, self.MappedNeighbourSampleValues, self.NearbySamples, self.NearbySampleDistances, self.BarycentricWeights

    def setValues(self, MappedNewPoint, NewPoint, NewFace, ResultantReplusiveForce, ResultantBarycentricWeights, EdgeValues):
        self.MappedResultantSample.append(MappedNewPoint) 
        self.ResultantSamples.append(NewPoint)
        self.ResultantFaces.append(NewFace)
        self.ResultantForces.append(ResultantReplusiveForce)
        self.ResultantBarycentric.append(ResultantBarycentricWeights)
        self.ResultantEdge.append(EdgeValues)

    def getValues(self):
        return(self.ResultantSamples,
        self.MappedResultantSample,
        self.ResultantFaces,
        self.ResultantForces,
        self.ResultantBarycentric,
        self.ResultantEdge)

def GrayScott(Model, MatrixA, MatrixB, DA, DB, Feed, Kill, TimeDelta):
    #Sanity Check
    if Model == 'GrayScott':
        #Get Diffusion
        DiffMatrixA = ApplyLaplacian(MatrixA) 
        DiffMatrixB = ApplyLaplacian(MatrixB)

        DeltaA = DA * DiffMatrixA
        DeltaB = DB * DiffMatrixB

        #Reaction
        Reaction = MatrixA * (MatrixB ** 2)
        DeltaA -= Reaction
        DeltaB += Reaction

        #Feed and Kill
        DeltaA += Feed * (1-MatrixA)
        DeltaB -= (Kill+Feed) * MatrixB

        MatrixA += DeltaA * TimeDelta
        MatrixB += DeltaB * TimeDelta

        return(MatrixA, MatrixB)
    
    else:
        return -1

def Meinhardt(Model, MatrixG1, MatrixG2, MatrixR, MatrixS1, MatrixS2, kAB, kC, kDE, DiffG, DiffS, TimeDelta):
    #Sanity Check
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

#I changed the order to be counter-clockwise
def NormalVector(Face):
    PointA = Face[0]
    PointB = Face[1]
    PointC = Face[2]

    Vector1 = PointB - PointA
    Vector2 = PointC - PointA

    NormalVector = NormaliseVector(list(VectorCrossProduct(Vector1, Vector2)))

    return NormalVector

#Tom Smilack, 2012. Stack overflow, Available at: https://stackoverflow.com/a/12643315    
#Get Matrox Det
def Determinant(Matrix):
    return Matrix[0][0]*Matrix[1][1]*Matrix[2][2] + Matrix[0][1]*Matrix[1][2]*Matrix[2][0] + Matrix[0][2]*Matrix[1][0]*Matrix[2][1] - Matrix[0][2]*Matrix[1][1]*Matrix[2][0] - Matrix[0][1]*Matrix[1][0]*Matrix[2][2] - Matrix[0][0]*Matrix[1][2]*Matrix[2][1]

#Unit normal vector of plane defined by points a, b, and c
def UnitNormal(a, b, c):
    x = Determinant([[1,a[1],a[2]],[1,b[1],b[2]],[1,c[1],c[2]]])
    y = Determinant([[a[0],1,a[2]],[b[0],1,b[2]],[c[0],1,c[2]]])
    z = Determinant([[a[0],a[1],1],[b[0],b[1],1],[c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5

    return [x/magnitude, y/magnitude, z/magnitude]

#dot product of vectors a and b
def VectorDotProduct(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

#cross product of vectors a and b
def VectorCrossProduct(VectorA, VectorB):
    x = VectorA[1] * VectorB[2] - VectorA[2] * VectorB[1]
    y = VectorA[2] * VectorB[0] - VectorA[0] * VectorB[2]
    z = VectorA[0] * VectorB[1] - VectorA[1] * VectorB[0]
    return (x, y, z)

#https://www.geeksforgeeks.org/how-to-normalize-an-array-in-numpy-in-python/
def NormaliseVector(Vector):
    NormalisedVector = (Vector-np.min(Vector))/(np.max(Vector)-np.min(Vector))
    return NormalisedVector

def GetUnitVector(Vector):
    UnitVector = Vector / np.linalg.norm(Vector)
    return UnitVector

#Maps FaceA's Face and any of its sample points onto FaceB's Plane
def RotateToPlane(FaceAIndex, FaceBIndex, FaceASample, Faces, Vertices, Debug=False):
    FaceIndices = Faces.VertexIndiciesZeroBased
    VertexValues = Vertices.Value

    #Fetch vertex indicies from face, and their coresponding coordinates
    FaceAVerticies = FaceIndices[FaceAIndex]
    FaceBVerticies = FaceIndices[FaceBIndex]
    FaceAVertexValues = VertexValues[FaceAVerticies]
    FaceBVertexValues = VertexValues[FaceBVerticies]

    #Calculate the shared and unshared verticies between the two faces
    SharedVertices = GetIntersectingRows(FaceAVertexValues, FaceBVertexValues)
    FaceAVertex = GetNonIntersectionRows(FaceAVertexValues, SharedVertices)[0]
    FaceBVertex = GetNonIntersectionRows(FaceBVertexValues, SharedVertices)[0]

    #Calculate Vector between the shared points
    SharedEdgeVector = SharedVertices[0] - SharedVertices[1]

    #Calculate vector that lie in the plane A and plane B
    PlaneAVector = FaceAVertex - SharedVertices[0]
    PlaneBVector = FaceBVertex - SharedVertices[0]

    #Calculate Cross-products to determine coefficients for plane equations, as well as dot-product to determine last coefficient
    PlaneANormal = np.cross(PlaneAVector, SharedEdgeVector)
    a1, b1, c1 = PlaneANormal
    d1 = -np.dot(PlaneANormal, FaceAVertex)
    PlaneBNormal = np.cross(PlaneBVector, SharedEdgeVector)
    a2, b2, c2 = PlaneBNormal
    d2 = -np.dot(PlaneBNormal, FaceBVertex)

    PlaneBUnitNormal = GetUnitVector(PlaneBNormal)
    a, b, c = PlaneBUnitNormal

    #Calculate the angle between two planes given their equations
    temp = (a1 * a2 + b1 * b2 + c1 * c2)
    tempA = math.sqrt(a1 * a1 + b1 * b1 + c1 * c1)
    tempB = math.sqrt(a2 * a2 + b2 * b2 + c2 * c2)
    d = temp / (tempA * tempB)
    Angle = (math.acos(d)) #Radians
    RemainingAngle = math.pi - Angle

    #Rotate the Sample Value around the arbitary axis of the edge

    #Need to determine wether to rotate it RemainingAngle or -Remaining Angle
    Rotated1 = RotatePointAroundLine(FaceASample, SharedVertices[0], SharedEdgeVector, RemainingAngle)
    Rotated2 = RotatePointAroundLine(FaceASample, SharedVertices[0], SharedEdgeVector, -RemainingAngle)

    Rotated1Distance = PointPlaneDistance(Rotated1, (a2, b2, c2, d2))
    Rotated2Distance = PointPlaneDistance(Rotated2, (a2, b2, c2, d2))
    
    if Rotated1Distance < Rotated2Distance:
        NewPoint = Rotated1
        Distance = Rotated1Distance
    else:
        NewPoint = Rotated2
        Distance = Rotated2Distance

    #DEBUG - Splits remaining angle into steps
    if Debug:
        Steps = 10
        RotatedVerticies = []
        DividedAngle = RemainingAngle/Steps
        AngleStep = 0
        for x in range(0, Steps):
            AngleStep += DividedAngle
            RotatedVertex = RotatePointAroundLine(FaceAVertex, SharedVertices[0], SharedEdgeVector, AngleStep)
            RotatedVerticies.append(RotatedVertex)
            print(f"AngleStep {x}: {AngleStep}")

        print('The equation is {0}x + {1}y + {2}z = {3}'.format(a1, b1, c1, d1))
        print('The equation is {0}x + {1}y + {2}z = {3}'.format(a2, b2, c2, d2))
        print(f"Plane A Vetex: {FaceAVertex}")
        print(f"Angle is {math.degrees(RemainingAngle)} Degrees")
        print(f"Shared Edge Vector: {SharedEdgeVector}")
        print(f"{Rotated1Distance} VS {Rotated2Distance}")
        return Rotated1, Rotated2, Angle, SharedVertices, SharedEdgeVector, Rotated1Distance, Rotated2Distance
    
    VectorLine = np.array([(SharedVertices[0] - (SharedEdgeVector * 2)).tolist(), (SharedVertices[0] - SharedEdgeVector).tolist(), SharedVertices[0].tolist(), (SharedVertices[0] + SharedEdgeVector).tolist()])

    return NewPoint.tolist(), SharedVertices, SharedEdgeVector, Distance


#https://stackoverflow.com/a/8317403
def GetIntersectingRows(Array1, Array2):
    Rows, Cols = Array1.shape
    dtype = {'names':['f{}'.format(i) for i in range(Cols)], 'formats':Cols * [Array1.dtype]}

    Intersect = np.intersect1d(Array1.view(dtype), Array2.view(dtype))
    Intersect = Intersect.view(Array1.dtype).reshape(-1, Cols)
    return Intersect

#https://stackoverflow.com/a/71001217
def GetNonIntersectionRows(Array1, Array2):
    Rows, Cols = Array1.shape
    NonIntersect = Array1[~np.all(Array1==Array2[:, None], axis=2).any(axis=0)]

    return NonIntersect


#https://sites.google.com/site/glennmurray/glenn-murray-ph-d/rotation-matrices-and-formulas/rotation-about-an-arbitrary-axis-in-3-dimensions
def RotatePointAroundLine(RotationVertex, LineVertex, DirectionVector, Theta):
    A, B, C = LineVertex
    U, V, W = GetUnitVector(DirectionVector)
    X, Y, Z = RotationVertex

    #Define Intermediate Values
    U2 = U*U
    V2 = V*V
    W2 = W*W
    CosT = np.cos(Theta)
    SinT = np.sin(Theta)
    OneMinusCosT = 1-CosT

    #Perform Calculations
    NewX = (A * (V2 + W2) - U*(B*V + C*W - U*X - V*Y - W*Z)) * OneMinusCosT + X*CosT + (-C*V + B*W - W*Y + V*Z) * SinT
    NewY = (B * (U2 + W2) - V*(A*U + C*W - U*X - V*Y - W*Z)) * OneMinusCosT + Y*CosT + (C*U - A*W + W*X - U*Z) * SinT
    NewZ = (C * (U2 + V2) - W*(A*U + B*V - U*X - V*Y - W*Z)) * OneMinusCosT + Z*CosT + (-B*U + A*V - V*X + U*Y) * SinT
    NewVertex = [NewX, NewY, NewZ]

    return NewVertex

# Get all faces with certain vertex
# https://github.com/pyvista/pyvista-support/issues/96
def FindFacesWithVertex(Index, Faces):
    FaceIndicies = Faces.VertexIndiciesZeroBased
    TouchingFaces = [i for i, face in enumerate(FaceIndicies) if Index in face]
    return TouchingFaces

# Get all Vertex Indicies connecting to a Node
def FindConnectedVerticies(index, Faces):
    FaceIndicies = Faces.VertexIndiciesZeroBased
    cids = FindFacesWithVertex(index, Faces)
    connected = np.unique(FaceIndicies[cids].ravel())
    Out = np.delete(connected, np.argwhere(connected == index))

    return Out

# Find Neighbouring Faces
def GetNeighbouringFaces(FaceIndex, Faces):
    FaceIndicies = Faces.VertexIndiciesZeroBased
    Face = FaceIndicies[FaceIndex]
    Sharing = set()
    for vid in Face:
        [Sharing.add(f) for f in FindFacesWithVertex(vid, Faces)]
        # {Sharing[FaceIndex].append(f) for f in FindFacesWithVertex(vid, Faces) if f != FaceIndex}
    Sharing.remove(FaceIndex)

    return list(Sharing)

def GetNeighbouringFacesByEdge(Index, Faces):
    FaceIndicies = Faces.VertexIndiciesZeroBased
    face = FaceIndicies[Index]
    a = set(f for f in FindFacesWithVertex(face[0], Faces))
    a.remove(Index)
    b = set(f for f in FindFacesWithVertex(face[1], Faces))
    b.remove(Index)
    c = set(f for f in FindFacesWithVertex(face[2], Faces))
    c.remove(Index)

    if list(a.intersection(b)) == []:
        NeighbourA = None
    else:  
        NeighbourA = list(a.intersection(b))[0]

    if list(b.intersection(c)) == []:
        NeighbourB = None
    else:
        NeighbourB = list(b.intersection(c))[0]

    if list(a.intersection(c)) == []:
        NeighbourC = None
    else:
        NeighbourC = list(a.intersection(c))[0]

    if NeighbourA == None or NeighbourB == None or NeighbourC == None:
        NoneFlag = True
        print(f"{Index}: None {[NeighbourA, NeighbourB, NeighbourC]}")

    return [NeighbourA, NeighbourB, NeighbourC]

#O(log(a2+b2+c2))
#https://www.geeksforgeeks.org/distance-between-a-point-and-a-plane-in-3-d/
def PointPlaneDistance(Point, Plane, true = False, roundvalue = 0.01):
    a, b, c, d = PlaneEquation(Plane)
    x, y, z = Point

    dist = (a * x + b * y + c * z + d)
    e = math.sqrt(a * a + b * b + c * c)
    Distance = abs(dist)/e

    if not true and Distance < roundvalue:
        Distance = 0

    return Distance

def Flatten2DList(List):
    FlattenedList = [j for sub in List for j in sub]
    return FlattenedList


#https://stackoverflow.com/a/31439438
def NestedDictValues(Dictionary, Numpy = True):
    #INITIAL VALUES
    # print(f"Dict: {Dictionary} - Type: {type(Dictionary)}")
    try:
        TestElement = Dictionary[0]
    except KeyError:
        DictItems = GetDictItems(Dictionary, False)
    else:
        DictItems = GetDictItems(Dictionary, True)
    # print(f"Dict Items: {DictItems}")
    # print(f"Dict Items[0]: {DictItems[0]}")

    #NESTED VALUES
    try:
        TestElement = DictItems[0]
    except KeyError:
        DictItems2 = GetDictItems(DictItems, False)
    except TypeError:
        DictItems2 = GetDictItems(DictItems, False)
    else:
        DictItems2 = GetDictItems(DictItems, True)
    # print(f"Dict Items 2 {DictItems2}")

    if Numpy:
        return np.array(list(DictItems2))
    else:
        return DictItems2
    

def GetDictItems(Dict, Merge = False):
    if Merge:
        MergedDict = {}
        for d in Dict:
            MergedDict.update(d)

        DictItems = MergedDict.values()
        return DictItems
    
    if not Merge:
        DictItems = list(Dict.values())
        return DictItems
    
    return -1

def PointToPointDistance(P1, P2):
    Distance = np.sqrt(np.sum((P1-P2)**2, axis=0))
    return Distance

#https://stackoverflow.com/a/41826126
def SetKey(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = value
    elif type(dictionary[key]) == list:
        dictionary[key].append(value)
    else:
        dictionary[key] = [dictionary[key], value]

#https://stackoverflow.com/a/61419477/22370980
#Computes the minimal distance between a line and a point given 3 points
def MinDistance(LineVertexA, LineVertexB, Point):
    #Convert to Numpy Arrays
    if type(LineVertexA) == list:
        LineVertexA = np.array(LineVertexA)
    if type(LineVertexB) == list:
        LineVertexB = np.array(LineVertexB)
    if type(Point) == list:
        Point = np.array(Point)

    A = LineVertexA - Point
    R = LineVertexB - A

    Min_T = np.clip(-A.dot(R) / (R.dot(R)), 0, 1)
    Distance = A + Min_T * R

    return np.sqrt(Distance.dot(Distance))

#https://stackoverflow.com/a/25516767/22370980
#Determines wether a point lays over a triangles face
def PointOverTriangle(Face, Point):

    VertexA = Face[0]
    VertexB = Face[1]
    VertexC = Face[2]

    ba = VertexB - VertexA
    cb = VertexC - VertexB
    ac = VertexA - VertexC
    n = VectorCrossProduct(ac, ba) #Eqivilent to n = ba x cas

    px = Point - VertexA
    nx = VectorCrossProduct(ba, px)
    if VectorDotProduct(nx, n) < 0:
        return False

    px = Point - VertexB
    nx = VectorCrossProduct(cb, px)
    if VectorDotProduct(nx, n) < 0:
        return False

    px = Point - VertexC
    nx = VectorCrossProduct(ac, px)
    if VectorDotProduct(nx, n) < 0:
        return False
    
    return True

#https://www.geeksforgeeks.org/program-to-find-equation-of-a-plane-passing-through-3-points/
# Function to find equation of plane.
def PlaneEquation(PlaneCoordinates): 
    x1, y1, z1 = PlaneCoordinates[0]
    x2, y2, z2 = PlaneCoordinates[1]
    x3, y3, z3 = PlaneCoordinates[2]

    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return a,b,c,d


#https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle
#https://math.stackexchange.com/questions/4322/check-whether-a-point-is-within-a-3d-triangle
#!!!!! Eh
def PointInTriangle(Face, Point):
    A = Face[0]
    B = Face[1]
    C = Face[2]

    AreaABC = np.linalg.norm(VectorCrossProduct((A-B), (A-C)))/2
    Alpha = np.linalg.norm(VectorCrossProduct((Point-B), (Point-C)))/2*AreaABC
    Beta = np.linalg.norm(VectorCrossProduct((Point-C), (Point-A)))/2*AreaABC
    Gamma = 1 - Alpha - Beta

    print(AreaABC, Alpha, Beta, Gamma)

    return A


#http://web.archive.org/web/20090609111431/http://www.crackthecode.us/barycentric/barycentric_coordinates.html
#https://users.csc.calpoly.edu/~zwood/teaching/csc471/2017F/barycentric.pdf <--
#https://vccimaging.org/Publications/Heidrich2005CBP/Heidrich2005CBP.pdf

# The cross product of two sides is a normal vector
# The norm of the cross product of two sides is twice the area
def CartesianToBarycentric(TriangleVerticies, Points, List = False, NpArray = True):
    VertexA = TriangleVerticies[0] 
    VertexB = TriangleVerticies[1] 
    VertexC = TriangleVerticies[2]

    #Triangle presented as VertexA, and two edge vectors UV
    U = VertexB - VertexA
    V = VertexC - VertexA

    try:
        if len(Points) == 0:
            return []
    except:
        traceback.print_exc()
        print(f"{Points}")

    if List:
        CoordinatesType = type(Points[0])
        if CoordinatesType == dict:
            Points = GetDictItems(Points, True)

        BarycentricArray = []
        for Point in Points:
            W = Point - VertexA
            N = np.cross(U,V)
            oneOver4ASquared = 1.0 / np.dot(N, N)

            B2 = np.dot(np.cross(U,W), N) * oneOver4ASquared
            B1 = np.dot(np.cross(W,V), N) * oneOver4ASquared
            B0 = 1.0 - B1 - B2

            BarycentricArray.append([B0, B1, B2])

        if NpArray:
            return np.array(BarycentricArray)
        else:
            return BarycentricArray

    else:
        W = Points - VertexA

        #Cross Product
        N = np.cross(U,V)

        #AreaConstant
        oneOver4ASquared = 1.0 / np.dot(N, N)

        B2 = np.dot(np.cross(U,W), N) * oneOver4ASquared
        B1 = np.dot(np.cross(W,V), N) * oneOver4ASquared
        B0 = 1.0 - B1 - B2
        # print(B0, B1, B2, B0+B1+B2)
        if NpArray:
            return np.array([B0, B1, B2])
        else:
            return [B0, B1, B2]
        

def BarycentricToCartesian(BarycentricWeights, TrianglePoints, List = False, NpArray = True):
    TriangleA = TrianglePoints[0]
    TriangleB = TrianglePoints[1]
    TriangleC = TrianglePoints[2]

    if List:
        CartesianArray = []
        for Weights in BarycentricWeights:
            X = (Weights[0] * TriangleA[0]) + (Weights[1] * TriangleB[0]) + (Weights[2] * TriangleC[0])
            Y = (Weights[0] * TriangleA[1]) + (Weights[1] * TriangleB[1]) + (Weights[2] * TriangleC[1])
            Z = (Weights[0] * TriangleA[2]) + (Weights[1] * TriangleB[2]) + (Weights[2] * TriangleC[2])

            Coordinate = [X,Y,Z]
            CartesianArray.append(Coordinate)

        if NpArray:
            return np.array(CartesianArray)
        else:
            return CartesianArray

    else:
        X = (BarycentricWeights[0] * TriangleA[0]) + (BarycentricWeights[1] * TriangleB[0]) + (BarycentricWeights[2] * TriangleC[0])
        Y = (BarycentricWeights[0] * TriangleA[1]) + (BarycentricWeights[1] * TriangleB[1]) + (BarycentricWeights[2] * TriangleC[1])
        Z = (BarycentricWeights[0] * TriangleA[2]) + (BarycentricWeights[1] * TriangleB[2]) + (BarycentricWeights[2] * TriangleC[2])

        Coordinate = [X,Y,Z]

        if NpArray:
            return np.array(Coordinate)
        else:
            return Coordinate

#https://www.geeksforgeeks.org/angle-between-two-planes-in-3d/
def PlaneToPlaneAngle(a1, b1, c1, a2, b2, c2):  
    d = ( a1 * a2 + b1 * b2 + c1 * c2 )
    e1 = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1)
    e2 = math.sqrt( a2 * a2 + b2 * b2 + c2 * c2)
    d = d / (e1 * e2)
    A = math.degrees(math.acos(d))
    print("Angle is"), A, ("degree")
 

#https://math.stackexchange.com/questions/1092912/find-closest-point-in-triangle-given-barycentric-coordinates-outside

def BarycentricClosestEdge(FaceCoordinates, FaceIndicies, BarycentricWeights):
    VertexA = FaceIndicies[0]
    VertexB = FaceIndicies[1]
    VertexC = FaceIndicies[2]
    VertexAValues = FaceCoordinates[0]
    VertexBValues = FaceCoordinates[1]
    VertexCValues = FaceCoordinates[2]
    Alpha = BarycentricWeights[0]
    Beta = BarycentricWeights[1]
    Gamma = BarycentricWeights[2]

    Edge = None
    EdgeValues = None
    Flag = None

    if Beta >= 0 and Gamma >= 0 and 1 <= Beta + Gamma:
        Edge = [VertexB, VertexC]
        EdgeValues = [VertexBValues, VertexCValues]
        return Edge, EdgeValues

    if Alpha >= 0 and Gamma >= 0 and 1 <= Alpha + Gamma:
        Edge = [VertexA, VertexC]
        EdgeValues = [VertexAValues, VertexCValues]
        return Edge, EdgeValues

    if Alpha >= 0 and Beta >= 0 and 1 <= Alpha + Beta:
        Edge = [VertexA, VertexB]
        EdgeValues = [VertexAValues, VertexBValues]
        return Edge, EdgeValues
    
    if not Flag:
        if Beta < 0 and Gamma < 0:
            if Beta < Gamma:
                Edge = [VertexB, VertexC]
                EdgeValues = [VertexBValues, VertexCValues]

            if Gamma < Beta:
                Edge = [VertexA, VertexB]
                EdgeValues = [VertexAValues, VertexBValues]

        if Alpha < 0 and Beta < 0:
            if Alpha < Beta:
                Edge = [VertexA, VertexC]
                EdgeValues = [VertexAValues, VertexCValues]

            if Beta < Alpha:
                Edge = [VertexB, VertexC]
                EdgeValues = [VertexBValues, VertexCValues]

        if Alpha < 0 and Gamma < 0:
            if Alpha < Gamma:
                Edge = [VertexB, VertexC]
                EdgeValues = [VertexBValues, VertexCValues]

            if Gamma < Alpha:
                Edge = [VertexA, VertexB]
                EdgeValues = [VertexAValues, VertexBValues]

    return Edge, EdgeValues


def PlotSample(Faces, Points, PyvistaFaces, IterationObject = None, SampledPoints = None, SampledPointsFaces = None, FaceSampleValues = None, FaceNeighbours = None, NeighbourSamples = None, NeighbourSampleValues = None, MappedNeighbourSampleValues = None, NearbySamples = None, NearbySampleDistances = None, ResultantSamples = None, MappedResultantSamples = None, ResultantFaces = None, ResultantBarycentric = None, ResultantEdge = None, PlottingSample = None, Var1 = None, Vars2 = None, Var3 = None):
    
    #Plot
    Mesh = pv.PolyData()
    Mesh.points = Points
    Mesh.faces = PyvistaFaces
    p = pv.Plotter()

    # print(f"IterationObjectList: {IterationObjectList}")

    #unpack iteration objectlist
    if IterationObject != None:
        if PlottingSample == None:
            LastObject = IterationObject

            MappedNewPoint, ResultantFaces = LastObject.getMappedPointAndFace()
            p.add_mesh(Mesh, show_edges = True)
            p.add_points(np.array(MappedNewPoint), color = 'green')
            p.show()

        else:
            LastObject = IterationObject
            (FaceSampleValues, NeighbourSamples, MappedNeighbourSampleValues, NearbySamples, NearbySampleDistances, BarycentricWeights) = LastObject.getSetupValues()
            (MappedNewPoint, NewPoint, NewFaces, ResultantReplusiveForce, ResultantBarycentricWeights, EdgeValues) = LastObject.getRepulsionValues()
            Face = NewFaces[PlottingSample]

            NearbySamps = NestedDictValues(NearbySamples[PlottingSample])
            FaceSampleVals = np.array(FaceSampleValues[PlottingSample])

            # print(f"Mapped Neighbour Samples (Normal) {MappedNeighbourSampleValues[PlottingSample]}\n   VS\nMapped Neighbour Samples (GetDictItems) {MappedNeighbours}")
            # print(f"   VS\nMapped Neighbour Samples (GetDictItems Merge) {MappedNeighboursMerge}")
            # print(f"   VS\nMapped Neighbour Samples (GetDictItems Merge x2) {GetDictItems(MappedNeighboursMerge)}")

            #Meshes
            p.add_mesh(Mesh.extract_cells(NewFaces[PlottingSample]), color = "red", show_edges = True)
            p.add_mesh(Mesh.extract_cells(FaceNeighbours[Face]), color = "blue", show_edges = True)

            #Relaxed points
            p.add_points(np.array(MappedNewPoint[PlottingSample]), color = 'blue', point_size = 10)
            p.add_points(FaceSampleVals, color = 'white')
            p.show()

    else:
        if PlottingSample != None:

            Face = SampledPointsFaces[PlottingSample]
            print(ResultantEdge[PlottingSample])

            # Parameters = [PlottingSample, SampledPoints[PlottingSample], SampledPointsFaces[PlottingSample], FaceSampleValues[PlottingSample], FaceNeighbours[PlottingSample], NeighbourSamples[PlottingSample], MappedNeighbourSampleValues[PlottingSample], NearbySamples[PlottingSample], NearbySampleDistances[PlottingSample], ResultantSamples[PlottingSample]]
            print(f"Sample Point - {PlottingSample} ({SampledPoints[PlottingSample]}")
            print(f"    Face - {SampledPointsFaces[PlottingSample]} ({Faces.VertexValues[PlottingSample]})")
            print(f"    FaceSampleValues - {FaceSampleValues[PlottingSample]}")
            print(f"    Neighbours - {FaceNeighbours[Face]}")
            print(f"    Neighbour Samples - {NeighbourSamples[PlottingSample]}")
            print(f"    Neighbour Sample Values - {NeighbourSampleValues[PlottingSample]}")
            print(f"    Mapped Neighbour Sample Values - {MappedNeighbourSampleValues[PlottingSample]}")
            print(f"    Nearby Samples - {NearbySamples[PlottingSample]}")
            print(f"    Nearby Sample Distances - {NearbySampleDistances[PlottingSample]}")
            print(f"    Resultant Sample - {ResultantSamples[PlottingSample]}")
            #getting list index out of range (probably using a sample index for a face-sized list)

            p.add_mesh(Mesh.extract_cells(SampledPointsFaces[PlottingSample]), color = "red", show_edges = True)
            p.add_mesh(Mesh.extract_cells(FaceNeighbours[Face]), color = "blue", show_edges = True)
            p.add_mesh(Mesh.extract_cells(ResultantFaces[PlottingSample]), color = "purple", show_edges = True)

            if ResultantEdge[PlottingSample] != None:
                p.add_points(np.array(ResultantEdge[PlottingSample]), color = 'pink')

            p.add_points(np.array(SampledPoints[PlottingSample]), color = 'black')
            p.add_points(np.array(GetDictItems(NearbySamples[PlottingSample])), color = 'yellow')

            #Relaxed points
            p.add_points(np.array(ResultantSamples[PlottingSample]), color = 'orange')
            p.add_points(np.array(MappedResultantSamples[PlottingSample]), color = 'green')
            p.show()

        else:
            p.add_mesh(Mesh, show_edges = True)
            p.add_points(np.array(Var1), color = 'orange', point_size = 10)
            p.add_points(np.array(Vars2), color = 'red', point_size = 10)
            p.add_points(np.array(Vars2), color = 'red', point_size = 10)
            p.show()

def SimplePlot(Faces, Points, PyvistaFaces, OldPoint, NewPoint, Face, TargetFace, Neighbours, Edge = None, Var = None):
    Mesh = pv.PolyData()
    Mesh.points = Points
    Mesh.faces = PyvistaFaces
    p = pv.Plotter()

    #Get Negighbouring Faces
    FaceIndicies = Faces.VertexIndiciesZeroBased
    NeighbouringFaces = GetNeighbouringFaces(Face, Faces)

    p.add_mesh(Mesh.extract_cells(Face), color = "red", show_edges = True)
    p.add_mesh(Mesh.extract_cells(NeighbouringFaces), color = "blue", show_edges = True)

    try:
        p.add_mesh(Mesh.extract_cells(TargetFace), color = "red", show_edges = True)
    except:
        print(f"SimplePlot: Couldn't locate TargetFace")

    #Relaxed points
    try:
        Labels = ['New Point', 'Old Point']
        RelaxedPoints = np.vstack((NewPoint, OldPoint))
        p.add_point_labels(RelaxedPoints, Labels, point_size = 10)
    except:
        print(f"SimplePlot: Adding Seperately")
        p.add_points(np.array(OldPoint), color = "white")
        p.add_points(np.array(NewPoint), color = "black", point_size = 20)
    
    try:
        p.add_points(np.array(Neighbours), color = 'white', point_size = 8)
    except:
        print(f"SimplePlot: No Neighbours")

    #Add Edge
    if Edge != None:
        try:
            p.add_points(np.array(Edge), color = 'white', point_size = 20)
        except:
            print(f"SimplePlot: Could not display Edge")

    try:
        p.add_points(np.array(Var), color = 'black', point_size = 8)
    except:
        print(f"SimplePlot: No Neighbours")

    p.show()

def TrianglePlotter(TriangleCoordinates, Point):

    Faces = [[0,1,2]]
    Verts =  [[TriangleCoordinates[i] for i in p] for p in Faces]  


    fig = plt.figure()
    ax = plt.axes(projection='3d',computed_zorder=False)
    ax.add_collection3d(Poly3DCollection(Verts))
    ax.scatter(Point[0], Point[1], Point[2], color='red', zorder=10)
    plt.show()

def SaveMesh(Mesh, Filename):
    pv.save_meshio(Filename, Mesh)


#https://www.geeksforgeeks.org/print-a-numpy-array-without-scientific-notation-in-python/
#defining the display funcrion 
def NumpyPrecision(arr,prec = 3):
    PreciseArray = np.array_str(arr, precision=prec, suppress_small=True)
    return PreciseArray


def PlotVoronoi(Faces, Points, PyvistaFaces, VoronoiRegions, SamplePoints = None, Matrix = None):
    
    #Create Mesh
    Mesh = pv.PolyData()
    Mesh.points = Points
    Mesh.faces = PyvistaFaces

    #Create Plotter
    p = pv.Plotter()
    # p.add_mesh(Mesh, show_edges = True)

    #Plot Points (by region)
    Red = 0
    Green = 0
    Blue = 0
    for Region in VoronoiRegions:
        Red += 1
        if Red == 255:
            Red = 0
            Green += 1
        if Green == 255:
            Green = 0
            Blue += 1
        if Blue == 255:
            Blue = 0

        if len(Region) != 0:
            p.add_points(np.array(Region), color = [Red/255, Green/255, Blue/255], point_size = 6)


    # if SamplePoints is not None:
    #     p.add_points(np.array(SamplePoints), color = "black")

    p.show()

def CloestPointOnSegment(PointA, PointB, Point):
    if PointA is None:
        return PointB
    elif PointB is None:
        return PointA

    #Convert to NumPy
    PointA = np.array(PointA)
    PointB = np.array(PointB)
    Point = np.array(Point)
    
    #Vectors
    AP = Point - PointA
    AB = PointB - PointA
    
    #Project AP onto AB
    t = np.dot(AP, AB) / np.dot(AB, AB)
    t = max(0, min(1, t))
    
    #Calculate closest point
    ClosestPoint = PointA + t * AB
    
    return ClosestPoint



#Pickling
def storeData(Filename, Data):
    File = open(Filename, 'ab')
    pickle.dump(Data, File)                    
    File.close()

def loadData(Filename):
    # for reading also binary mode is important
    File = open(Filename, 'rb')    
    Data = pickle.load(File)
    File.close()
    return Data
