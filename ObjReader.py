from Methods import *
from scipy.spatial import Delaunay
import numpy as np
import sys

class VertexList(object):

    def __init__(self, VertexArrayList):
        self.Value = np.array(VertexArrayList)

    def __str__(self):
        return f"{self.Value}"
    
    def __repr__(self):
        return f"{self.Value}\n"
    
    #Return all columns
    def getColumns(self):
        return self.Value[:,0],self.Value[:,1], self.Value[:,2] 

class FaceList(object):

    def __init__(self, FaceArray):
        self.ListType = FaceArray[0].Type
        self.TotalArea = 0

        self.Values = [] #[3x2xN] Array
        self.VertexValues = [] #[3x3xN] Array (Vertex Coordinates)
        self.VertexIndicies = [] #[3xN] Array (Vertex Indicies)
        self.VertexIndiciesZeroBased = [] #[3xN] Array (Vertex Indicies)
        self.NormalValues = []
        self.NormalIndicies = []
        self.TextureValues = []
        self.TextureIndicies = []
        self.PyVista = [] #[4N] Array (Len Faces, F1, FN)*Len of Faces
        self.SurfaceAreas = [] # [1xN] Array of Areas

        if self.ListType == 'Vertex Normal Indicies without Texture Coordinate Indicies':
            #Loop through array
            for Line in FaceArray:
                
                Values = Line.Values[1:]
                NumElemets = Line.Values[0]
                FaceArea = Line.SurfaceArea

                VertexVals = Line.VertexValues
                VertexInds = [int(x[0]) for x in Line.Values[1:]]
                ZeroBasedVertexInds = [int(x[0])-1 for x in Line.Values[1:]]

                NormalVals = Line.NormalList
                NormalInds = [int(x[1]) for x in Line.Values[1:]]

                PyVistaValues = [NumElemets] + [int(x[0])-1 for x in Line.Values[1:]]
                self.Values.append(Values)
                self.VertexValues.append(VertexVals)
                self.VertexIndicies.append(VertexInds)
                self.VertexIndiciesZeroBased.append(ZeroBasedVertexInds)
                self.NormalValues.append(NormalVals)
                self.NormalIndicies.append(NormalInds)
                self.PyVista.append(PyVistaValues)
                self.TotalArea += FaceArea
                self.SurfaceAreas.append(FaceArea)

            self.PyVista = np.hstack(self.PyVista)

        if self.ListType == 'Vertex Normal Indicies and Texture Coordinates':
            #Loop through array
            for Line in FaceArray:
                
                Value = Line.Values[1:]
                NumElemets = Line.Values[0]
                FaceArea = Line.SurfaceArea

                VertexVals = Line.VertexValues
                VertexInds = [int(x[0]) for x in Line.Values[1:]]
                ZeroBasedVertexInds = [int(x[0])-1 for x in Line.Values[1:]]

                TextureVals = Line.TextureList
                TextureInds = [int(x[1]) for x in Line.Values[1:]]

                NormalVals = Line.NormalList
                NormalInds = [int(x[2]) for x in Line.Values[1:]]

                PyVistaValues = [NumElemets] + [int(x[0])-1 for x in Line.Values[1:]]

                self.Values.append(Value)
                self.VertexValues.append(VertexVals)
                self.VertexIndicies.append(VertexInds)
                self.VertexIndiciesZeroBased.append(ZeroBasedVertexInds)
                self.TextureValues.append(TextureVals)
                self.TextureIndicies.append(TextureInds)
                self.NormalValues.append(NormalVals)
                self.NormalIndicies.append(NormalInds)
                self.PyVista.append(PyVistaValues)
                self.TotalArea += FaceArea
                self.SurfaceAreas.append(FaceArea)

            self.PyVista = np.hstack(self.PyVista)

        #Maybe doesn't work
        if self.ListType == 'Vertex Indicies':
            for Line in FaceArray:
                Value = Line.Values
                VertexVals = Line.VertexValues
                VertexInds = [x[0] for x in Line.Values]

    def __str__(self):
        return f"{self.Values}\n"
    
    def __repr__(self):
        return f"{self.Values}\n"

class Vertex(object):

    def __init__(self, XCoordinate, YCoordinate, ZCoordinate):
        if ZCoordinate != None:
            self.X = float(XCoordinate)
            self.Y = float(YCoordinate)
            self.Z = float(ZCoordinate)
            self.Values = np.array((self.X, self.Y, self.Z))
        else:
            self.Texture1 = float(XCoordinate)
            self.Texture2 = float(YCoordinate)
            self.Values = np.array((self.Texture1, self.Texture2))

    def __str__(self):
        return [self.Values]
    
    def __repr__(self):
        return f"[{self.Values}]\n"
    
class Face(object):

    def __init__(self, FaceArray, Vertexes, Textures = None, Normals = None, Type = None):
        
        CountedVertexList = []
        VertexList = []

        CountedVertexIndexList = []
        VertexIndexList = []

        NormalList = []
        TextureList = []

        if FaceArray[0] != 'f':
            print(f"Face: {FaceArray} not recognised as a face.")

        else:
            #If vertex normal indicies without texture coordinates indicies
            if Type == 'Vertex Normal Indicies without Texture Coordinate Indicies':
                NumElements = len(FaceArray)-1
                CountedVertexIndexList.append(NumElements)
                CountedVertexList.append(NumElements)

                #Loop through each element and split by double slash
                for SubIndex, SubValue in enumerate(FaceArray[1:]):
                    VertexArrayIndex = int(FaceArray[SubIndex+1][0])-1
                    NormalArrayIndex = int(FaceArray[SubIndex+1][1])-1

                    #Get Corresponding Vertex and Normal Values
                    try:
                        FaceArray[0] = NumElements
                        FaceArray[1:][0] = Vertexes[VertexArrayIndex]
                        FaceArray[1:][1] = Normals[NormalArrayIndex]
                    except:
                        print(f"Cannot find vertex/normal: {SubValue}")

                    CountedVertexList.append(Vertexes[VertexArrayIndex])
                    CountedVertexIndexList.append(VertexArrayIndex)
                    VertexList.append(Vertexes[VertexArrayIndex])
                    VertexIndexList.append(VertexArrayIndex)

                    NormalList.append(NumElements)
                    NormalList.append(Normals[NormalArrayIndex])
                    
            else:
                #Split Face Components by /
                NumSlashes = FaceArray[1].count('/')

                #Vertex Indicies
                if Type == 'Vertex Indicies':
                    NumElements = len(FaceArray)-1
                    CountedVertexIndexList.append(NumElements)
                    CountedVertexList.append(NumElements)

                    for SubIndex, SubValue in enumerate(FaceArray[1:]):
                        VertexIndices = int(FaceArray[SubIndex+1][0])

                        try:
                            FaceArray[1:][0] = Vertexes[VertexIndices-1]
                            VertexList.append(Vertexes[VertexIndices-1])
                        except:
                            print(f"Cannot find vertex: {SubValue}")

                #Vertex Texture Coordinate Indicies
                if Type == 'Vertex Texture Coordinate Indicies':
                    NumElements = len(FaceArray)-1
                    CountedVertexIndexList.append(NumElements)
                    CountedVertexList.append(NumElements)

                    for SubIndex, SubValue in enumerate(FaceArray[1:]):

                        VertexArrayIndex = int(FaceArray[SubIndex+1][0])-1
                        TextureArrayIndex = int(FaceArray[SubIndex+1][1])-1

                        try:
                            FaceArray[1:][0] = Vertexes[VertexArrayIndex]
                            FaceArray[1:][1] = Textures[TextureArrayIndex]

                            VertexList.append(Vertexes[VertexArrayIndex])
                            NormalList.append(Normals[TextureArrayIndex])
                        except:
                            print(f"Cannot find vertex/texture: {SubValue}")

                #Vertex Normal Indicies and Texture Coordinates
                if Type == 'Vertex Normal Indicies and Texture Coordinates':
                    NumElements = len(FaceArray)-1
                    CountedVertexIndexList.append(NumElements)
                    CountedVertexList.append(NumElements)

                    for SubIndex, SubValue in enumerate(FaceArray[1:]):

                        VertexArrayIndex = int(FaceArray[SubIndex+1][0])-1
                        TextureArrayIndex = int(FaceArray[SubIndex+1][1])-1
                        NormalArrayIndex = int(FaceArray[SubIndex+1][2])-1

                        try:
                            FaceArray[0] = NumElements
                            FaceArray[1:][0] = Vertexes[VertexArrayIndex]
                            FaceArray[1:][1] = Textures[TextureArrayIndex]
                            FaceArray[1:][2] = Normals[NormalArrayIndex]
                            
                        except:
                            print(f"Cannot find vertex/texture/normal: {SubValue}")

                        VertexList.append(Vertexes[VertexArrayIndex])
                        TextureList.append(Textures[TextureArrayIndex])
                        NormalList.append(Normals[NormalArrayIndex])

                        CountedVertexList.append(Vertexes[VertexArrayIndex])
                        CountedVertexIndexList.append(VertexArrayIndex)

            self.Values = FaceArray

            self.CountedVertexValues = CountedVertexList
            self.CountedVertexIndexes = CountedVertexIndexList
            self.VertexValues = VertexList
            self.VertexIndexes = VertexIndexList
            
            self.TextureList = TextureList
            self.NormalList = NormalList

            self.Length = len(FaceArray)
            self.SurfaceArea = self.GetArea()
            self.Type = Type

    def GetArea(self):
        VertexValues = self.CountedVertexValues[1:]

        if len(VertexValues) < 3:
            return -1

        Total = [0, 0, 0]
        for i in range(len(VertexValues)):
            vi1 = VertexValues[i]       

            if i is len(VertexValues)-1:
                vi2 = VertexValues[0]
            else:
                vi2 = VertexValues[i+1]

            prod = VectorCrossProduct(vi1, vi2)
            Total[0] += prod[0]
            Total[1] += prod[1]
            Total[2] += prod[2]

        Result = VectorDotProduct(Total, UnitNormal(VertexValues[0], VertexValues[1], VertexValues[2]))
        return abs(Result/2)

    def __str__(self):
        return f"{self.Values}\n"
    
    def __repr__(self):
        return f"{self.Values}\n"

class ParseOBJ(object):

    def __init__(self, Filename = None):
        OBJFile = open(Filename, 'r')
        Vertexes = []
        VertexNormals = []
        VertexTextures = []
        Faces = []

        for Line in OBJFile:
            SplitLine = Line.split()
            if len(SplitLine) == 0:
                continue

            #Check if Vertex
            if SplitLine[0] == 'v':
                Vertexes.append(Vertex(SplitLine[1], SplitLine[2], SplitLine[3]).Values)

            #Check if Vertex Normals
            elif SplitLine[0] == 'vn':
                VertexNormals.append(Vertex(SplitLine[1], SplitLine[2], SplitLine[3]).Values)
            #Check if Texture Coordinates
            elif SplitLine[0] == 'vt':
                VertexTextures.append(Vertex(SplitLine[1], SplitLine[2], None).Values)

            #Check if Faces
            elif SplitLine[0] == 'f':
                FaceLine = SplitLine
                Type = GetMeshType(FaceLine)                
                SplitFace = SplitFaceLine(FaceLine, Type)
            
                #Triangulate face if it is not a triangle
                Triangles = Triangulate(SplitFace, Type, Vertexes)

                #Add each new triangle as a new Face
                if type(Triangles) == tuple:
                    for Triangle in Triangles:
                        Faces.append(Face(Triangle, Vertexes, VertexTextures, VertexNormals, Type))
                else:
                    Faces.append(Face(Triangles, Vertexes, VertexTextures, VertexNormals, Type))

        self.Vertexes = VertexList(Vertexes)
        self.Faces = FaceList(Faces)
            
def Triangulate(Face, Type, VertexList):
    NumElements = len(Face)-1
    Values = (Face[1:])
    
    #If element is a triangle do nothing
    if NumElements == 3:
        return Face
    
    elif NumElements == 4:
        TriangleOne = ["f", Values[0], Values[1], Values[2]]
        TriangleTwo = ["f", Values[2], Values[3], Values[0]]
        return TriangleOne, TriangleTwo
    # Vertexes = [i[0] for i in Values]
    # # VertexValues = np.array([VertexList[int(x)] for x in Vertexes])

    # # print(VertexList)
    # # print(f"Values: {Values}")
    # # print(f"VertexValues: {VertexValues}")
    # # print(Type)

    # #Vertex Normal without Texture
    # if Type == 'Vertex Normal Indicies without Texture Coordinate Indicies':
    #     Normals = [i[1] for i in Values]

    # #Vertex Indicies
    # if Type == 'Vertex Indicies':
    #     pass

    # #Vertex Texture Coordinate Indicies
    # if Type == 'Vertex Texture Coordinate Indicies':
    #     Vertexes = [i[0] for i in Values]
    #     Textures = [i[1] for i in Values]

    # #Vertex Normal Indicies and Texture Coordinates
    # if Type == 'Vertex Normal Indicies and Texture Coordinates':
    #     Textures = [i[1] for i in Values]
    #     Normals = [i[0] for i in Values]

    #If element is a triangle do nothing
    



#Get the 'Type' of Mesh from OBJ
def GetMeshType(Face):
    if '//' in Face[1]:
        Type = 'Vertex Normal Indicies without Texture Coordinate Indicies'

    else:
        NumSlashes = Face[1].count('/')
        match NumSlashes:
            #Vertex Indicies
            case 0:
                Type = 'Vertex Indicies'
            #Vertex Texture Coordinate Indicies
            case 1:
                Type = 'Vertex Texture Coordinate Indicies'
            #Vertex Normal Indicies and Texture Coordinates
            case 2:
                Type = 'Vertex Normal Indicies and Texture Coordinates'
    return Type

def SplitFaceLine(Face, Type):
    if Type == 'Vertex Normal Indicies without Texture Coordinate Indicies':
        for SubIndex, SubValue in enumerate(Face):
            Face[SubIndex] = SubValue.split('//')
        Face[0] = 'f'
    
    else:
        for SubIndex, SubValue in enumerate(Face):
            Face[SubIndex] = SubValue.split('/')
            Face[0] = 'f'
    
    return Face