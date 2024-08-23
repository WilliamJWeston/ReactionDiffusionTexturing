import numpy as np
import matplotlib.pyplot as plt
import math
from Methods import *
from collections import defaultdict
from scipy.spatial import Voronoi, Delaunay

def ConstructVoronoi(Points, ShowFigure = False):
    Vor = Voronoi(Points)
    Regions, Vertices = VoronoiFinitePolygons2D(Vor)

    #Adding Labels
    for i in range(len(Points)):
        plt.text(Points[i,0], Points[i,1], str(i))

    #MatPlotLib
    Fig, Ax = plt.subplots()

    # colorize
    for region in Regions:
        polygon = Vertices[region]
        plt.fill(*zip(*polygon), alpha = 0.4)

    plt.plot(Points[:,0], Points[:,1], 'ko')
    plt.xlim(Vor.min_bound[0] - 0.1, Vor.max_bound[0] + 0.1)
    plt.ylim(Vor.min_bound[1] - 0.1, Vor.max_bound[1] + 0.1)

    for i in range(len(Points)):
        plt.text(Points[i,0], Points[i,1], str(i))

    if ShowFigure:
        plt.show()
  
    plt.close()

    return Vor, Regions, Vertices

def VoronoiFinitePolygons2D(vor, radius = None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def DetermineVoronoiAdjacency(Points, PointsIndices, ShowFigure = False):
    #Try to construct voronoi regions, return an error if cannot (likely due to not enough points [4] to create)
    PointsAB = Points[:, [0, 1]]
    ShowFigure = False
    try:
        Vor, Regions, Vertices = ConstructVoronoi(PointsAB, ShowFigure = ShowFigure)
    except:
        Vor = None
        Regions = None
        Vertices = None

    Del = Delaunay(PointsAB)
    NeighbourInd, Neighbours = Del.vertex_neighbor_vertices
    PointsAB = Vor.points
    Vertices = Vor.vertices
    Regions = Vor.regions
    PointRegion = Vor.point_region

    #Get Sample Point Information
    SamplePoint = PointsAB[0]
    SamplePointNeighbours = Neighbours[NeighbourInd[0]:NeighbourInd[1]]
    SamplePointRegion = Regions[PointRegion[0]]
    SampleEdgeLengths = []
    SampleDistances = []
    SampleNeighbourIndices = []

    #Get Sample Point Neighbour Information
    for Neighbour in SamplePointNeighbours:
        NeighbourRegion = Regions[PointRegion[Neighbour]]
        RegionIntersection = list(set(SamplePointRegion).intersection(NeighbourRegion))
        NeighbourDist = math.dist(SamplePoint, PointsAB[Neighbour])
        SampleDistances.append(NeighbourDist)
        if len(RegionIntersection) > 1:
            EdgeLength = math.dist(Vertices[RegionIntersection[0]], Vertices[RegionIntersection[1]])
            SampleEdgeLengths.append(EdgeLength)

        NeighbourSampleIndex = PointsIndices[Neighbour-1]
        SampleNeighbourIndices.append(NeighbourSampleIndex)

    #Convert SamplePointRegion Indexes to values, and readd 3rd column of weights
    SamplePointRegionValues = [Vertices[i] for i in SamplePointRegion]
    
    tempWeights = []
    for RegionValues in SamplePointRegionValues:
        Weight = 1.0 - RegionValues[0] - RegionValues[1]
        tempWeights.append([Weight])
    tempWeights = np.array(tempWeights)
    SamplePointRegionValues = np.append(SamplePointRegionValues, tempWeights, axis=1)
    SampleNeighbourValues = [Points[i] for i in SamplePointNeighbours]

    return SamplePointRegionValues, SamplePointNeighbours, SampleNeighbourValues, SampleNeighbourIndices, SampleEdgeLengths, SampleDistances

def toUVCoordinates(Triangle, Coordinates, PlaneUV, NPArray = True, PreserveDict = False):    
    U, V = PlaneUV

    if isinstance(Coordinates, np.ndarray):
        U_Coord = VectorDotProduct(U, Coordinates)
        V_coord = VectorDotProduct(V, Coordinates)
        
        if NPArray:
            return np.array([U_Coord, V_coord])
        else:
            return [U_Coord, V_coord]
    
    if not Coordinates:
        return []
    
    # print(f"==========================\nCoordinates: {Coordinates} - Type: {type(Coordinates)}")
    # print(f"==========================\nCoordinates Indexed: {Coordinates[0]} - Type: {CoordinatesType}")
    # for x in Coordinates:
    #     print(f"\n\t{x} - Type: {type(x)}")
    # print(f"Type - {CoordinatesType}")
    CoordinatesType = type(Coordinates[0])
    if PreserveDict:
        UVCoordinateDict = {}
        
        if CoordinatesType == dict:
            DictValues = NestedDictValues(Coordinates)
            for SampleIndex, Sample in DictValues:
                U_Coord = VectorDotProduct(U, Sample)
                V_coord = VectorDotProduct(V, Sample)
                SetKey(UVCoordinateDict, SampleIndex, [U_Coord, V_coord])
        
        if CoordinatesType == list:

            MergedDict = {}
            for d in Coordinates:
                MergedDict.update(d)
            # print(f"MergedDict: {MergedDict} - Type: {type(MergedDict)}")

            for SampleIndex, SampleValue in MergedDict.items():
                U_Coord = VectorDotProduct(U, SampleValue)
                V_coord = VectorDotProduct(V, SampleValue)
                SetKey(UVCoordinateDict, SampleIndex, [U_Coord, V_coord])

        return UVCoordinateDict

    if not PreserveDict:
        CoordinatesArray = []

        if CoordinatesType == dict:
            DictValues = GetDictItems(Coordinates, True)
            # print(f"DICT VALUES: {DictValues}")
            for Sample in DictValues:
                U_Coord = VectorDotProduct(U, Sample)
                V_coord = VectorDotProduct(V, Sample)
                CoordinatesArray.append([U_Coord, V_coord])

        if CoordinatesType == list:
            for Sample in Coordinates:
                U_Coord = VectorDotProduct(U, Sample)
                V_coord = VectorDotProduct(V, Sample)
                CoordinatesArray.append([U_Coord, V_coord])

        if NPArray:
            return np.array(CoordinatesArray)
        else:
            return CoordinatesArray

def MergeUVValues(UVPoint, UVMappedNeighbours, UVFaceSamples):
    if type(UVMappedNeighbours) == list and type(UVFaceSamples) == list:
        UVValues = UVPoint
    elif type(UVFaceSamples) == list:
        UVValues = np.vstack((UVPoint, UVMappedNeighbours))  
    elif type(UVMappedNeighbours) == list:
        UVValues = np.vstack((UVPoint, UVFaceSamples))
    else:
        UVValues = np.vstack((UVPoint, UVMappedNeighbours, UVFaceSamples))  

    return UVValues

def MergeUVValues(UVPoint, NearbySamples):
    if len(NearbySamples) == 0:
        UVValues = UVPoint
    else:
        UVValues = np.vstack((UVPoint, NearbySamples))  

    return UVValues
