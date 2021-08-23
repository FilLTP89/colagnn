# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
    Compute adjacency matrix for SEM3D monitors
"""
import argparse
import numpy
import pandas
import geopandas
import shapely.ops
import shapely.geometry
from shapely.ops import triangulate
from shapely.geometry import MultiPoint, Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d
from pysal.lib import weights
import pyvista


def ComputeSEM3DAdjacencyMatrix(sf="./stations.txt",af="./adjacency.txt",**kwargs):
    coords = numpy.loadtxt(sf,delimiter=',').T
    xl,yl,zl = coords
    xl,yl,zl = map(numpy.unique,(xl,yl,zl))
    dx,dy,dz = map(numpy.diff,(xl,yl,zl))
    dx,dy,dz = map(numpy.average,(dx,dy,dz))
    xl -= dx
    yl -= dy
    zl -= dz
    xl = numpy.append(xl,[xl[-1]+dx])
    yl = numpy.append(yl,[yl[-1]+dy])
    zl = numpy.append(zl,[zl[-1]+dz])
    xm,ym,zm = numpy.meshgrid(xl,yl,zl)
    mesh = pyvista.StructuredGrid(xm,ym,zm)

    # points = MultiPoint(list(map(tuple,zip(xl,yl,zl))))
    
    polygs = [Polygon(list([tuple(p) for p in mesh.cell_points(c)])) for c in range(mesh.n_cells)]

    # triang = triangulate(points)
    # Convert to GeoSeries
    polygs = geopandas.GeoSeries(polygs)
    gdf = geopandas.GeoDataFrame({'geometry':polygs,
        'id':['P-%s'%str(i).zfill(2) for i in range(len(polygs))]})

    wr = weights.contiguity.Rook.from_dataframe(gdf)
    wq = weights.contiguity.Queen.from_dataframe(gdf)
    ad = pandas.DataFrame(*wr.full()).astype(int)
    ad.to_csv(path_or_buf=af,sep=',',header=False,index=False)
    return


def ComputeAmitAdjacencyMatrix(sf="./amit-raw-adj.txt",af="./amit-adj.txt",**kwargs):
    # Load adjacency
    ad = pandas.read_csv(sf,sep="\t",header=1,skiprows=0,index_col=1).iloc[:,1:].astype(numpy.float32)
    ad.to_csv(path_or_buf=af,sep=',',header=False,index=False)
    return

def ParseOptions():
    OptionParser = argparse.ArgumentParser(prefix_chars='@')
    OptionParser.add_argument('@t','@@tag',type=str,default='amit',help='Station file')
    OptionParser.add_argument('@s','@@sf',type=str,default='../data/sem3dtraces/stations.txt',help='Station file')
    OptionParser.add_argument('@a','@@af',type=str,default='../data/sem3d-adj.txt',help='Adjacency matrix file')
    options = OptionParser.parse_args().__dict__
    return options

if __name__=='__main__':
    
    options = ParseOptions()
    if "amit" in options['tag']:
        ComputeAmitAdjacencyMatrix(**options)
    elif "sem3d" in options['tag']:
        ComputeSEM3DAdjacencyMatrix(**options)
