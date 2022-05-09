import matplotlib.image as img
import numpy as np
import matplotlib.pyplot as plt
import exterieur as ex
import interieur as inter
from shapely import geometry
from matplotlib import patches
import shapely
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely import affinity
import shapely.geometry as sg
import shapely.ops as so
import matplotlib.pyplot as plt
import utils as ut
import copy
import trimesh
import math
from descartes import PolygonPatch
from shapely.ops import cascaded_union
class PatchPolygon:

    def __init__(self, polygon, **kwargs):
        polygon_path = self.pathify(polygon)
        self._patch = PathPatch(polygon_path, **kwargs)

    @property
    def patch(self):
        return self._patch

    @staticmethod
    def pathify(polygon):
        ''' Convert coordinates to path vertices. Objects produced by Shapely's
            analytic methods have the proper coordinate order, no need to sort.

            The codes will be all "LINETO" commands, except for "MOVETO"s at the
            beginning of each subpath
        '''
        vertices = list(polygon.exterior.coords)
        codes = [Path.MOVETO if i == 0 else Path.LINETO
                 for i in range(len(polygon.exterior.coords))]

        for interior in polygon.interiors:
            vertices += list(interior.coords)
            codes += [Path.MOVETO if i == 0 else Path.LINETO
                      for i in range(len(interior.coords))]

        return Path(vertices, codes)
def poly_union(polys):
    # Check for self intersection while building up the cascaded union
    union = geometry.Polygon([])
    for polygon in polys:
        try:
            union = cascaded_union([polygon, union])
        except ValueError:
            pass
    return union
def projected_area(mesh, normal):
    normal = np.array(normal)
    normal = normal/np.linalg.norm(normal)
    m = copy.deepcopy(mesh)
    dir_rot = normal + np.array([1,0,0])
    if np.linalg.norm(dir_rot) < 1e-6:
        dir_rot = np.array([1,0,0])
    m.apply_transform(trimesh.transformations.rotation_matrix(math.pi, dir_rot))
    mr = copy.deepcopy(m)
    m.apply_transform(trimesh.transformations.projection_matrix((0,0,0),(1,0,0)))
    polygons = [
            shapely.geometry.Polygon(triangle[:,1:])
            for index_triangle, triangle in enumerate(m.triangles)
            if np.linalg.norm(m.face_normals[index_triangle] - np.array([1,0,0])) < 1e-6
        ]
    return mr,m,polygons

def chaine(no):
    image = img.imread('../mailleSelection/maille'+str(no)+'.jpg')
    image = image[:, :, 1]
    con = ex.contourExterieur(image)
    con.contour()
    con.orientation()
    # interi=inter.interieur(image,con.D,con.G)#
    print(con.G,con.D)
    if con.G[0]>con.D[0]:
        interi=inter.interieur(image,con.G,con.D)#
    else:
        interi=inter.interieur(image,con.D,con.G)#
    interi.contour()

    plt.plot(con.x, con.y, color="blue")
    print(con.M)
    plt.plot(interi.contourGauche[0],interi.contourGauche[1])
    plt.plot(interi.contourDroit[0],interi.contourDroit[1])
    polAc= Polygon([*zip(con.x,con.y)])
    pol= Polygon([*zip(interi.contourGauche[0],interi.contourGauche[1])])
    polEu= Polygon([*zip(interi.contourDroit[0],interi.contourDroit[1])])
    difference = polAc.difference(polEu).difference(pol)
    if interi.moAngle>0:
        rot=np.pi/2-interi.moAngle
    else:
        rot=-np.pi/2-interi.moAngle

    poly1=copy.deepcopy(difference)

    poly = affinity.rotate(difference, rot*180/np.pi, 'center')

    poly = affinity.scale(poly, xfact = 1/(max(con.x)-min(con.x)),yfact=1/(max(con.y)-min(con.y)),origin = (con.M[0],con.M[1]))

    xs, ys = poly.exterior.xy

    poly = affinity.translate(poly,xoff=-min(xs), yoff=-min(ys))
    #

    print('rotation',-rot*180/np.pi)


    # poly = affinity.scale(difference, xfact = 1/(max(con.x)),yfact=1/(max(con.y)), origin = (0, 0))
    # poly = affinity.scale(difference, xfact = 1/(max(con.x)),yfact=1/(max(con.y)), origin = (con.M[0],con.M[1]))

    # rot=np.pi/2-interi.moAngle
    # print("rotation",interi.moAngle)
    # xr,yr=ut.rotate_around_point_highperf([con.x,con.y],rot, con.M)
    # xra,yra=ut.rotate_around_point_highperf([interi.contourGauche[0],interi.contourGauche[1]],rot, con.M)
    # xrb,yrb=ut.rotate_around_point_highperf([interi.contourDroit[0],interi.contourDroit[1]],rot, con.M)
    # plt.plot(xr,yr)
    # plt.plot(xra,yra)
    # plt.plot(xrb,yrb)
    # plt.plot(xr,yr)
    # print(180/np.pi*interi.moAngle)

    return poly
if __name__ == "__main__":
    ##

    # dif1,dif2=chaine(19),chaine(18)
    # different = dif1.difference(dif2)
    # fig, axs = plt.subplots()
    # axs.set_aspect('equal', 'datalim')
    # print(different.area)
    # for geom in different.geoms:
    #     xs, ys = geom.exterior.xy
    #     axs.fill(xs, ys, alpha=0.5, fc='r', ec='none')
    # plt.show()

    ##
    dif=chaine(17)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    # ax1.axis("equal")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    mesh = trimesh.load('./mailleCAO.stl')
    # mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi/4,[0,1,0]))#------
    # mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi/4,[0,0,1]))#plan
    # mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi/5,[1,0,0]))#|
    m,r,mesh4=projected_area(mesh,[0,0,1])
    union1=poly_union(mesh4)
    xs, ys = union1.exterior.xy
    union1 = affinity.scale(union1, xfact = 1/(max(xs)-min(xs)),yfact=1/(min(ys)-max(ys)),origin = "center")
    xs, ys = union1.exterior.xy
    union1 = affinity.translate(union1,xoff=-min(xs), yoff=-min(ys))
    ax1.add_patch(PatchPolygon(union1, facecolor='none', edgecolor='green').patch)
    ax1.add_patch(PatchPolygon(dif, facecolor='none', edgecolor='red').patch)
    differencie = union1.difference(dif)
    differencie1 = dif.difference(union1)
    difTo= shapely.ops.unary_union([differencie,differencie1])
    patch1 = PolygonPatch(difTo.buffer(0))
    ax1.add_patch(patch1)
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # plt.show()
    # for geom in differencie.geoms:
    #     xs, ys = geom.exterior.xy
    #     print("a")
    #     ax1.fill(xs, ys, alpha=0.5, fc='r', ec='none')    # plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()
    # todo interne
    # todo rotation selon orientation interne