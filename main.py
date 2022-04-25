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

def chaine(no):
    image = img.imread('../mailleSelection/maille'+str(no)+'.jpg')
    image = image[:, :, 1]
    con = ex.contourExterieur(image)
    con.contour()
    con.orientation()
    interi=inter.interieur(image,con.D,con.G)#
    interi.contour()
    
    plt.plot(con.x, con.y, color="blue")
    
    plt.plot(interi.contourGauche[0],interi.contourGauche[1])
    plt.plot(interi.contourDroit[0],interi.contourDroit[1])
    polAc= Polygon([*zip(con.x,con.y)])
    pol= Polygon([*zip(interi.contourGauche[0],interi.contourGauche[1])])
    polEu= Polygon([*zip(interi.contourDroit[0],interi.contourDroit[1])])
    difference = polAc.difference(polEu).difference(pol)
    
    # xr,yr=ut.rotate_around_point_highperf([con.x,con.y], np.pi/2-interi.moAngle, con.M)
    
    # plt.plot(xr,yr)
    print(180/np.pi*interi.moAngle)
    
    return difference

if __name__ == "__main__":
    chaine(19)
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

    # fig, ax1 = plt.subplots(figsize=(6, 4))
    # ax1.set_xlim(10, 1300)
    # ax1.set_ylim(10, 800)
    # ax1.add_patch(PatchPolygon(different, facecolor='blue', edgecolor='red').patch)
    # plt.imshow(image, cmap=plt.get_cmap('gray'))
    # plt.show()
    # todo interne
    # todo rotation selon orientation interne