import numpy as np
import utils as ut
import matplotlib.pyplot as plt
import matplotlib.image as img
def rotate_around_point_highperf(xy, radians, origin=(0, 0)):
    """Rotate a point around a given point.

    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It's less readable than the previous
    function but it's faster.
    """
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return qx, qy

def rotate(src,angle,pivot,imshape):
    rotation_mat = np.transpose(np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]))
    h,w=imshape
    pivotX,pivotY = pivot[0],pivot[1]
    sortie = np.zeros(src.shape,dtype='u1')
    for height in range(h):
        for width in range(w):
            xy_mat=np.array([[width-pivotX],[height-pivotY]])
            rotate_mat = np.dot(rotation_mat,xy_mat)
            new_x = pivotX + int(rotate_mat[0])
            new_y = pivotY + int(rotate_mat[1])
            if(0<=new_x<=w-1) and (0<=new_y<=h-1):
                sortie[new_y,new_x]=src[height,width]
    return sortie

class contourExterieur(object):
    def __init__(self,image):
        """
        Donnée utile pour la création du contour exterieur
        """
        self.image = image #image à detecter
        self.seuil = 8 #seuil de dérivation qui détermine un point du contour
        self.x = None # coordonnées des points du contour
        self.y = None
        self.M = None
        self.G = None
        self.D = None

    def derive1D(self,amplitudeArray):
        """
        Détermine si le gradient sur une ligne
        dépasse une certaine valeur
        -> a liste(uint8) : ligne d'image
        <- liste(bool) : gradient diverge ?
        """
        # décalage pour réaliser le gradient en parallel
        vectorAmplDecale = np.roll(amplitudeArray, -1)
        return amplitudeArray+128-vectorAmplDecale > 128+self.seuil # gradient

    def deriveIndiceArray(self,amplitudeArray):
        """
        Determine les indices de divergence
        dans les deux sens de la liste
        -> a (int) : lignes d'image
        <- x,y : indice où le grad diverge
        """
        indiceDeriv = ut.indiceDerivChaine(
            amplitudeArray, self.derive1D)  # indice du premier divergent +
        amplitudeArrayDecale = np.flip(amplitudeArray)
        indiceDerivPartie2 = ut.indiceDerivChaine(
            amplitudeArrayDecale, self.derive1D)  # indice du premier divergent -
        indiceDerivPartie2 = None if indiceDerivPartie2 == None else (
            amplitudeArray.size-indiceDerivPartie2 if indiceDerivPartie2 != 0 else 0)  # a.size
        return indiceDeriv, indiceDerivPartie2

    def contour(self):
        """
        1.réccupération des points x,y où le gradient
        selon la direction dépasse une certaine valeur
        2.on enlève les points à partir d'une grande déviation
        -> image (int(int)) : liste en niveaux de gris
        <- (int) : points du contour exterieur
        """
        # gradient selon x haut et bas -> #[0, None , 2 ,...]
        B = np.apply_along_axis(self.deriveIndiceArray, 0, self.image)

        # indices y gradient diverge -> [x,(y)] point exacte
        C1 = np.where(B != None)
        D1 = B[B != None]  # indice x gradient diverge -> de meme

        # même pour gradient y droite gauche
        B = np.apply_along_axis(self.deriveIndiceArray, 1, self.image)
        C = np.where(B != None)
        D = B[B != None]

        # 2. reccuperation de points en suivant le contour obtenu
        L = []
        mid = len(C1[0])//2
        for i, j in zip([D1[0:mid], D1[mid:]], [C1[1][0:mid], C1[1][mid:]]):
            y, x = ut.lissage(i, j)
            ordre = np.argsort(x)
            x, y = x[ordre], y[ordre]
            L.append([x, y])
        for i, j in zip([C[0][::2], C[0][1::2]], [D[::2], D[1::2]]):
            x, y = ut.lissage(j, i)
            L.append([x, y])

        a=np.argmax(L[0][0])
        b=np.argmax(L[1][0])
        c=np.argmax(L[2][0])
        d=np.argmax(L[3][0])
        T=[]
        D=[]
        if L[3][0][d]>L[2][0][c]:
            T=L[3][0]
            L[3][0]=L[2][0]
            L[2][0]=T
            D=L[3][1]
            L[3][1]=L[2][1]
            L[2][1]=D

        # plt.plot(L[0][0],L[0][1],'yellow')
        # plt.plot(L[1][0],L[1][1],'blue')
        # plt.plot(L[2][0],L[2][1],'magenta')
        # plt.plot(L[3][0],L[3][1],'green')


        # plt.plot(L[0][0][a],L[0][1][a],'yo')
        # plt.plot(L[1][0][b],L[1][1][b],'bo')
        # plt.plot(L[2][0][c],L[2][1][c],'o')
        # plt.plot(L[3][0][d],L[3][1][d],'go')

        # 3. suppression des redondances en recolant les contours
        droite = lambda a,b,c,mn,mx : (np.concatenate((a[0][mn:],b[0],c[0][mx:])),np.concatenate((a[1][mn:],b[1],c[1][mx:])))
        gauche = lambda a,b,c,mn,mx : (np.concatenate((a[0][:mn],b[0],c[0][:mx])),np.concatenate((a[1][:mn],b[1],c[1][:mx])))
        l=list(map(ut.concatenationRange,[L[3]]*2,[L[0],L[1]],[L[2]]*2,[gauche,droite]))
        # l=list(map(ut.concatenationRange,[L[3]],[L[0]],[L[2]],[gauche]))
        # plt.plot(l[0][0],l[0][1])
        self.x,self.y=np.concatenate((l[0][0],np.flip(l[1][0]))),np.concatenate((l[0][1],np.flip(l[1][1])))

    def orientation(self):
        """
        obtient les points des 4 etremites de la forme

        """
        # UTILE
        X,Y=self.x,self.y
        xe1,xe2=ut.extremites(X,Y,0,0) #points extremes

        xg,yg=X[xe1],Y[xe1] #point gauche
        xd,yd=X[xe2],Y[xe2] #point droit

        a=(yg-yd)/(xg-xd) #coefficient directeur droite milieux
        b=yg-a*xg #ordonnée àl'origine

        # UTILE
        xm,ym=min(xg,xd)+abs(xd-xg)/2,min(yg,yd)+abs(yd-yg)/2 #point milieux du grand côte

        # xmg,ymg=int(max(xg,xm)-abs(xm-xg)*2/3),int(min(yg,ym)+abs(ym-yg)*2/3) #point milieux gauche
        # xmd,ymd=int(max(xd,xm)-abs(xm-xd)/3),int(min(yd,ym)+abs(ym-yd)/3) #point milieux droit


        print(xg,xm,xd)

        if xg<xd:
            xmg,ymg=int(max(xg,xm)-abs(xm-xg)/3),int(min(yg,ym)+abs(ym-yg)/3) #point milieux gauche
            xmd,ymd=int(max(xd,xm)-abs(xm-xd)*2/3),int(min(yd,ym)+abs(ym-yd)*2/3) #point milieux droit
        else:
            xmg,ymg=int(max(xg,xm)-abs(xm-xg)*2/3),int(min(yg,ym)+abs(ym-yg)*2/3) #point milieux gauche
            xmd,ymd=int(max(xd,xm)-abs(xm-xd)/3),int(min(yd,ym)+abs(ym-yd)/3) #point milieux droit
        #rotation
        theta = np.arctan(abs(ym-yg)/abs(xm-xg))#np.pi/180*
        xx,yy=rotate_around_point_highperf([xg,yg],-theta,[xm,ym])
        xx1,yy1=rotate_around_point_highperf([xd,yd],-theta,[xm,ym])


        #TRACE
        x=np.linspace(min(X),max(X),41,dtype=np.int) #trace de droite

        # UTILE
        y=a*x+b #equation de droite milieux

        #UTILE perp passant par la droite
        ap=-1/a #coef dir
        bp=ym-ap*xm #ord ori

        #UTILE
        idxs=np.where(X>xm)#proche perp theori

        #UTILE
        ih=ut.chercheProche(idxs[0][0],ap,bp,X,Y)
        ib=ut.chercheProche(idxs[0][-1],ap,bp,X,Y)

        # TRACE
        xPerp=np.linspace(X[ih],X[ib],41,dtype=np.int) #vecteur trace
        y1=ap*xPerp+bp #equation droite perpendiculaire

        # TRACE
        xpf,ypf=X[idxs[0][0]],Y[idxs[0][0]]     # perp point haut
        d=abs(ap*xpf-ypf+bp)/np.sqrt(ap**2+1)   #distance a la perp
        app=-1/ap #coef dir
        bpp=ypf-app*xpf #ord ori
        y2=app*xPerp+bpp #eqn droite perp

        # UTILE
        xR,yR=[],[]
        for i,j in zip(X,Y):
            a,b=rotate_around_point_highperf([i,j],-theta,[xm,ym])
            xR.append(a)
            yR.append(b)

        # UTILE
        self.M=[xm,ym]
        self.G=[xmg,ymg]
        self.D=[xmd,ymd]
        self.theta=theta

        plt.plot(xm,ym,'ro')
        plt.plot(xmd,ymd,'ro')
        plt.plot(xmg,ymg,'ro')

    def __repr__(self):
        plt.plot(L[0][0], L[0][1])  # bs
        plt.plot(L[1][0], L[1][1])#haut
        plt.plot(L[2][0], L[2][1])#droite
        plt.plot(L[3][0], L[3][1])#gauche

print("module \"exterieur charge \"")
if __name__ == "__main__":
    no = 17
    image = img.imread('../mailleSelection/maille'+str(no)+'.jpg')
    image = image[:, :, 1]
    con = contourExterieur(image)
    con.contour()
    con.orientation()
    # soso = rotate(image,-theta,[int(xm),int(ym)],image.shape)
    plt.plot(con.x,con.y,color="blue")
    plt.imshow(image,cmap = plt.get_cmap('gray'))
    plt.axis("equal")
    plt.show()
#     todo interne
#     todo rotation selon orientation interne
