import numpy as np
import matplotlib.pyplot as plt
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
def polyfit_with_fixed_points(n, x, y, xf, yf):
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x**np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[:n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[:n + 1, :n + 1] = np.take(x_n, idx)
    xf_n = xf**np.arange(n + 1)[:, None]
    mat[:n + 1, n + 1:] = xf_n / 2
    mat[n + 1:, :n + 1] = xf_n.T
    mat[n + 1:, n + 1:] = 0
    vec[:n + 1] = yx_n
    vec[n + 1:] = yf
    params = np.linalg.solve(mat, vec)
    return params[:n + 1]

def merge(listea, listeb, x0=0, y0=0):
    pas = 20
    dec = len(listea)//pas
    if x0 != 0:
        Xn, Yn = np.array([x0], dtype='int'), np.array([y0], dtype='int')
    else:
        Xn, Yn = np.array([listea[0]], dtype='int'), np.array(
            [listeb[0]], dtype='int')
    for i in range(dec):
        subx = listea[i*pas:i*pas+pas-1]
        suby = listeb[i*pas:i*pas+pas-1]
        if i == 0 and x0 == 0:
            params = polyfit_with_fixed_points(1, subx, suby, [], [])
        else:
            params = polyfit_with_fixed_points(1, subx, suby, Xn[-1:], Yn[-1:])
        poly = np.polynomial.Polynomial(params)
        xx = np.linspace(min(subx), max(subx)-1,
                         int(max(subx)-min(subx)), dtype='int')
        Xn = np.append(Xn, xx)
        Yn = np.append(Yn, poly(xx))
    subx = listea[dec*pas:]
    suby = listeb[dec*pas:]

    if len(subx) > 10:
        params = polyfit_with_fixed_points(1, subx, suby, Xn[-1:], Yn[-1:])
        poly = np.polynomial.Polynomial(params)
        xx = np.linspace(int(min(subx)), int(max(subx))-1,int(max(subx)-min(subx)), dtype='int')
        Xn = np.append(Xn, xx)
        Yn = np.append(Yn, poly(xx))
    return Xn, Yn

def concatenationRange(a,b,c,func):
    """
    concatenation
    et fit
    """
    mn=np.argmin(a[0])
    mx=np.argmax(c[0])
    tempa,tempb = func(a,b,c,mn,mx)
    ordre= np.argsort(tempa)
    x,y=tempa[ordre],tempb[ordre]
    ys=np.array([])
    for i in range(abs(x[0]-x[-1]-1)):
        yor=y[np.where(x==(x[0]+i))[0]]
        sorted(yor,reverse=True)
        ys=np.concatenate((ys,yor))
    return merge(x,ys)

def premierDerivIndex(derivSeuilArray):
    """
    determine l'indice du premier divergent
    partir duquel le gradient diverge
    -> C liste(bool): premier diverge
    <- (int) : indice
    """
    vectorDeriv = np.where(
        derivSeuilArray == True)  # indices où le gradient diverge
    indiceDeriv = None
    if vectorDeriv[0].size != 0:
        indiceDeriv = vectorDeriv[0][0]  # premier indice où ca diverge
    return indiceDeriv

def indiceDerivChaine(amplitudeArray, fonction):
    """
    GLOBAL : applique une fonction et applique une autre fonction
    APPLICATION : Determine si le gradient diverge et l'indice du premier divergent
    et l'indice où ca se passe
    ->a liste(uint8) : lignes d'images
    ->func : fonction à appliquer
    <- (int) : indice
    """
    derivSeuilArray = fonction(amplitudeArray)  # gradient
    return premierDerivIndex(derivSeuilArray)  # indice

def lissage(x, y):
    """
    tronque une liste à partir d'un point divergent
    -> a : indice variable
    -> b : indice linéaire
    <- liste des indices jusqu'à divergence
    """
    def derive1D(amplitudeArray):
        """
        determine si le gradient depasse un seuil
        """
        dev = 5
        vectorAmplDecale = np.roll(amplitudeArray, -1)  # décalage pour
        return np.abs(amplitudeArray-vectorAmplDecale) > dev  # gradient

    def deriveIndiceArray(liste):
        """
        pour une ligne
        ->liste
        <-indice du premier divergent
        """
        y = indiceDerivChaine(liste, derive1D)
        return y

    def reccuperation(lis, Y2):
        """
        -> liste(int) : abscice
        -> Y1 (int) : indice supposé divergence
        <- Y1 (int) : indice rectifié
        """
        # difference entre l'abscisse du point scruté et un point eloigné pouvant être accepté
        madev = 16
        excur = 7  # point suivant scute au maximum
        Ysup = np.apply_along_axis(deriveIndiceArray, 0, lis[Y2:])
        if Ysup != None:
            Y1 = Ysup+Y2
        else:
            return lis[:Y2], Y2
        if Y1 < len(lis)-excur:
            for i in range(2, excur):
                if np.abs(lis[Y1+i]-lis[Y1]) < madev:
                    for j in range(1, i):
                        lis[Y1+j] = int(np.abs(lis[Y1+i]-lis[Y1]) /
                                        2+min(lis[Y1], lis[Y1+i]))
                    return reccuperation(lis, Y1+i-1)
        return lis[:Y1], Y1
    mid = len(x)//2
    gauche = x[mid-1::-1]
    droite = x[mid:]
    droite, Y2 = reccuperation(droite, 0)
    gauche, Y1 = reccuperation(gauche, 0)
    return np.concatenate((gauche[Y1-1::-1], droite[:Y2])), y[mid-Y1:Y2+mid]


def chercheProche(Init,a,b,X,Y):
    """
    utilisé pour obtenir les points extremes perpendiculaires,
    idéalement pour obtenir l'orientation de la pièce
    """
    distance = lambda a,b,x,y : abs(a*x-y+b)/np.sqrt(a**2+1)
    best=100 #meilleur distance
    prev= distance(a,b,X[Init],Y[Init]) #precedente distance
    iBest=Init #indice du meilleur
    prevBest=best+1
    pas=1
    idx=Init+pas #indice observe
    while True:
        act = distance(a,b,X[idx],Y[idx])
        prevBest=best
        if act < best:
            best = act
            iBest = idx
        if act>prev:
            pas=-pas
        idx+=pas
        if prevBest==best:
            break
    return iBest

def extremites(X,Y,o,oo):
    """
    Detection des points les plus éloignés
    """
    plt.axis("equal")
    norm = lambda p1,p2 : np.sqrt(abs(p1[1]-p2[1])**2+abs(p1[0]-p2[0])**2)
    a=np.where(X==o)
    if len(a[0])==0:
        ind=100
    else:
        ind=a[0][0]
    b=np.where(X==oo)
    if len(b[0])==0:
        indg=200
    else:
        indg=b[0][0]
    pas = 30
    ecartSucc=1
    pd=[X[indg],Y[indg]]
    for i in range(10):
        ecartPrec= norm(pd,[X[ind],Y[ind]])
        ind+=pas
        ecart = norm(pd,[X[ind],Y[ind]])
        if ecart<ecartPrec:
            pas=-pas
            ecartPrec=0
        while ecart>ecartPrec:
            ind+=pas
            if ind>len(X)-1:
                ind=0
            ecartPrec=ecart
            ecart = norm(pd,[X[ind],Y[ind]])
        diff=abs(ecartPrec-ecartSucc)
        if diff<1:
            break
        ecartSucc=ecartPrec
        ind2 = indg
        indg=ind-pas
        ind=ind2
        if indg>len(X):
            indg-=len(X)
        if ind>len(X):
            ind-=len(X)
        pd=[X[indg],Y[indg]]
        if i>0:
            pas=1
    return ind-pas,indg

if __name__ == "__main__":
    print(12)