import numpy as np
import matplotlib.pyplot as plt
import utils as ut
class interieur(object):
    def __init__(self,image,centreGauche,centreDroit):
        self.image = image
        self.centreGauche=centreGauche
        self.centreDroit=centreDroit
        self.contourDroit=[]
        self.contourGauche=[]

    def detectEdge(self,liste,tol):
        """
        detecte du derive
        """
        vectorAmplDecale = np.roll(liste,-1)
        A=liste+128-vectorAmplDecale < 128+tol
        return np.where(A==False)[0][0]

    def bordpm(self,xmid,ymid):
        """
        trouver les extrémités
        """
        tol = 3
        i=self.detectEdge(self.image[ymid][xmid:],tol)
        y=self.detectEdge(self.image[ymid][xmid::-1],tol)
        return [xmid-y,xmid+i]

    def cotes(self,image,bord1,bord2,origine):
        """
        points des côtés
        """
        tol=5
        y,y1=[],[]
        for i in range(bord1,bord2):
            j=self.detectEdge(image[i][origine::-1],tol)
            k=self.detectEdge(image[i][origine:],tol)
            y.append(origine-j)
            y1.append(origine+k)
        return y,y1

    def interne(self,origine):
        lr=self.bordpm(origine[0],origine[1]) #bords interieurs
        #bords trou gauche
        xh  = np.linspace(lr[0],lr[1]-1,lr[1]-lr[0],dtype='int')
        yh,yb = self.cotes(np.transpose(self.image),lr[0],lr[1],origine[1])
        xc = np.linspace(max(yb)-1,min(yh),max(yb)-min(yh),dtype='int')
        yc1,yc2 = self.cotes(self.image,min(yh),origine[1],xh[np.argmin(yh)])
        yc3,yc4 = self.cotes(self.image,origine[1],max(yb),xh[np.argmax(yb)])
        return yc1+yc3,yc2+yc4,xc,yh,yb,xh

    def interneComp(self,bords):

        a,b,c,d,e,f=self.interne(bords)
        x1,y1=ut.lissage(a[::-1],c)
        x2,y2=ut.lissage(b[::-1],c)
        x3,y3=ut.lissage(f,e)
        x4,y4=ut.lissage(f,d)
        # cote
        mx=np.argmax(y3)
        mn=np.argmin(y4)
        x,y=np.concatenate((x2,x3[mx:],x4[mn:])),np.concatenate((y2,y3[mx:],y4[mn:]))
        ordre=np.argsort(y)
        x,y=x[ordre],y[ordre]
        xm,ym=ut.merge(y,x)
        # haut
        mn=np.argmin(y4)
        mm=np.argmin(x1)
        xa1,ya1=np.concatenate((x1[mm:],x4[:mn])),np.concatenate((y1[mm:],y4[:mn]))
        ordre=np.argsort(xa1)
        xa1,ya1=xa1[ordre],ya1[ordre]
        xa1=np.delete(xa1,0)
        ya1=np.delete(ya1,0)
        xma,yma=ut.merge(xa1,ya1)
        # bas
        mn=np.argmax(y3)
        mm=np.argmin(x1)
        xa1,ya1=np.concatenate((x1[:mm],x3[:mn])),np.concatenate((y1[:mm],y3[:mn]))
        ordre=np.argsort(xa1)
        xa1,ya1=xa1[ordre],ya1[ordre]
        xa1=np.delete(xa1,0)
        ya1=np.delete(ya1,0)
        xmb,ymb=ut.merge(xa1,ya1)
        self.contourGauche.append(np.concatenate((np.flip(xmb),xma,ym)))
        self.contourGauche.append(np.concatenate((np.flip(ymb),yma,xm)))


    def interneComp1(self,bords):
        #todo : cette section commune
        a,b,c,d,e,f=self.interne(bords)
        x1,y1=ut.lissage(a[::-1],c)
        x2,y2=ut.lissage(b[::-1],c)
        x3,y3=ut.lissage(f,e)
        x4,y4=ut.lissage(f,d)
        # cote
        mx=np.argmax(y3)
        mn=np.argmin(y4)
        x,y=np.concatenate((x1,x3[:mx],x4[:mn])),np.concatenate((y1,y3[:mx],y4[:mn]))
        ordre=np.argsort(y)
        x,y=x[ordre],y[ordre]
        xm,ym=ut.merge(y,x)
        # haut
        mn=np.argmin(y4)
        mx=np.argmax(x2)
        xa,ya=np.concatenate((x2[mx:],x4[mn:])),np.concatenate((y2[mx:],y4[mn:]))
        ordre=np.argsort(ya)
        xa,ya=xa[ordre],ya[ordre]
        xma,yma=ut.merge(ya,xa)
        # bas
        mn=np.argmax(y3)
        mx=np.argmax(y1)
        mm=np.argmax(x2)
        xa1,ya1=np.concatenate((x2[:mm],x3[mn:],x1[:mx])),np.concatenate((y2[:mm],y3[mn:],y1[:mx]))
        ordre=np.argsort(xa1)
        xa1,ya1=xa1[ordre],ya1[ordre]
        xmb,ymb=ut.merge(xa1,ya1)
        self.contourDroit=[np.concatenate((np.flip(ym),yma,np.flip(xmb))),np.concatenate((np.flip(xm),xma,np.flip(ymb)))]

    def contour(self):
        self.interneComp(self.centreDroit)
        self.interneComp1(self.centreGauche)

print("module interieur charge")
if __name__=="__main__":
    image = img.imread('../mailleSelection/maille'+str(no)+'.jpg')
    image = image[:, :, 1]
    interi=interieur(image,[813,382],[425,363])#
    interi.contour()
    # xa,ya=interi.contourGauche[0],interi.contourGauche[1]
    xb,yb=interi.contourDroit[0],interi.contourDroit[1]
    plt.plot(xa,ya)
    # plt.plot(xb,yb)
    plt.axis("equal")
    plt.show()