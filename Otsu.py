import numpy as np
import collections as coll
import cv2
import DifferentialEvolution as DE


class Otsu:
    def __init__(self, file, K):
        # número de cortes
        self.K = K
        self.Img = loadImgBW(file)
        self.N = len(self.Img)

        # obtener valores mínimos y máximos presentes en la imagen
        diffGray = np.unique(self.Img)
        self.zero = int(diffGray.min())
        self.L = int(diffGray.max())

        # arreglo de valores posibles
        self.grays = np.arange(self.zero, self.L+1)

        # obtener histograma
        self.Hist = np.zeros(len(self.grays))
        for i in range(len(self.Hist)):
            self.Hist[i] = int(list(self.Img).count(self.grays[i]))

        # obtener cortes a partir de la evolución diferencial
        umb = DE.DiffEvo(self.K, self.zero, self.L, 0.8, 0.9, 30, 25, self.sigma2B)
        self.umbrales = umb.getBest()

    # recibe un vector con los índices de corte, aka fitness
    def sigma2B(self, T):
        # obtener Ck de acuerdo con los cortes propuestos
        Ck = self.fillCk(self.extend(T))

        # obtener probabilidad acumulada, media por grupo y media global
        w_k = np.zeros(self.K + 1)
        mu_k = np.zeros(self.K + 1)
        mu_T = 0
        for i in range(self.K + 1):
            w_k[i] = self.getW_k(np.array(Ck[i]))
            mu_k[i] = self.getMu_k(Ck[i], w_k[i])
            mu_T += w_k[i] * mu_k[i]

        # obtener varianza inversa
        sigma2B = 0
        for i in range(self.K + 1):
            sigma2B += w_k[i] * pow(mu_k[i] - mu_T, 2)

        return 1/sigma2B

    # obtener probabilidad acumulada por grupo
    def getW_k(self, Ck):
        sumW_k = 0
        for i in Ck:
            sumW_k += (self.Hist[i]) / self.N
        return sumW_k

    # obtener media por grupo
    def getMu_k(self, Ck, w_k):
        sumMu_k = 0
        for i in Ck:
            sumMu_k += (((self.Hist[i]) / self.N) * i)/w_k
        return sumMu_k

    # extender vector de índices de corte para que incluya el gris min y el gris max
    def extend(self, T):
        T_ext = np.zeros(len(T) + 2)
        T_ext[0] = 0
        T_ext[len(T_ext) - 1] = (self.L - self.zero + 1)
        T = sorted(T)
        for j in range(len(T)):
            T_ext[j + 1] = int(T[j])

        return T_ext

    # llenar los Ck de acuerdo con los índices propuestos
    def fillCk(self, T):
        Ck = coll.deque()
        # los guardamos como enteros para posteriormente evaluarlos como índices
        for c in range(self.K + 1):
            Ck.append(np.array(list(map(int, np.arange(T[c], T[c + 1])))))
        return Ck

    def getTransImg(self):
        # obtener Ck de acuerdo con los cortes óptimos encontrados
        Ck = self.fillCk(self.extend(self.umbrales))

        # igualar cada pixel con la media del grupo al que pertenece
        for i in range(len(self.Img)):
            for c in Ck:
                if self.Img[i] in self.grays[np.array(list(map(int, c)))]:
                    self.Img[i] = int(np.mean(np.array(c)))

        # regresa vector de pixeles de gris ya segmentado
        return self.Img


# cargar imagen en B&W a partir del nombre del archivo
def loadImgBW(file):
    img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape
    B = cv2.split(img)
    B = np.reshape(B, (rows * cols))
    return np.array(B)
