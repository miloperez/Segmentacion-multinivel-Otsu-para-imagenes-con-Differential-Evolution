import numpy as np
import random as rnd


class DiffEvo:
    def __init__(self, D, low, high, F, Cr, NP, MaxGen, fitness):
        self.D = D  # dimensiones
        self.low = low  # valor minimo (indice)
        self.high = high  # valor máximo (indice)
        self.F = F  # tasa de mutación
        self.Cr = Cr  # tasa de cruza
        self.NP = NP  # tamaño de población
        self.MxG = MaxGen  # total de generaciones
        self.fit = fitness  # función objetivo

    def getBest(self):
        # generar miembros de la generación de acuerdo con los max y min especificados
        X = np.random.randint(self.low, self.high - self.low, size=(self.NP, self.D))
        # inicializar sujeto de cambio
        V = np.zeros(self.D)

        # total de generaciones
        for G in range(self.MxG):
            # total de miembros de la población
            for i in range(0, self.NP):
                # seleccionar tres sujetos de la población, excluyendo el sujeto que está siendo evaluado
                # para posteriormente construir al sujeto de cambio
                rs = self.__choices(range(0, self.NP), 3, i)
                # seleccionar indice aleatorio
                j_rand = np.random.randint(0, self.D)
                # para cada dimension
                for j in range(self.D):
                    # cruza y mutación
                    if np.random.uniform(0, 1) < self.Cr or j == j_rand:
                        V[j] = X[rs[0]][j] + (self.F * (X[rs[1]][j] - X[rs[2]][j]))
                    else:
                        V[j] = X[i][j]
                # revisar que el sujeto de cambio no tenga valores fuera de los límites
                for d in range(self.D):
                    if V[d] < self.low:
                        V[d] = self.low
                    elif V[d] > self.high - self.low:
                        V[d] = self.high - self.low
                # si el fitness del sujeto de cambio es mejor que el sujeto evaluado, reemplazarlo
                if self.fit(V) < self.fit(X[i]):
                    X[i] = V

        # seleccionar al mejor de la población
        best = self.fit(X[0])
        selec = X[0]
        for k in X:
            var = self.fit(k)
            if var < best:
                best = var
                selec = k

        # regresa el vector ordenado
        return sorted(selec)

    # seleccionar n elementos del arreglo arr, excluyendo exc
    def __choices(self, Arr, n, exc):
        while True:
            res = rnd.choices(Arr, k=n)
            if exc not in res:
                return res
