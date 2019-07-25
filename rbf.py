#contains the RBF network implementations


import numpy as np
from numpy import linalg as la
import math
from itertools import product

#classic RBF Network for interpolation/function approximation
class RBF:
    def __init__(self, centers, r=1, function='gauss'):
        self.r=r #radius
        self.function=function
        self.centers=centers #array of function centers
        self.w=np.random.randn(self.centers.shape[0]) #initialize random weights

    def _gauss(self, x, c): #Gaussian
        self.alpha=1/(2*self.r**2) #controls width of the curve
        return math.exp(-self.alpha*(la.norm(x-c)**2))

    def _laplace(self, x, c): #Laplace distribution
        self.alpha = 1 / (2 * self.r ** 2)
        return math.exp(-self.alpha * (la.norm(x - c)))

    def _cauchy(self, x, c): #Cauchy distribution
        self.alpha = 1 / (2 * self.r ** 2)
        self.subbed=np.subtract(x,c)
        self.pwr=np.power(self.subbed, 2)
        self.t=1/(self.alpha*self.pwr+1)
        return 1/(self.alpha*np.sum(self.pwr)+1)

    def _linear2(self, x, c): # Piecewise linear RBF
        self.alpha = 1 / self.r
        self.out= 1 - self.alpha * la.norm(x - c)
        if self.out < 0: self.out=0
        return self.out

    def _wigner(self, x, c): #Wigner semicircle distribution
        self.alpha=self.r
        self.tempx=[]
        if type(x) is np.ndarray:
            for curx, curc in zip(x,c):
                if (curx<curc-self.alpha) or (curx>curc+self.alpha):
                    self.tempx=0
                else: self.tempx=np.append(self.tempx, self.alpha**2-(curx-curc)**2)
            self.out=(math.sqrt(np.sum(self.tempx)/x.shape[0]))/self.alpha
        else:
            if (x<c-self.alpha) or (x>c+self.alpha):
                self.out=0
            else: self.out=(math.sqrt(self.alpha**2-(x-c)**2))/self.alpha
        return self.out

    def _function_decide(self, x, c ): #calculate the output of a function based on a chosen RBF
        if (self.function=="gauss"):
            self.midres=self._gauss(x,c)
        if (self.function=="laplace"):
            self.midres=self._laplace(x,c)
        if (self.function=="cauchy"):
            self.midres=self._cauchy(x,c)
        if (self.function=="linear"):
            self.midres=self._linear(x,c)
        if (self.function == "linear2"):
            self.midres = self._linear2(x, c)
        if (self.function=="wigner"):
            self.midres=self._wigner(x,c)
        return self.midres

    def _hidden_layer(self, Xs): #calculate outputs of neurons in the hidden layer
        self.inpt = np.zeros((Xs.shape[0], self.centers.shape[0]))
        for i in range(0, Xs.shape[0]):
            for j in range(0, self.centers.shape[0]):
                self.inpt[i][j] = self._function_decide(Xs[i], self.centers[j])
        return self.inpt

    def fit(self, X, Y): #training using least squares
        self.input=self._hidden_layer(X)
        self.temp1=self.input.T
        self.temp2=self.temp1.dot(self.input)
        self.temp2=la.pinv(self.temp2)
        #self.temp2 = la.inv(self.temp2)
        self.temp2=self.temp2.dot(self.temp1)
        self.w=self.temp2.dot(Y)
        self.temp1, self.temp2, self.input = None, None, None
        return self

    def predict(self, X): #calculate the output of the network
        self.inpoot=self._hidden_layer(X)
        #self.out=np.zeros((X.shape[0], self.w.shape[1]))
        self.out=self.inpoot.dot(self.w)
        return self.out

    def score(self, x, y): #calculate error
        self.result=self.predict(x)
        self.max=np.amax(y)
        self.min=np.amin(y)
        self.delta=np.subtract(y,self.result)
        self.delta=np.abs(self.delta)
        self.deltab=np.sum(self.delta)
        return (self.deltab/(self.max-self.min))/self.result.shape[0]

#Alternative RBF-network
class TwoLayerRBF(RBF):

    def _hidden_layer(self, Xs, cntr): #calculate output of the first hidden layer
        self.inpt = np.zeros((Xs.shape[0], cntr.shape[0]))
        for i in range(0, Xs.shape[0]):
            for j in range(0, cntr.shape[0]):
                self.inpt[i][j] = self._function_decide(Xs[i], cntr[j])
        return self.inpt

    def _sum_layer(self, inxs): #sum the combinations of outputs of first hidden layer in the second hidden layer
        self.combxs=[]
        for prod in product(*inxs):
            self.combxs.append(prod)
        self.tempxs=np.array(self.combxs)
        self.combxs=None
        self.tempxs=np.sum(self.tempxs, axis=1)
        self.koef=1/(len(inxs)**len(inxs))
        for i in range (0, self.tempxs.shape[0]):
            for j in range (0, self.tempxs.shape[1]):
                self.tempxs[i][j]=self.koef*(self.tempxs[i][j]**len(inxs))
        return self.tempxs

    def fit(self, X, Y): #training
        self.xs = np.hsplit(X, X.shape[1]) #split the array of inputs into individual vectors
        self.unixlist=[]
        for t in self.xs:
            self.unixlist.append(np.unique(t))
        self.xs=None
        self.cxs=np.hsplit(self.centers, self.centers.shape[1]) #split the array of centers into ind. vectors
        self.unicxlist=[]
        for t in self.cxs:
            self.unicxlist.append(t)
        self.cxs=None
        self.inptlist=[]
        for i in range(0, len(self.unixlist)): #calculate the outputs of 1st hidden layer of each individual input
            self.inptlist.append(self._hidden_layer(self.unixlist[i], self.unicxlist[i]))
        self.input=self._sum_layer(self.inptlist) #combine the outputs in the second layer
        self.inptlist=None
        self.temp1 = self.input.T
        self.temp2 = self.temp1.dot(self.input)
        self.temp2 = la.pinv(self.temp2)
        self.temp2 = self.temp2.dot(self.temp1)
        self.w = self.temp2.dot(Y)
        self.temp1, self.temp2, self.input = None, None, None
        return self

    def predict(self, X): #output of the network
        self.pxs = np.hsplit(X, X.shape[1])
        self.xlist=[]
        for t in self.pxs:
            self.xlist.append(np.unique(t))
        self.pxs=None
        self.inpot=[]
        for i in range(0, len(self.xlist)):
            self.inpot.append(self._hidden_layer(self.xlist[i], self.unicxlist[i]))
        self.inpoot=self._sum_layer(self.inpot)
        self.out = self.inpoot.dot(self.w)
        return self.out



