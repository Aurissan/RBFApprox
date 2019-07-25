##preprocess input files

import numpy as np
import pandas as pd
import re
from itertools import product
import random

def _getfile(fname): #check extension and table header
    index = 1
    while index < len(fname):
        if fname[-index] == ".":
            extension = fname[-index + 1:]
            break
        index += 1
    if (extension == "xlsx") | (extension == "xls") | (extension == "xlsm"):
        data = pd.read_excel(fname, dtype='float64')
    elif (extension == "dat") | (extension == "txt") | (extension == "glo"):
        data = pd.read_table(fname, dtype='float64')

    r = re.compile("x|y.")

    header=data.columns.tolist()
    if not header:
        return "ER0" #no column names
    else:
        for i in header:
            if not r.match(i):
                return "ER1"

        else:
            return data

def _getnumbers(dframe): #check amount of inputs and outputs
    header=dframe.columns.tolist()
    countx=0
    county=0
    for i in range(0, len(header)):
        header[i]=header[i].lower()
        if (header[i].lower()).startswith('x'):
            countx+=1
        if (header[i].lower()).startswith('y'):
            county+=1
    return countx, county

def readtrain(filearr,  numskip=1, predflag=False): #normcheck, normrange,

    try:
        traindata = _getfile(filearr[0])
    except ValueError:
        return "ER2" #file has non-numeric values

    if (not isinstance(traindata, pd.DataFrame)) and traindata=="ER1":
        return "ER1"
    elif  (not isinstance(traindata, pd.DataFrame)) and traindata=="ER0": return "ER0"
    else:
        for i in filearr[1:]:
            trdt = _getfile(i)
            if (not isinstance(trdt, pd.DataFrame))and trdt=="ER1":
                return "ER1" #columns are not named properly

            else:
                traindata = traindata.append(trdt, ignore_index=True, sort=False)


        if numskip==0: numskip = 1
        traindata=traindata[::numskip] #skip every numskip index
        traindata.drop_duplicates(inplace=True)
        traindata.dropna(inplace=True)
        traindata.reset_index(drop=True, inplace=True)
        #traindata.sort_values(by=['x1'], inplace=True)
        trainarr=traindata.values
        #trainarr = trainarr[np.lexsort((trainarr[:, 3], trainarr[:, 0]))] ####temporary

        xdim, ydim = _getnumbers(traindata)

        if xdim ==0:
            if predflag:
                return trainarr
            else: return "ER1"

        if ydim ==0:
            return trainarr,trainarr

        #make separate arrays for input and output data
        trainsplits=np.hsplit(trainarr, xdim+ydim)

        trainx=[]
        trainy=[]

        for i in range (0, xdim):
            trainx.append(trainsplits[i])
        for i in range (xdim, ydim+xdim):
            trainy.append(trainsplits[i])


        trainX=np.concatenate(trainx,axis=1)
        trainY=np.concatenate(trainy,axis=1)
        return trainX, trainY


def _getminmax(arr): #find the minimum and maximum values of input array
    inpts=np.hsplit(arr, arr.shape[1])
    xmax=[]
    xmin=[]
    for i in inpts:
        xmax.append(np.amax(i))
        xmin.append(np.amin(i))

    return xmin,xmax

def lincen(arr, nums): #generate nums centers that are equidistant from each other
    xmax, xmin=_getminmax(arr)
    cenlist=[]
    cenlist2=[]
    for i in range(0, len(xmax)):
        cenlist.append(np.linspace(xmin[i],xmax[i],nums))
    for i in cenlist:
        cenlist2.append(np.unique(i,axis=0))
    cenlist=None
    cenlist3=[]
    for prod in product(*cenlist2):
        cenlist3.append(prod)
    centers=np.array(cenlist3)
    return centers

def rancen(arr, nums): #generate nums random centers
    xmin,xmax= _getminmax(arr)
    cenlist=[]
    rans=[]
    for i in range(0, len(xmax)):
        rans=[]
        for j in range (nums):
            rans.append(random.uniform(xmin[i],xmax[i]))
        cenlist.append(rans)
    cenlist2=[]
    for i in cenlist:
        cenlist2.append(np.unique(i,axis=0))
    cenlist=None
    cenlist3=[]
    for prod in product(*cenlist2):
        cenlist3.append(prod)
    centers=np.array(cenlist3)
    return centers





