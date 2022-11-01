# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 17:04:54 2022

@author: kevin
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path

from typing import List, Optional
from numpy.typing import ArrayLike, DTypeLike


class RML:
    def __init__(self,
                 filePaths: List[str],
                 excludeDefaultRow: Optional[bool] = False,
                 permuteTimeSeries: Optional[bool] = True,
                 vectorizeRatings: Optional[bool] = True,
                 dtype: DTypeLike = np.float64) \
            -> None:
        self.data: np.ndarray = np.array([],dtype=dtype)
        self.filePaths: List[str] = filePaths
        self.lenTimeSequence: int = len(filePaths)
        self.excludeDefaultRow: bool = excludeDefaultRow
        self.permuteTimeSeries: bool = permuteTimeSeries
        self.vectorizeRatings: bool = vectorizeRatings
        self.dtype = dtype
        self.ratings: List[str] = []
        self.Kcols : int = -1
        self.Krows: int = -1

    def loadData(self) \
            -> None:
        data: List[np.ndarray] = []
        for filePath in self.filePaths:
            path = Path.cwd() / filePath
            files = path.glob('*.csv')

            temp: List[np.ndarray] = []
            for f in files:
                df = pd.read_csv(f.absolute(), index_col=0, delimiter=';')
                temp.append(df.to_numpy().astype(self.dtype))

            tempArray = np.array(temp,dtype=self.dtype)
            if self.excludeDefaultRow:
                tempArray = tempArray[:, :-1, :]
            if df.index.name == '%':
                tempArray /= 100
            data.append(tempArray)
        self.ratings: List[str] = df.columns.tolist()
        colLen = len(self.ratings)
        rowLen = colLen - int(self.excludeDefaultRow)
        self.Krows = rowLen
        self.Kcols = colLen
        if self.permuteTimeSeries and self.lenTimeSequence > 1:
            numEntries = []
            for i in range(0, self.lenTimeSequence):
                numEntries.append(np.arange(0, data[i].shape[0], dtype=np.int64))
            permuList = np.meshgrid(*numEntries, indexing='ij')
            permArray = permuList[0].ravel().reshape(-1, 1)
            for i in range(1, self.lenTimeSequence):
                permArray = np.concatenate((permArray, permuList[i].ravel().reshape(-1, 1)), axis=1)
            self.data = np.zeros((permArray.shape[0], self.lenTimeSequence, rowLen, colLen),dtype=self.dtype)
            for i in range(0, permArray.shape[0]):
                currSeq = []
                for j in range(0, permArray.shape[1]):
                    currSeq.append(data[j][permArray[i, j], :, :])
                self.data[i, :, :, :] = np.array(currSeq)
            # self.data = self.data.transpose((0, 2, 3, 1))
        elif self.lenTimeSequence > 1:
            raise ValueError('Not yet implemented please set permuteTimeSeries = True')
        else:
            self.data = data[0]
            # add a new axis for sequence index <-> "batch size, timesteps, features"
            self.data = self.data[:, np.newaxis, :, :]
        if self.vectorizeRatings:
            self.data = np.reshape(self.data,(self.data.shape[0],self.data.shape[1],self.data.shape[2]*self.data.shape[3]))

    def testRatingProperties(self):
        if self.vectorizeRatings:
            data=self.data.reshape(self.data.shape[0],self.data.shape[1],self.Krows,self.Kcols)
        else:
            data = self.data
        mDC = RML.monotoneDefaultColumn(data)
        sDD = RML.stronglyDiagonalDominant(data)
        dML = RML.downMoreLikely(data)
        iRS = RML.increasingRatingSpread(data)
        print(mDC)
        print(sDD)
        print(dML)
        print(iRS)
    @staticmethod
    def monotoneDefaultColumn(data : np.ndarray)\
            -> np.ndarray:
        value = np.diff(data[:, :, :, -1],n=1,axis=2)
        return np.squeeze(np.mean(value>=0,axis=0)).T

    @staticmethod
    def stronglyDiagonalDominant(data : np.ndarray)\
            -> np.ndarray:
        valueDiag = data[:,:,np.arange(0,data.shape[2]),np.arange(0,data.shape[3])]
        dataTemp = data.copy()
        dataTemp[:,:,np.arange(0,data.shape[2]),np.arange(0,data.shape[3])]=0
        valueOffDiag=np.sum(dataTemp,axis=3)
        return  np.mean(valueDiag-valueOffDiag>0,axis=0)

    @staticmethod
    def downMoreLikely(data : np.ndarray)\
            -> np.ndarray:
        upper = np.triu(data,k=1)
        lower = np.tril(data,k=-1)
        upperValue = np.sum(upper,axis=(2,3))
        lowerValue = np.sum(lower, axis=(2, 3))
        return np.mean(upperValue-lowerValue>0,axis=0)

    @staticmethod
    def increasingRatingSpread(data : np.ndarray)\
            -> np.ndarray:
        valueDiag = data[:, :, np.arange(0, data.shape[2]), np.arange(0, data.shape[3])]
        # defaultValue = data[:,:,:,-1]
        # return np.mean(np.diff(valueDiag-defaultValue,n=1,axis=1)<=0,axis=0)
        return np.mean(np.diff(valueDiag, n=1, axis=1) <= 0, axis=0).T

    def tfData(self,
               batch_size : Optional[int] = None) \
            -> tf.Tensor:
        if batch_size is None:
            return tf.convert_to_tensor(self.data)
        else:
            return tf.convert_to_tensor(self.data[:(self.data.shape[0]//batch_size)*batch_size,:,:])


if __name__ == '__main__':
    import time as timer

    filePaths = ['../Data/'+'SP_' + str(x) + '_month_small' for x in [1, 3, 6, 12]]
    rml = RML(filePaths, excludeDefaultRow=False, dtype=np.float32)
    # print(rml.filePaths)
    tic = timer.time()
    rml.loadData()
    ctime = timer.time() - tic
    print(f'Elapsed time {ctime} s')
    rml.testRatingProperties()
    # print(rml.data.dtype)
    # print(rml.data.shape)
    # # print(rml.ratings)
    # print(rml.tfData(batch_size=512).shape)
