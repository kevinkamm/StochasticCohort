# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:50:41 2022

@author: kevin
"""

import numpy as np
import tensorflow as tf

from typing import Optional, Union, Type
from numpy.typing import ArrayLike, DTypeLike


def getTimeIndex(T: float,
                 N: int,
                 t: ArrayLike,
                 endpoint: Optional[bool] = True)\
        -> ArrayLike:
    return np.floor(np.array(t) * (N - int(endpoint)) / T).astype(np.int64)

class BrownianMotion:
    def __init__(self,
                 T : float,
                 N : int,
                 dtype: DTypeLike = np.float64,
                 seed: Optional[int] = None)\
            -> None:
        self.T = T
        self.N = N
        self.dtype = dtype
        self.rng = np.random.default_rng(seed)

    def sample(self,
               M: int,
               timeInd: Optional[np.ndarray] = None)\
            -> np.ndarray:
        """
        Samples 'n' indpendent Brownian motions with 'N' time steps
        from 0 to 'T' and 'M' paths. Shape = (Time,Samples,NumberOfBM)

        Parameters
        ----------
        M : int
            Number of trajectories.
        timeInd : Optional[Union[str,np.ndarray]], optional
            Decide if you want the whole trajectory (all), only the 
            endpoint (end) or specific time indices given in a numpy array. 
            The default is 'all'.

        Returns
        -------
        Brownian motions
        """
        dW = np.sqrt(self.T / (self.N - 1)) * self.rng.standard_normal((M,self.N - 1)).astype(self.dtype)
        W = np.zeros((M,self.N), dtype=self.dtype)
        W[:,1:] = dW
        W = np.cumsum(W, axis=1)
        if isinstance(timeInd,np.ndarray) and timeInd.size>0:
            return W[:,timeInd]
        else:
            return W

    def tfW(self,
            M: int,
            timeInd: Optional[np.ndarray] = None)\
            -> tf.Tensor:
        return tf.convert_to_tensor(self.sample(M, timeInd=timeInd))


if __name__ == '__main__':
    T = 1
    N = 1000
    M = 10
    timeInd = np.array([0, int(N / 2), N - 1], dtype=np.int64)
    BM = BrownianMotion(T,N,dtype=np.float32,seed=0)
    W = BM.sample(M)
    print(W[:,timeInd])
    BM = BrownianMotion(T, N, dtype=np.float32, seed=0)
    print(BM.tfW(M,timeInd=timeInd))

