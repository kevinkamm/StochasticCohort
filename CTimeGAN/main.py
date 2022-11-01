from RatingTimeGAN import TimeGAN,BrownianMotion,getTimeIndex,RML

import numpy as np
import tensorflow as tf

from typing import List

import time as timer

tf.config.set_visible_devices([], 'GPU')
print(tf.config.experimental.get_synchronous_execution())
print(tf.config.experimental.list_physical_devices())
print(tf.config.threading.get_inter_op_parallelism_threads())
print(tf.config.threading.get_intra_op_parallelism_threads())

'Data type for computations'
# use single precision for GeForce GPUs
dtype = np.float32

# seed for reproducibility
seed = 0
tf.random.set_seed(seed)

'Parameters for Brownian motion'
# time steps of Brownian motion, has to be such that mod(N-1,12)=0
# N = 5 * 12 + 1
N = 30 * 12 + 1
# trajectories of Brownian motion will be equal to batch_size for training
# M = batch_size = 1

'Load rating matrices'
# choose between 1,3,6,12 months
times = np.array([1, 3, 6, 12])
# times = np.array([1, 6, 12, 48])
lenSeq = times.size
T = times[-1] / 12

timeSpan : str = ''
if T<=1:
    timeSpan = 'shortTerm_' + '_'.join(map(str,times))
else:
    timeSpan = 'longTerm_' + '_'.join(map(str,times))

# Brownian motion class with fixed datatype
BM = BrownianMotion(T, N, dtype=dtype, seed=seed)
timeIndices = getTimeIndex(T, N, times / 12)

# relative path to rating matrices:
filePaths: List[str] = ['Data/'+'SP_' + str(x) + '_month_small' for x in times]
# exclude default row, don't change
# excludeDefaultRow = False
# permuteTimeSeries, don't change
# permuteTimeSeries = True
# load rating matrices
RML = RML(filePaths,
              dtype=dtype)
print('Load data')
ticRML = timer.time()
RML.loadData()
ctimeRML = timer.time() - ticRML
print(f'Elapsed time for loading data {ctimeRML} s.')

'Build GAN'
# number of ratings
Krows = RML.Krows
Kcols = RML.Kcols
# batch size
batch_size = 128
# training data
rm_train = RML.tfData()
print(f'Data shape: (Data,Time Seq,From Rating*To Rating)={rm_train.shape}')

# buffer size should be greater or equal number of data,
# is only important if data doesn't fit in RAM
buffer_size = rm_train.shape[0]

dataset = tf.data.Dataset.from_tensor_slices(rm_train)
dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True).batch(batch_size,drop_remainder=True)

epochs = 40
saveDir = 'RatingTimeGAN/modelParams'+'_'+timeSpan
tGAN = TimeGAN(lenSeq, Krows, Kcols, batch_size, BM, timeIndices, dtype=dtype)
tGAN.trainTimeGAN(dataset, epochs, loadDir=saveDir)
tGAN.save(saveDir)
samples = tGAN.sample(10)
print(samples.shape)
for wi in range(0, 3):
    print(f'Trajectory {wi}\n')
    for ti in range(0, samples.shape[1]):
        print(f'Time {timeIndices[ti]}')
        print(samples[wi, ti, :, :])
        print(np.sum(samples[wi, ti, :, :], axis=1))

# saveCSVDir = 'RatingTimeGAN/CSV'+'_'+timeSpan
# print('Save CSV_shortTerm_1_3_6_12')
# ticCSV=timer.time()
# tGAN.exportToCSV(100,saveCSVDir,ratings = RML.ratings)
# ctimeCSV=timer.time()-ticCSV
# print(f'Elapsed time for saving CSV_shortTerm_1_3_6_12 files: {ctimeCSV} s')

cohortSeries=tGAN.loadCohortTimeSeries('Data/SP_cohortTruth_1_3_6_12',
                                        'TimeGAN',
                                        2022,
                                        4,
                                        times)
cohortSeries=cohortSeries.reshape(lenSeq,Krows,Kcols)
for ti in range(0,lenSeq):
    cohortSeries[ti,0,0]=cohortSeries[ti,0,0]*(1 - (ti + 1)/lenSeq * 0.1)
    cohortSeries[ti, 1, 1] = cohortSeries[ti,1,1]*(1 - (ti + 1)/lenSeq * 0.2)
    cohortSeries[ti, 1, -1] =cohortSeries[ti,1,-1]*(1 - (ti + 1)/lenSeq * 0.2)
    cohortSeries[ti, 2, 1] = cohortSeries[ti, 2, 1] * (1 - (ti + 1)/lenSeq * 0.2)
    cohortSeries[ti, 2, 2] = cohortSeries[ti, 2, 2] * (1 - (ti + 1)/lenSeq * 0.1)
    cohortSeries[ti, 2, 3] = cohortSeries[ti, 2, 3] * (1 - (ti + 1)/lenSeq * 0.2)
tGAN.cohortToCSV('Data/SP_cohort_1_3_6_12',
                 'TimeGAN_2022_4',
                 cohortSeries,
                 times/12,
                 ratings = RML.ratings)
cohortSeries=tGAN.loadCohortTimeSeries('Data/SP_cohort_1_3_6_12',
                                        'TimeGAN',
                                        2022,
                                        4,
                                        times)
# C=cohortSeries.reshape(lenSeq,-1)

tGAN.trainConditionalGenerator(cohortSeries,10000)
Crec=tf.convert_to_tensor(tGAN.stochasticReconstruction(cohortSeries,10)).numpy().reshape(-1,lenSeq,Krows,Kcols)
print(Crec[0,-1,:,:])
print(Crec[30,-1,:,:])
print(np.mean(Crec,0))
tGAN.cohortToCSV('RatingTimeGAN/cohort_'+timeSpan,
                 'TimeGAN_2022_4',
                 np.mean(Crec,0,keepdims=False),
                 times/12,
                 ratings = RML.ratings)
# print(tf.reduce_sum(tf.reshape(Crec[1][0],(-1,lenSeq,Krows,Kcols)),3))
