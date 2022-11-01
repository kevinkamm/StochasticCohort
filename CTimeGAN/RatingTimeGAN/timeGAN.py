# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 17:28:08 2022

@author: kevin
this code is an adaption of
https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/timeseries/timegan/model.py
"""
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import GRU, Dense, Reshape
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam

import time as timer

from tqdm import tqdm, trange

import RatingTimeGAN
import RatingTimeGAN.brownianMotion as bm
import RatingTimeGAN.loadRatingMatrices as lrm

from pathlib import Path
import shutil

from typing import List, Optional
from numpy.typing import DTypeLike

# force tensorflow to use CPU, has to be on start-up
# tf.config.threading.set_inter_op_parallelism_threads(2)
# tf.config.threading.set_intra_op_parallelism_threads(6)
# from datetime import datetime
# logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
#                                                       histogram_freq = 1,
#                                                       profile_batch = '500,510')
class Generator(Model):
    def __init__(self,
                 hiddenDims: List[int],
                 outputDims: int,
                 RNNLayer : Optional[tf.keras.layers.Layer] = GRU) \
            -> None:
        self.hiddenDims = hiddenDims
        self.outputDims = outputDims
        self.RNNLayer = RNNLayer

    def build(self) \
            -> Model:
        model = Sequential(name='Generator')
        for i, hD in enumerate(self.hiddenDims):
            model.add(self.RNNLayer(units=hD,
                          return_sequences=True,
                          name=f'Generator_{self.RNNLayer.__name__}_{i + 1}'))
        model.add(Dense(units=self.outputDims,
                        activation='sigmoid',
                        name='Generator_OUT'))
        return model

class ConditionalGeneratorRNN(Model):
    def __init__(self,
                 hiddenDims: List[int],
                 outputDims: int,
                 RNNLayer : Optional[tf.keras.layers.Layer] = GRU) \
            -> None:
        self.hiddenDims = hiddenDims
        self.outputDims = outputDims
        self.RNNLayer = RNNLayer

    def build(self) \
            -> Model:
        model = Sequential(name='Generator')
        for i, hD in enumerate(self.hiddenDims):
            model.add(self.RNNLayer(units=hD,
                          return_sequences=True,
                          name=f'ConditionalGenerator_{self.RNNLayer.__name__}_{i + 1}'))
        model.add(Dense(units=self.outputDims,
                        activation='sigmoid',
                        name='ConditionalGenerator_OUT'))
        return model
class ConditionalGeneratorDense(Model):
    def __init__(self,
                 hiddenDims: List[int],
                 outputDims: int) \
            -> None:
        self.hiddenDims = hiddenDims
        self.outputDims = outputDims

    def build(self) \
            -> Model:
        model = Sequential(name='ConditionalGenerator')
        for i, hD in enumerate(self.hiddenDims):
            model.add(Dense(units=hD,
                            activation='sigmoid',
                            name=f'ConditionalGenerator_{i + 1}'))
        model.add(Dense(units=self.outputDims,
                        activation='sigmoid',
                        name='ConditionalGenerator_OUT'))
        # model.add(Reshape((self.outputDims[0],self.outputDims[1])))
        return model

class Discriminator(Model):
    def __init__(self,
                 hiddenDims: List[int],
                 RNNLayer : Optional[tf.keras.layers.Layer] = GRU):
        self.hiddenDims = hiddenDims
        self.RNNLayer = RNNLayer

    def build(self):
        model = Sequential(name='Discriminator')
        for i, hD in enumerate(self.hiddenDims):
            model.add(GRU(units=hD,
                          return_sequences=True,
                          name=f'Discriminator_{self.RNNLayer.__name__}_{i + 1}'))
        model.add(Dense(units=1,
                        activation='sigmoid',
                        name='Discriminator_OUT'))
        return model


class Recovery(Model):
    def __init__(self,
                 hiddenDims: List[int],
                 featureDims: int,
                 RNNLayer : Optional[tf.keras.layers.Layer] = GRU) \
            -> None:
        self.hiddenDims = hiddenDims
        self.featureDims = featureDims
        self.RNNLayer = RNNLayer

    def build(self) \
            -> Model:
        model = Sequential(name='Recovery')
        for i, hD in enumerate(self.hiddenDims):
            model.add(GRU(units=hD,
                          return_sequences=True,
                          name=f'Recovery_{self.RNNLayer.__name__}_{i + 1}'))
        model.add(Dense(units=self.featureDims,
                        activation='sigmoid',
                        name='Recovery_OUT'))
        return model


class Embedder(Model):
    def __init__(self,
                 hiddenDims: List[int],
                 outputDims: int,
                 RNNLayer : Optional[tf.keras.layers.Layer] = GRU) \
            -> None:
        self.hiddenDims = hiddenDims
        self.outputDims = outputDims
        self.RNNLayer = RNNLayer

    def build(self) \
            -> Model:
        model = Sequential(name='Embedder')
        for i, hD in enumerate(self.hiddenDims):
            model.add(GRU(units=hD,
                          return_sequences=True,
                          name=f'Embedder_{self.RNNLayer.__name__}_{i + 1}'))
        model.add(Dense(units=self.outputDims,
                        activation='sigmoid',
                        name='Embedder_OUT'))
        return model


class Supervisor(Model):
    def __init__(self,
                 hiddenDims: List[int],
                 outputDims: int,
                 RNNLayer : Optional[tf.keras.layers.Layer] = GRU) \
            -> None:
        self.hiddenDims = hiddenDims
        self.outputDims = outputDims
        self.RNNLayer = RNNLayer

    def build(self) \
            -> Model:
        model = Sequential(name='Supervisor')
        for i, hD in enumerate(self.hiddenDims):
            model.add(GRU(units=hD,
                          return_sequences=True,
                          name=f'Supervisor_GRU_{i + 1}'))
        model.add(Dense(units=self.outputDims,
                        activation='sigmoid',
                        name='Supervisor_OUT'))
        return model


class TimeGAN:
    def __init__(self,
                 lenSeq: int,
                 Krows: int,
                 Kcols: int,
                 batch_size: int,
                 BM : RatingTimeGAN.BrownianMotion,
                 timeInd : np.ndarray,
                 dtype: DTypeLike = np.float32):

        self.BM = BM
        self.timeInd = timeInd

        self.lenSeq = lenSeq
        self.Krows = Krows
        self.Kcols = Kcols
        self.batch_size = batch_size
        self.dtype = dtype
        'Input placeholders'
        # Placeholder for real data
        X = Input(shape=[lenSeq, Krows * Kcols], batch_size=batch_size, name='RealData')
        # Placeholder for noise
        Z = Input(shape=[lenSeq, 1], batch_size=batch_size, name='BM')

        # Network compatibility:
        # X -> embedder -> supervisor -> recovery -> X (Supervised Autoencoder)
        # X -> embedder -> recovery -> X (Unsupervised Autoencoder)
        # X -> embedder -> discriminator -> 1 (Embedded Discriminator)
        # Z -> generatorEmbedded -> supervisor -> discriminator -> 1 (Supervised GAN)
        # Z -> generatorEmbedded -> recovery -> X (TimeGAN generator)
        self.embedder = Embedder([3, 2, 3], lenSeq).build()
        self.recovery = Recovery([3, 2, 3], Krows * Kcols).build()
        self.supervisor = Supervisor([lenSeq, lenSeq], lenSeq).build()
        self.embeddedGenerator = Generator([lenSeq, lenSeq, lenSeq], lenSeq).build()
        self.discriminator = Discriminator([lenSeq, lenSeq, lenSeq]).build()

        # Autoencoder for parameter reduction
        H = self.embedder(X)
        X_tilde = self.recovery(H)
        self.autoencoder = Model(inputs=X,
                                 outputs=X_tilde,
                                 name='Autoencoder')

        # Supervised GAN
        E_hat = self.embeddedGenerator(Z)
        H_hat = self.supervisor(E_hat)
        Y_fake = self.discriminator(H_hat)
        self.supervisedGAN = Model(inputs=Z,
                                   outputs=Y_fake,
                                   name='supervisedGAN')

        # Adversarial architecture in latent space
        Y_fake_e = self.discriminator(E_hat)
        self.embeddedGAN = Model(inputs=Z,
                                 outputs=Y_fake_e,
                                 name='embeddedGAN')

        # Synthetic data generation
        X_hat = self.recovery(H_hat)
        self.generator = Model(inputs=Z,
                               outputs=X_hat,
                               name='TimeGANgenerator')

        # Final discriminator model
        Y_real = self.discriminator(H)
        self.discriminator_model = Model(inputs=X,
                                         outputs=Y_real,
                                         name='RealDiscriminator')

        self.mse = MeanSquaredError()
        self.bce = BinaryCrossentropy()
        self.gamma = 1
        self.epochs = 10
        self._isloaded = False

        # Conditional GAN for cohort series
        # cZ = Input(shape=[lenSeq, 1], batch_size=1, name='cBM') #1BM for each time
        # cZ = Input(shape=[lenSeq, Krows*Kcols], batch_size=1, name='cBM') #1BM for each time and entry
        cZ = Input(shape=[lenSeq, 1], batch_size=self.batch_size, name='cBM')  # 1BM for each time with batch
        self.conditionalGenerator = ConditionalGeneratorDense([Krows*Krows, Krows*Krows*lenSeq**2, Krows*Krows*lenSeq], Krows*Krows).build()
        # self.conditionalGenerator = ConditionalGeneratorRNN([Krows * Krows, Krows * Krows, Krows * Krows], Krows * Krows).build()
        C_hat=self.conditionalGenerator(cZ)
        self.conditionalGenerator_model = Model(inputs=cZ,outputs=C_hat)
        print(self.conditionalGenerator_model.summary())
        print(self.conditionalGenerator.summary())

    @tf.function
    def train_autoencoder(self, x, opt):
        with tf.GradientTape() as tape:
            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self.mse(x, x_tilde)
            e_loss_0 = 10 * tf.sqrt(embedding_loss_t0)

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss_0, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return tf.sqrt(embedding_loss_t0)


    @tf.function
    def train_supervisor(self, x, opt):
        with tf.GradientTape() as tape:
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self.mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

        var_list = self.supervisor.trainable_variables + self.generator.trainable_variables
        gradients = tape.gradient(generator_loss_supervised, var_list)
        apply_grads = [(grad, var) for (grad, var) in zip(gradients, var_list) if grad is not None]
        opt.apply_gradients(apply_grads)
        return generator_loss_supervised

    @tf.function
    def train_embedder(self, x, opt):
        with tf.GradientTape() as tape:
            # Supervised Loss
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self.mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

            # Reconstruction Loss
            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self.mse(x, x_tilde)
            e_loss = 10.0 * tf.sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return tf.sqrt(embedding_loss_t0)

    def discriminatorLoss(self, x,z):
        y_real = self.discriminator_model(x)
        discriminator_loss_real = self.bce(y_true=tf.ones_like(y_real),
                                      y_pred=y_real)

        y_fake = self.supervisedGAN(z)
        discriminator_loss_fake = self.bce(y_true=tf.zeros_like(y_fake),
                                      y_pred=y_real)

        y_fake_e = self.embeddedGAN(z)
        discriminator_loss_fake_e = self.bce(y_true=tf.zeros_like(y_fake_e),
                                        y_pred=y_fake_e)

        return (discriminator_loss_real +
                discriminator_loss_fake +
                self.gamma * discriminator_loss_fake_e)

    @staticmethod
    def calc_generator_moments_loss(y_true, y_pred):
        y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
        y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])
        g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
        g_loss_var = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1.0e-6) - tf.sqrt(y_pred_var + 1.0e-6)))
        return g_loss_mean + g_loss_var

    @tf.function
    def train_generator(self, x, z, opt):
        with tf.GradientTape() as tape:
            y_fake = self.supervisedGAN(z)
            generator_loss_unsupervised = self.bce(y_true=tf.ones_like(y_fake),
                                                   y_pred=y_fake)

            y_fake_e = self.embeddedGAN(z)
            generator_loss_unsupervised_e = self.bce(y_true=tf.ones_like(y_fake_e),
                                                y_pred=y_fake_e)

            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self.mse(h[:,1:,:],h_hat_supervised[:,:-1,:])

            x_hat = self.generator(z)
            generator_moment_loss = TimeGAN.calc_generator_moments_loss(x,x_hat)

            generator_loss = (generator_loss_unsupervised +
                              generator_loss_unsupervised_e +
                              100.0 * tf.sqrt(generator_loss_supervised) +
                              100.0 * generator_moment_loss)
        var_list = self.embeddedGenerator.trainable_variables + self.supervisor.trainable_variables
        gradients = tape.gradient(generator_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss

    @tf.function
    def train_discriminator(self, x, z, opt):
        with tf.GradientTape() as tape:
            discriminator_loss = self.discriminatorLoss(x,z)

        var_list = self.discriminator.trainable_variables
        gradients = tape.gradient(discriminator_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))

        return discriminator_loss

    def trainTimeGAN(self,dataset,epochs,loadDir: Optional[str] = ''):
        self.epochs = epochs
        if loadDir != '':
            if self.load(loadDir):
                return

        autoencoder_opt = Adam(1e-4)
        supervisor_opt = Adam(1e-4)
        generator_opt = Adam(1e-4)
        embedder_opt = Adam(1e-4)
        discriminator_opt = Adam(1e-4)

        # train embedder
        print('\ntrain autoencoder')
        for _ in tqdm(range(epochs), desc='Autoencoder network training'):
            for step, X_ in enumerate(dataset):
                step_e_loss_t0 = self.train_autoencoder(X_, autoencoder_opt)

        # train supervisor
        print('\ntrain supervisor')
        for _ in tqdm(range(epochs), desc='Supervised network training'):
            for step, X_ in enumerate(dataset):
                step_e_loss_t0 = self.train_supervisor(X_, supervisor_opt)

        # joint training

        step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0
        print('\ntrain joint model')
        for _ in tqdm(range(epochs), desc='Joint network training'):
            currK = 1
            for step, X_ in enumerate(dataset):
                Z_ = self.BM.tfW(self.batch_size,timeInd=self.timeInd)
                if currK <= 2:

                    #   Train generator

                    step_g_loss_u, step_g_loss_s, step_g_loss_v = self.train_generator(X_, Z_, generator_opt)

                    #   Train embedder

                    step_e_loss_t0 = self.train_embedder(X_, embedder_opt)
                    currK += 1
                else:
                    step_d_loss = self.discriminatorLoss(X_, Z_)
                    if step_d_loss > 0.15:
                        step_d_loss = self.train_discriminator(X_, Z_, discriminator_opt)
                    currK = 1

    def sample(self,n_batches: int):
        data = []
        for _ in trange(n_batches, desc='Synthetic data generation'):
            Z_ = self.BM.tfW(self.batch_size,timeInd=self.timeInd)
            records = self.generator(Z_)
            data.append(records)
        temp=np.array(np.vstack(data))
        return temp.reshape(temp.shape[0],temp.shape[1],self.Krows,self.Kcols)

    def exportToCSV(self,
                    n_batches:int,
                    saveDir : str,
                    ratings : Optional[List[str]] = None)\
            -> bool:
        savePath = Path.cwd() / Path(saveDir + '/' + self.paramString + f'_N{self.BM.N}')
        if savePath.exists():
            shutil.rmtree(savePath)
        savePath.mkdir(parents=True, exist_ok=True)
        data = []
        Z = []
        for _ in trange(n_batches, desc='Synthetic data generation'):
            Z_ = self.BM.sample(self.batch_size)
            records = self.generator(tf.convert_to_tensor(Z_[:,self.timeInd]))
            Z.append(Z_)
            data.append(records)
        fakeMatrices = np.array(np.vstack(data))
        fakeMatrices=fakeMatrices.reshape(fakeMatrices.shape[0], fakeMatrices.shape[1], self.Krows, self.Kcols)
        W = np.array(np.vstack(Z)).T
        t = np.linspace(0,self.BM.T,self.BM.N,endpoint=True).reshape(-1,1)
        for wi in range(0,W.shape[1]):
            dfW = pd.DataFrame(data=np.concatenate((t,W[:,wi].reshape(-1,1)),axis=1),columns=['Time','Brownian Path'])
            dfW.to_csv(str(savePath)+f'/W_{wi}.csv',sep=';')
            for ti in range(0,fakeMatrices.shape[1]):
                dfRM = pd.DataFrame(data=fakeMatrices[wi,ti,:,:],index=ratings,columns=ratings)
                dfRM.to_csv(str(savePath)+f'/RM_{wi}_{int(np.round(t[self.timeInd[ti]]*12))}.csv',sep=';')
        shutil.make_archive(str(savePath),'zip',Path.cwd() / Path(saveDir),self.paramString + f'_N{self.BM.N}')
        shutil.rmtree(savePath)
        return True

    @property
    def paramString(self)\
            -> str:
        return 'AE{0:d}_G{1:d}_lenSeq{2:d}_batch{3:d}_epochs{4:d}'.format(self.autoencoder.count_params(),
                                                              self.generator.count_params(),
                                                              self.lenSeq,
                                                              self.batch_size,
                                                              self.epochs)

    def save(self,
             saveDir : str)\
        -> None:
        if not self._isloaded:
            savePath=Path.cwd() / Path(saveDir +'/' + self.paramString)
            if savePath.exists():
                shutil.rmtree(savePath)
            savePath.mkdir(parents=True, exist_ok=True)
            self.generator.save(str(savePath)+'/'+'generator')
            self.autoencoder.save(str(savePath) + '/' + 'autoencoder')
            self.recovery.save(str(savePath) + '/' + 'recovery')
            self.embedder.save(str(savePath) + '/' + 'embedder')
            self.supervisor.save(str(savePath) + '/' + 'supervisor')
            self.discriminator.save(str(savePath) + '/' + 'discriminator')
            self.discriminator_model.save(str(savePath) + '/' + 'discriminator_model')
            self.embeddedGAN.save(str(savePath) + '/' + 'embeddedGAN')
            self.embeddedGenerator.save(str(savePath) + '/' + 'embeddedGenerator')

    def load(self,
             loadDir : str)\
            -> bool:
        loadPath = Path.cwd() / Path(loadDir + '/' + self.paramString)
        if loadPath.exists():
            self.generator = tf.keras.models.load_model(str(loadPath)+'/'+'generator')
            self.autoencoder = tf.keras.models.load_model(str(loadPath) + '/' + 'autoencoder')
            self.embedder = tf.keras.models.load_model(str(loadPath) + '/' + 'embedder')
            self.supervisor = tf.keras.models.load_model(str(loadPath) + '/' + 'supervisor')
            self.discriminator = tf.keras.models.load_model(str(loadPath) + '/' + 'discriminator')
            self.discriminator_model = tf.keras.models.load_model(str(loadPath) + '/' + 'discriminator_model')
            self.embeddedGAN = tf.keras.models.load_model(str(loadPath) + '/' + 'embeddedGAN')
            self.embeddedGenerator.save(str(loadPath) + '/' + 'embeddedGenerator')
            self._isloaded = True
            return True
        else:
            return False

    def loadCohortTimeSeries(self,
                             fileDir: str,
                             name: str,
                             year: int,
                             dataset: int,
                             months: List[int])\
            -> np.ndarray:
        files = [name+'_'+str(year)+'_'+str(dataset)+'_{0:1.2f}'.format(x/12)+'*y.csv' for x in months]
        cohortSeries: List[np.ndarray]=[]
        for file in files:
            path = Path.cwd()/fileDir
            for f in path.glob(file):
                cohortSeries.append(pd.read_csv(f, index_col=0, delimiter=';').to_numpy(dtype=self.dtype))
        C=np.array(cohortSeries)
        C = C.reshape(1,C.shape[0], -1)
        return C
    # @tf.function
    # def train_conditionalGenerator(self, c, z, opt):
    #     with tf.GradientTape() as tape:
    #         y = self.conditionalGenerator_model(z)
    #         h = self.embedder(c+y)
    #         h_supervised = self.supervisor(h)
    #
    #         temp = c+y
    #         temp = tf.reshape(temp,(-1,self.lenSeq,self.Krows,self.Kcols))
    #         rowSums = tf.reduce_sum(temp,3)
    #         d = self.discriminator(h)
    #         d_supervised = self.discriminator(h_supervised)
    #
    #         d_real = self.discriminator_model(c+y)
    #
    #         generator_loss_supervised = self.mse(h[:,1:,:],h_supervised[:,:-1,:])
    #         discriminator_loss_unsupervised = self.mse(d,tf.ones_like(d))
    #         discriminator_loss_supervised = self.mse(d_supervised,tf.ones_like(d_supervised))
    #
    #         # discriminator_loss_real = self.bce(y_true=tf.ones_like(d_real),
    #         #                                    y_pred=d_real)
    #         discriminator_loss_real = self.mse(d_real,tf.ones_like(d_real))
    #
    #
    #         # generator_loss = (generator_loss_supervised + discriminator_loss_unsupervised + discriminator_loss_supervised)
    #         generator_loss = discriminator_loss_unsupervised
    #         # generator_loss = discriminator_loss_supervised
    #         # generator_loss = discriminator_loss_real
    #
    #         generator_loss += tf.reduce_mean(tf.abs(rowSums-tf.ones_like(rowSums)),[0,1,2])
    #     var_list = self.conditionalGenerator.trainable_variables
    #     gradients = tape.gradient(generator_loss, var_list)
    #     opt.apply_gradients(zip(gradients, var_list))
    #     return generator_loss

    @tf.function
    def discriminate_conditionalGenerator(self, c, z, opt):
        with tf.GradientTape() as tape:
            y = self.conditionalGenerator_model(z)

            print(f'z shape {z.shape}')
            print(f'y shape {y.shape}')
            print(f'c shape {c.shape}')

            cMatrix = tf.reshape(c,(-1,self.lenSeq,self.Krows,self.Kcols))
            yMatrix = tf.reshape(y, (-1, self.lenSeq, self.Krows, self.Kcols))

            withdrawal = 1-tf.reduce_sum(cMatrix,3,keepdims=True)
            totalWeight = tf.reduce_sum(yMatrix,3,keepdims=True)

            print(f'withdrawal shape {withdrawal.shape}')
            print(f'total weight shape {totalWeight.shape}')

            weights = yMatrix/totalWeight

            print(f'weight shape {weights.shape}')

            cRec=cMatrix+withdrawal*weights
            temp=tf.reduce_sum(cRec[0, -1, :, :], 1)
            cRec=tf.reshape(cMatrix+withdrawal*weights,(-1, self.lenSeq, self.Krows* self.Kcols))
            # print(f' row sum {tf.reduce_sum(cRec[0,-1,:,:],1)}')

            d = self.discriminator_model(cRec)
            # h = self.embedder(cRec)
            # h_supervised = self.supervisor(h)
            # d = self.discriminator(h_supervised)


            # discriminator_loss_real = self.mse(d,tf.ones_like(d))
            # generator_loss = 10 * tf.sqrt(discriminator_loss_real)

            discriminator_loss_real = self.mse(d, tf.ones_like(d))
            generator_loss = discriminator_loss_real

            # discriminator_loss_real = self.bce(y_true=tf.ones_like(d),
            #                                    y_pred=d)
            # generator_loss = discriminator_loss_real

        var_list = self.conditionalGenerator.trainable_variables
        gradients = tape.gradient(generator_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return generator_loss
        # return temp

    @tf.function
    def encode_conditionalGenerator(self, c, z, opt):
        with tf.GradientTape() as tape:
            y = self.conditionalGenerator_model(z)

            print(f'z shape {z.shape}')
            print(f'y shape {y.shape}')
            print(f'c shape {c.shape}')

            cMatrix = tf.reshape(c,(-1,self.lenSeq,self.Krows,self.Kcols))
            yMatrix = tf.reshape(y, (-1, self.lenSeq, self.Krows, self.Kcols))

            withdrawal = 1-tf.reduce_sum(cMatrix,3,keepdims=True)
            totalWeight = tf.reduce_sum(yMatrix,3,keepdims=True)

            print(f'withdrawal shape {withdrawal.shape}')
            print(f'total weight shape {totalWeight.shape}')

            weights = yMatrix/totalWeight

            print(f'weight shape {weights.shape}')

            cRec=cMatrix+withdrawal*weights
            # temp=tf.reduce_sum(cRec[0, -1, :, :], 1)
            # cRecVar = tf.reduce_mean(tf.math.reduce_variance(tf.linalg.diag_part(cRec, k=0), axis=0, keepdims=True),
            #                          axis=2, keepdims=True)
            # cRecVar = tf.reduce_mean(tf.math.reduce_variance(cRec[:,:,:-1,:], axis=0, keepdims=True), axis=(2,3), keepdims=True)
            cRec=tf.reshape(cRec,(-1, self.lenSeq, self.Krows* self.Kcols))
            # print(f' row sum {tf.reduce_sum(cRec[0,-1,:,:],1)}')

            zVar = tf.math.reduce_variance(z, axis=0, keepdims=True)
            # zVar = tf.constant([1,3,6,12],shape=(1,4),dtype=tf.float32)/12
            cRecVar = tf.reduce_mean(tf.math.reduce_variance(cRec,axis=0,keepdims=True),axis=2,keepdims=True)


            # cRec=tf.reduce_mean(cRec,axis=0,keepdims=True)

            a = self.autoencoder(cRec)



            # discriminator_loss_real = self.mse(d,tf.ones_like(d))
            # generator_loss = 10 * tf.sqrt(discriminator_loss_real)

            recovery_loss_real = self.mse(cRec, a)
            variance_loss = self.mse(cRecVar,zVar)
            # generator_loss = recovery_loss_real + 100*variance_loss
            generator_loss = recovery_loss_real

            # discriminator_loss_real = self.bce(y_true=tf.ones_like(d),
            #                                    y_pred=d)
            # generator_loss = discriminator_loss_real

        var_list = self.conditionalGenerator.trainable_variables
        gradients = tape.gradient(generator_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return recovery_loss_real,variance_loss
        # return temp
        # return variance_loss

    def trainConditionalGenerator(self,
                                  cohortSeries: np.ndarray,
                                  epochs: int)\
            -> None:
        encodeGenerator_opt = Adam(1e-4)
        discriminateGenerator_opt = Adam(1e-4)
        k=0
        errEncode=100
        errDiscriminate=100
        for _ in (pbar := tqdm(range(epochs), desc='Conditional GAN')):
            # Z = self.BM.tfW(1, timeInd=self.timeInd)
            Z = self.BM.tfW(self.batch_size, timeInd=self.timeInd)
            # Z =tf.reshape(tf.transpose(self.BM.tfW(self.Kcols*self.Krows, timeInd=self.timeInd)),(1,self.lenSeq,self.Krows*self.Kcols))
            if k==3:
                errDiscriminate = self.discriminate_conditionalGenerator(cohortSeries, Z, discriminateGenerator_opt)
                k=0
            else:
                errEncode=self.encode_conditionalGenerator(cohortSeries,Z,encodeGenerator_opt )
                # k=k+1
            pbar.set_postfix({'encode error': errEncode,'disc error': errDiscriminate})

    def stochasticReconstruction(self,
                                 cohortSeries: np.ndarray,
                                 M: int)\
            -> np.ndarray:
        Crec = []
        C = cohortSeries.reshape(-1,self.lenSeq,self.Krows,self.Kcols)
        for _ in tqdm(range(M), desc='sample reconstruction'):
            # Z = self.BM.tfW(1, timeInd=self.timeInd)
            Z = self.BM.tfW(self.batch_size, timeInd=self.timeInd)
            print(Z[0,0])
            # Z =tf.reshape(tf.transpose(self.BM.tfW(self.Kcols*self.Krows, timeInd=self.timeInd)),(1,self.lenSeq,self.Krows*self.Kcols))
            rec = tf.reshape(self.conditionalGenerator(Z),(-1,self.lenSeq,self.Krows,self.Kcols))
            recWeights=rec/tf.reduce_sum(rec,3,keepdims=True)
            # Crec.append(cohortSeries+rec)
            Crec.append(C + recWeights * (1 - np.sum(C, 3, keepdims=True)))
        return Crec
    def cohortToCSV(self,
                    saveDir : str,
                    saveName: str,
                    recCohort: np.ndarray,
                    time: np.ndarray,
                    ratings : Optional[List[str]] = None)\
            -> bool:
        savePath = Path.cwd() / Path(saveDir)
        # if savePath.exists():
        #     shutil.rmtree(savePath)
        savePath.mkdir(parents=True, exist_ok=True)

        for ti in range(0,recCohort.shape[0]):
            dfRM = pd.DataFrame(data=recCohort[ti,:,:],index=ratings,columns=ratings)
            dfRM.to_csv(str(savePath)+f'/{saveName}_'+'{0:1.3f}y.csv'.format(time[ti]),sep=';')
        # shutil.make_archive(str(savePath),'zip',Path.cwd() / Path(saveDir),self.paramString + f'_N{self.BM.N}')
        # shutil.rmtree(savePath)
        return True

if __name__ == '__main__':
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
    N = 5 * 12 + 1
    # trajectories of Brownian motion will be equal to batch_size for training
    # M = batch_size = 1

    'Load rating matrices'
    # choose between 1,3,6,12 months
    times = np.array([1, 3, 6, 12])
    lenSeq = times.size
    T = times[-1] / 12

    timeSpan: str = ''
    if T <= 1:
        timeSpan = 'shortTerm_' + '_'.join(map(str, times))
    else:
        timeSpan = 'longTerm_' + '_'.join(map(str, times))

    # Brownian motion class with fixed datatype
    BM = bm.BrownianMotion(T, N, dtype=dtype, seed=seed)
    timeIndices = bm.getTimeIndex(T, N, times / 12)

    # relative path to rating matrices:
    filePaths: List[str] = ['../Data/'+'SP_' + str(x) + '_month_small' for x in times]
    # exclude default row, don't change
    # excludeDefaultRow = False
    # permuteTimeSeries, don't change
    # permuteTimeSeries = True
    # load rating matrices
    RML = lrm.RML(filePaths,
                  dtype=dtype)
    print('Load data')
    ticRML = timer.time()
    RML.loadData()
    ctimeRML = timer.time() - ticRML
    print(f'Elapsed time for loading data {ctimeRML} s.')

    'Build GAN'
    # training data
    rm_train = RML.tfData()
    print(f'Data shape: (Data,Time Seq,From Rating*To Rating)={rm_train.shape}')
    # number of ratings
    Krows = RML.Krows
    Kcols = RML.Kcols
    # batch size
    batch_size = 512

    # buffer size should be greater or equal number of data,
    # is only important if data doesn't fit in RAM
    buffer_size = rm_train.shape[0]

    dataset = tf.data.Dataset.from_tensor_slices(rm_train)
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True).batch(batch_size)

    epochs=10
    saveDir = 'modelParams' + timeSpan
    tGAN=TimeGAN(lenSeq, Krows, Kcols, batch_size, BM, timeIndices, dtype=dtype)
    tGAN.trainTimeGAN(dataset,epochs,loadDir = saveDir)
    tGAN.save(saveDir)
    samples=tGAN.sample(1)
    print(samples.shape)
    for wi in range(0,3):
        print(f'Trajectory {wi}\n')
        for ti in range(0,samples.shape[1]):
            print(f'Time {timeIndices[ti]}')
            print(samples[wi,ti,:,:])
            print(np.sum(samples[wi,ti,:,:],axis=1))
