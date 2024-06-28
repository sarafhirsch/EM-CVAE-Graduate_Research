'''
    Adapted from 
https://www.tensorflow.org/tutorials/generative/cvae
https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py
    with input from
https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py
    Transposed convolution guide:
https://arxiv.org/pdf/1603.07285v1.pdf
'''

#!
import os
import numpy as np
import time
# from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import glob
# import PIL
# import imageio
import matplotlib.pyplot as plt

# from keras.models import Sequential, Model
# from keras.layers import Input, Dense, Activation, Flatten, Reshape
# from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
# from keras.layers import LeakyReLU, Dropout
# from keras.layers import BatchNormalization, Lambda
# from keras.optimizers import Adam, RMSprop, SGD
# from keras import backend as K


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

def soft_binary_accuracy(y_true, y_pred):
    ''' get accuracy even using soft labels '''
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

class CVAE(tf.keras.Model):
    def __init__(self, sensitivity_file, min_model, max_model, norm_pad=0.1, img_rows=32, img_cols=32, channels=1, n_data=100, data_std=1, model_std=1, latent_dim=50):
        super(CVAE, self).__init__()

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.latent_dim = latent_dim
        self.n_data = n_data
        self.data_std = data_std
        self.model_std = model_std
        self.sensitivity = tf.convert_to_tensor(
            np.load(sensitivity_file)
            .reshape(-1,img_cols,img_rows)
            .transpose((0,2,1))
        )
        # self.min_model = min_model
        # self.max_model = max_model
        # self.norm_pad = norm_pad
        self.model_shift = (max_model+min_model)/2
        self.model_scale = 2*(1-norm_pad)/(max_model-min_model)

        self.inference_net = tf.keras.Sequential(
            [
                # 32 x 32 x 1, depth = 1
                tf.keras.layers.InputLayer(input_shape=(img_rows, img_cols, channels)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                # 16 x 16 x 1, depth=32
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                # 8 x 8 x 1, depth=64
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim+n_data,)),
                tf.keras.layers.Dense(units=8*8*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(8, 8, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim+self.n_data))
        return self.decode(eps, apply_tanh=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_tanh=False):
        tanhs = self.generative_net(z)
        if apply_tanh:
            probs = tf.tanh(tanhs)
            return probs
        return tanhs

    def model_to_tanhs(self,model):
        '''
        Rescale model between -1 and 1
        '''
        return (model - self.model_shift)*self.model_scale

    def tanhs_to_model(self,tanhs):
        '''
        Rescale from (-1, 1) to model parameter range
        '''
        return tanhs/self.model_scale + self.model_shift

    def predict(self,model,zero_first_datum=False,subtract_model_mean=False):
        '''
        Predict data, given a model
        if zero_first_datum, shift all data such that first datum is 0
        if subtract_model_mean, remove the model mean before modeling (density contrast)
        NOTE: must convert output tanhs to model (density) before this
        '''
        m64 = tf.cast(model,np.float64)
        m64 = tf.reshape(m64,(-1,self.img_rows,self.img_cols))
        if subtract_model_mean:
            m64 -= tf.expand_dims(tf.expand_dims(tf.reduce_mean(m64,axis=(1,2)),axis=1),axis=2)
        data = tf.linalg.einsum('ijk,ljk...->li...',self.sensitivity,m64)
        if zero_first_datum:
            data -= data[:,0:1,...]
        return data

    def predict_tanh(self,tanhs,zero_first_datum=False,subtract_model_mean=False):
        '''
        Predict data, given an output
        '''
        return self.predict(self.tanhs_to_model(tanhs),
                            zero_first_datum=zero_first_datum,
                            subtract_model_mean=subtract_model_mean
                           )

    def plot_models(self, save2file=False, folder='.', samples=16, latent=None, step=0):
        filename = folder+'/model.png'
        if latent is None:
            latent = np.random.normal(0.0, 1.0, size=[samples, self.latent_dim+self.n_data])
        else:
            filename = folder+"/model_%05d.png" % step
        tanhs = self.decode(latent,apply_tanh=True)
        samples = tanhs.shape[0]
        tanhs = np.reshape(tanhs,(samples, self.img_rows, self.img_cols))
        plot_images(tanhs, save2file=save2file, filename=filename, step=step)

    def plot_data(self, save2file=False, folder='.', samples=16, latent=None, step=0):
        filename = folder+'/gravity.png'
        if latent is None:
            latent = np.random.normal(0.0, 1.0, size=[samples, self.latent_dim+self.n_data])
        else:
            filename = folder+"/gravity_%05d.png" % step
        tanhs = self.decode(latent,apply_tanh=True)
        samples = tanhs.shape[0]
        d_obs = latent[...,self.latent_dim:]
        d_pre = tf.reshape(self.predict_tanh(tanhs),(samples,self.n_data))
        data = np.stack((d_obs,d_pre),axis=-1)
        plot_lines(data, save2file=save2file, filename=filename, step=step)



def plot_images(images, save2file=False, filename='./model.png', step=0):
    plt.figure(figsize=(10,10))
    samples = images.shape[0]
    subplot_rows = np.floor(np.sqrt(samples))
    subplot_cols = np.ceil(samples/subplot_rows)
    for i in range(images.shape[0]):
        plt.subplot(subplot_rows, subplot_cols, i+1)
        image = images[i, ...]
        plt.imshow(image, cmap='viridis', vmin=-1., vmax=1.)
        plt.axis('off')
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.suptitle('Epoch %d'%step)
    if save2file:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()

def plot_lines(data, save2file=False, filename='./gravity.png', step=0):
    plt.figure(figsize=(10,10))
    samples = data.shape[0]
    subplot_rows = np.floor(np.sqrt(samples))
    subplot_cols = np.ceil(samples/subplot_rows)
    for i in range(data.shape[0]):
        plt.subplot(subplot_rows, subplot_cols, i+1)
        data_i = data[i, ...]
        plt.plot(data_i)
        plt.ylim(-1.5,1.5)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.suptitle('Epoch %d'%step)
    if save2file:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

@tf.function
def compute_loss(network, x):
    '''
    total loss function
    '''
    mean_error = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    # L1 norm: tf.losses.mae
    # mean_error = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    beta_vae = 10
    d_true = tf.cast(network.predict_tanh(x), np.float32)
    # Add noise to d_true
    eps = tf.random.normal(shape=d_true.shape)
    d_true += eps*network.data_std
    mean, logvar = network.encode(x)
    z = network.reparameterize(mean, logvar)
    # print(x.shape)
    # print(z.shape)
    # print(d_true.shape)
    zd = tf.concat((z, d_true), -1)
    x_tanh = network.decode(zd, apply_tanh=True)
    d_pre = tf.cast(network.predict_tanh(x_tanh), np.float32)
    data_misfit = mean_error(d_true, d_pre, sample_weight=1/(network.data_std**2))
    # print(x_tanh.shape, x.shape)
    logpx_z = mean_error(tf.reshape(x_tanh, (-1, network.img_rows*network.img_cols)), 
                         tf.reshape(x, (-1, network.img_rows*network.img_cols)), 
                         sample_weight=1/(network.model_std**2))
    # logpx_z = tf.losses.mse(x_tanh, x)
    logpx_z = tf.cast(logpx_z, np.float32)
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    # print(data_misfit.dtype, logpx_z.dtype, logpz.dtype, logqz_x.dtype)
    # print(data_misfit.shape, logpx_z.shape, logpz.shape, logqz_x.shape)
    return -tf.reduce_mean(-data_misfit - logpx_z + beta_vae*logpz - beta_vae*logqz_x)

@tf.function
def compute_reconstruction_loss(network, x):
    '''
    No data misfit
    '''
    mean_error = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    # L1 norm: tf.losses.mae
    # mean_error = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    beta_vae = 1
    mean, logvar = network.encode(x)
    z = network.reparameterize(mean, logvar)
    zd = tf.concat((z, tf.zeros((*mean.shape[:-1],network.n_data))),-1)
    x_tanh = network.decode(zd, apply_tanh=True)
    logpx_z = mean_error(tf.reshape(x_tanh, (-1, network.img_rows*network.img_cols)), 
                         tf.reshape(x, (-1, network.img_rows*network.img_cols)), 
                         sample_weight=1/(network.model_std**2))
    logpx_z = tf.cast(logpx_z, np.float32)
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(-logpx_z + beta_vae*logpz - beta_vae*logqz_x)


@tf.function
def compute_apply_gradients(network, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(network, x)
        # loss = compute_reconstruction_loss(network, x)
    gradients = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, network.trainable_variables))

def compute_losses(network,x):
    '''
    compute each loss separately, for evaluating performance
    '''
    mean_error = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    # mean_error = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    beta_vae = 1
    d_true = tf.cast(network.predict_tanh(x), np.float32)
    mean, logvar = network.encode(x)
    z = network.reparameterize(mean, logvar)
    zd = tf.concat((z, d_true), -1)
    x_tanh = network.decode(zd, apply_tanh=True)
    d_pre = tf.cast(network.predict_tanh(x_tanh), np.float32)
    data_misfit = mean_error(d_true, d_pre, sample_weight=1/(network.data_std**2))
    logpx_z = mean_error(tf.reshape(x_tanh, (-1, network.img_rows*network.img_cols)), 
                  tf.reshape(x, (-1, network.img_rows*network.img_cols)),
                  sample_weight=1/(network.model_std**2))
    logpx_z = tf.cast(logpx_z, np.float32)
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    # print(data_misfit.dtype, logpx_z.dtype, logpz.dtype, logqz_x.dtype)
    # print(data_misfit.shape, logpx_z.shape, logpz.shape, logqz_x.shape)
    return(data_misfit, logpx_z, -beta_vae*logpz + beta_vae*logqz_x)




