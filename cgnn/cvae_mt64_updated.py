'''
    Adapted from
https://www.tensorflow.org/tutorials/generative/cvae
https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py
    with input from
https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py
    Transposed convolution guide:
https://arxiv.org/pdf/1603.07285v1.pdf
'''

import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.losses import Reduction
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.layers import (InputLayer, Dense, Flatten, Reshape,
                                     Conv1D, Conv1DTranspose)
import keras.backend as K

# from .mt1d import forward_1_freq, gradient_Z_1_freq, gradient_Z_con_1_freq
# from cgnn import mt1d_updated
from .mt1d_updated import EM, forward_vec_freq, gradient


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self, sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"

    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time))


class CVAE(Model):
    def __init__(self, depths, min_model, max_model,
                 times=np.arange(0, 15), norm_pad=0.1,
                 channels=1, data_std=1, model_std=1, latent_dim=50,
                 beta_vae=1, model_loss_type='mse', data_loss_type='mse',
                 log_data=True, norm_data=False, data_shift=0, data_scale=1,
                 initializer=tf.keras.initializers.GlorotUniform()
                ):
        super(CVAE, self).__init__()

        self.depths = depths
        self.thicknesses = depths - np.r_[0, depths[:-1]]
        print('thicknesses',self.thicknesses.shape)
        n_model = len(depths)+1
        self.n_model = n_model
        self.channels = channels
        self.latent_dim = latent_dim
        self.times = times
        n_time = len(times)
        self.n_time = n_time
        # In orginal code, it is 2*n_freqs? Do I need to do this for times? Causes a lot of dimensionality issues
        n_data = 2*n_time
        self.n_data = n_data
        self.data_std = data_std
        self.model_std = model_std
        self.beta_vae = beta_vae
        # self.min_model = min_model
        # self.max_model = max_model
        # self.norm_pad = norm_pad
        self.model_shift = (max_model+min_model)/2
        self.model_scale = 2*(1-norm_pad)/(max_model-min_model)
        self.log_data = log_data
        self.norm_data = norm_data
        self.data_shift = data_shift
        self.data_scale = data_scale

        self.simulation = EM(times=self.times,thicknesses = self.thicknesses)
        print(self.simulation)

        if model_loss_type == 'se':
            self.model_mean_error = MeanSquaredError(
                reduction=Reduction.NONE)
            self.model_weights = n_model/(model_std**2)
        if model_loss_type == 'mse':
            self.model_mean_error = MeanSquaredError(
                reduction=Reduction.NONE)
            self.model_weights = 1/(model_std**2)
        if model_loss_type == 'ae':
            self.model_mean_error = MeanAbsoluteError(
                reduction=Reduction.NONE)
            self.model_weights = n_model/model_std
        if model_loss_type == 'mae':
            self.model_mean_error = MeanAbsoluteError(
                reduction=Reduction.NONE)
            self.model_weights = 1/model_std

        if data_loss_type == 'se':
            self.data_mean_error = MeanSquaredError(
                reduction=Reduction.NONE)
            self.data_weights = n_data/(data_std**2)
        if data_loss_type == 'mse':
            self.data_mean_error = MeanSquaredError(
                reduction=Reduction.NONE)
            self.data_weights = 1/(data_std**2)
        if data_loss_type == 'ae':
            self.data_mean_error = MeanAbsoluteError(
                reduction=Reduction.NONE)
            self.data_weights = n_data/data_std
        if data_loss_type == 'mae':
            self.data_mean_error = MeanAbsoluteError(
                reduction=Reduction.NONE)
            self.data_weights = 1/data_std

        # initializer = tf.keras.initializers.HeNormal()
        # initializer = tf.keras.initializers.GlorotUniform()

        self.inference_net = Sequential([
            # 64 x 1, depth = 1
            InputLayer(input_shape=(32, channels)),
            Conv1D(filters=16,
                   kernel_size=3,
                   strides=(2,),
                   activation='relu',
                   kernel_initializer=initializer
                  ),
            # 32, depth=16
            Conv1D(filters=32,
                   kernel_size=3,
                   strides=(2,),
                   activation='relu',
                   kernel_initializer=initializer
                  ),
            # 16, depth=32
            Conv1D(filters=64,
                   kernel_size=3,
                   strides=(2,),
                   activation='relu',
                   kernel_initializer=initializer
                  ),
            # 8, depth=64
            Flatten(),
            # No activation
            Dense(latent_dim + latent_dim,
                  kernel_initializer=initializer
                 )
        ])

        self.generative_net = Sequential([
            InputLayer(input_shape=(latent_dim+n_data,)),
            Dense(units=8*32, activation=tf.nn.relu,
                  kernel_initializer=initializer
                 ),
            Reshape(target_shape=(4, 64)),
            Conv1DTranspose(
                filters=64,
                kernel_size=3,
                strides=(2,),
                padding="SAME",
                activation='relu',
                kernel_initializer=initializer
            ),
            # 16, depth=64
            Conv1DTranspose(
                filters=32,
                kernel_size=3,
                strides=(2,),
                padding="SAME",
                activation='relu',
                kernel_initializer=initializer
            ),
            # 32, depth=32
            Conv1DTranspose(
                filters=16,
                kernel_size=3,
                strides=(2,),
                padding="SAME",
                activation='relu',
                kernel_initializer=initializer
            ),
            # 64, depth=16
            # No activation
            Conv1DTranspose(
                filters=1,
                kernel_size=3,
                strides=(1,),
                padding="SAME",
                kernel_initializer=initializer
            )
        ])

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim+self.n_data))
        return self.decode(eps, apply_tanh=True)

    def invert(self, y, z=None, n=1):
        yshape = y.shape
        if len(yshape) == 1:
            y = tf.reshape(y, (1, -1))
            num_ys = 1
        else:
            num_ys = yshape[-2]
        if z is None:
            zshape0 = y.shape[0]*n
            zshape1 = self.latent_dim
            z = tf.random.normal(shape=(zshape0, zshape1))
        ytile = tf.tile(y, [n, 1])
        yz = tf.concat((z, tf.tile(y, [n, 1])), 1)
        tanhs = self.decode(yz, apply_tanh=True)
        return tanhs

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(
            x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_tanh=False):
        tanhs = self.generative_net(z)
        if apply_tanh:
            probs = tf.tanh(tanhs)
            return probs
#         print(tahns.summary())
        return tanhs

    def model_to_tanhs(self, model):
        '''
        Rescale model between -1 and 1
        '''
        return (model - self.model_shift)*self.model_scale

    def tanhs_to_model(self, tanhs):
        '''
        Rescale from (-1, 1) to model parameter range
        '''
        return tanhs/self.model_scale + self.model_shift

    def data_to_input(self, data):
        if self.log_data:
            d_input = tf.math.log(-data)
        else:
            d_input = data
        if self.norm_data:
            d_input = (d_input - self.data_shift)*self.data_scale
        return d_input

    def input_to_data(self, d_input):
        if self.norm_data:
            data = d_input/self.data_scale + self.data_shift
        else:
            data = d_input
        if self.log_data:
            data = -tf.exp(data)
            # print('log')
        return data

    def data_input_noise(self, d_input, rel_noise):
        if self.norm_data:
            noise = (rel_noise * 
                     # self.data_shift *
                     self.data_scale * 
                     tf.random.normal(shape=d_input.shape)
                    )
        else:
            noise = (rel_noise * 
                     tf.random.normal(shape=d_input.shape)
                    )
        if self.log_data:
            return d_input + tf.math.log(abs(1 + noise))
        else:
            # return d_input + noise
            return d_input*(1 + noise)
        
    def forward_np(self, x, thicknesses, times):
        '''
        Use numpy for forward modeling, return tensorflow object
        '''
        nb = x.shape[0]
        nc = x.shape[1]
        nt = len(times)
        # print('nc', nc)
        xn = tf.reshape(x, (-1, nc)).numpy()
        Zss = np.zeros((nb, nc))
        ic = 0
        for ic, c in enumerate(xn):
            # print('EM', EM.forward_vec_freq(c,thicknesses,times))
            Zss[ic, :] = forward_vec_freq(self.simulation, c)
        # Rs = np.real(Zss)
        # Is = np.imag(Zss)
        # data_array = np.c_[Rs, Is]
        data_array = Zss
        # print('Zss', Zss)
        # print(data_array)
        return tf.cast(data_array, tf.float32)
    
    def gradient_np(self, x, y, dy, thicknesses, times):
        '''
        Use numpy for gradient, return tensorflow object
        '''
        nb = x.shape[0]
        nc = x.shape[1]
        nt = len(times)
        # print(type(x), type(y), type(dy))
        dd = np.reshape(dy, (-1, 2*nt))#.numpy()
        # nb = dd.shape[0]
        # xn = x#.numpy()
        xn = tf.reshape(x, (-1, nc)).numpy()
        xn = xn[:,:32]
        # print('xn',xn.shape)
        # vJp = np.zeros((nb, nc))
        vJp = np.zeros(x.shape)

        # Zss = y.numpy()[:, :nf, :] + 1j*y.numpy()[:, nf:, :]
        Zss = y[:, :32]
        # print('Zss',Zss.shape)
        for ib, (Zs, c) in enumerate(zip(Zss, xn)):
            # print('Zs',Zs.shape)
            # print('c',c.shape)
            vJp[ib,:32,0] = gradient(self.simulation, c, Zs)
        # for ib, Zs in enumerate(Zss):
    #         # Z = forward_vec_freq(c, thicknesses, frequencies)
    #         dZdZ1 = gradient_Z(Zs, c, thicknesses, times)
    #         dZidconi = gradient_Z_con(Zs, c, thicknesses, times)
    #         dZ1dZi = np.cumprod(dZdZ1, axis=1)
    #         dZdcon = dZidconi
    #         dZdcon[:, 1:] *= dZ1dZi
    #         # return dZdcon
    # #         print(dZdcon.T.shape)
    # #         print(dd[ib,nf:].shape)
    # #         print(vJp[ib,:30,0].shape)
    #         vJp[ib, :30, 0] = (np.dot(np.real(dZdcon.T), dd[ib, :nt]) +
    #                         np.dot(np.imag(dZdcon.T), dd[ib, nt:]))



    #       for ifreq, (frequency, Z) in enumerate(zip(frequencies, Zs)):
    #           #print(Z.shape, c.shape, thicknesses.shape)
    #           dZdZ1 = gradient_Z_1_freq(Z, c, thicknesses, frequency)
    #           dZidconi = gradient_Z_con_1_freq(Z, c, thicknesses, frequency)
    #           dZ1dZi = np.cumprod(dZdZ1)
    #           dZdcon = dZidconi
    #           dZdcon[1:] *= dZ1dZi
    #           # print(vJp.shape, dZdcon.shape, dd.shape)
    #           vJp[ib, :, 0] += (np.real(dZdcon)*dd[ib, ifreq] +
    #                             np.imag(dZdcon)*dd[ib, nf + ifreq])

        # create a tensorflow array for output
        return tf.cast(vJp, tf.float32)
    
    @tf.custom_gradient
    def predict_data(self, model):
        '''
        Accepts conductivity model
        Outputs data, varying fastest in frequency and slowest in real/imag
        Returns data and gradient as a tuple, as per
        https://www.tensorflow.org/api_docs/python/tf/custom_gradient
        https://www.tensorflow.org/guide/advanced_autodiff#custom_gradients
        https://stackoverflow.com/questions/56657993/how-to-create-a-keras-layer-with-a-custom-gradient-in-tf2-0
        https://stackoverflow.com/questions/58223640/custom-activation-with-custom-gradient-does-not-work
        '''
        ys = tf.numpy_function(
            self.forward_np, [model, self.thicknesses, self.times],
            model.dtype)
#         print(self.frequencies.shape)
#         print(self.thicknesses.shape)
#         print('model', model[-1])
        print('ys', ys.shape)
        # Why?
        ys_test = ys[..., 0]
        print('ys_test', ys_test.shape)
        def tdem_grad(ddata):
            '''
            Return J^T ddata
            '''
            # gradient(self.simulation,model,ddata)
            return tf.numpy_function(
                self.gradient_np, [model, ys, ddata, self.thicknesses, self.times],
                model.dtype)
        return ys, tdem_grad

    # def predict_log(self, logs):
    #     '''
    #     Predict data, given log conductivities
    #     '''
    #     return self.predict_data(tf.exp(logs))

    def predict_tanh(self, tanhs):
        '''
        Predict data, given an output
        '''
        print('tanhs', tanhs.shape)
        return self.predict_data(tf.exp(self.tanhs_to_model(tanhs)))

    def plot_models(self, save2file=False, folder='.', samples=16,
                    latent=None, step=None):
        if latent is None:
            latent = np.random.normal(
                0.0, 1.0, size=[samples, self.latent_dim+self.n_data])
        if step is None:
            filename = folder+"/model.png"
        else:
            filename = folder+"/model_%05d.png" % step
        tanhs = self.decode(latent, apply_tanh=True)
        samples = tanhs.shape[0]
        print(tanhs.shape)
        tanhs = np.reshape(tanhs, (samples, self.n_model))
        plot_logs(np.exp(self.tanhs_to_model(tanhs)), save2file=True,
                  filename=filename, step=16, depths=self.depths)

    def overlay_models(self, save2file=False, folder='.', samples=16,
                       latent=None, step=None):
        if latent is None:
            latent = np.random.normal(
                0.0, 1.0, size=[samples, self.latent_dim+self.n_data])
        if step is None:
            filename = folder+"/model.png"
        else:
            filename = folder+"/model_%05d.png" % step
        tanhs = self.decode(latent, apply_tanh=True)
        samples = tanhs.shape[0]
        tanhs = np.reshape(tanhs, (samples, self.n_model))
        plot_log(np.exp(self.tanhs_to_model(tanhs)), save2file=save2file,
                 filename=filename, step=step, depths=self.depths)

    def plot_data(self, save2file=False, folder='.', samples=16,
                  latent=None, step=None, ylims=(1e-22, 1e-10)):
        if latent is None:
            latent = np.random.normal(
                0.0, 1.0, size=[samples, self.latent_dim+self.n_data])
        if step is None:
            filename = folder+"/data.png"
        else:
            filename = folder+"/data_%05d.png" % step
        # print('latent',latent)
        tanhs = self.decode(latent, apply_tanh=True)
        samples = tanhs.shape[0]
        d_obs = (latent[..., self.latent_dim+16:])
        print('d_obs',d_obs)
        d_pre = tf.reshape(self.predict_tanh(tanhs), (samples, self.n_data))
        d_pre = d_pre[...,16:]
        print('d_pre',d_pre)
        data = np.stack((d_obs[..., :self.n_time],
                        d_pre[..., :self.n_time:]), axis=-1)
        # data = d_pre[...,:self.n_time]
        print('data',data)
        # print('data min', data.min)
        plot_lines(data, save2file=True, filename=filename, step=16,
                   ylims=ylims, times=self.times[1:],
                   legend_labels=['Obs','Pre'],x_label='Times (s)', y_label='dB/dt')

    def plot_residuals(self, save2file=False, folder='.', samples=16,
                       latent=None, step=None, ylims=(1e-25, 1e-10),
                       weighted=True):
        '''
        Plot data residuals;
        I can't even see the data difference
        '''
        if latent is None:
            latent = np.random.normal(
                0.0, 1.0, size=[samples, self.latent_dim+self.n_data])
        if step is None:
            filename = folder+"/residual.png"
        else:
            filename = folder+"/residual_%05d.png" % step
        tanhs = self.decode(latent, apply_tanh=True)
        samples = tanhs.shape[0]
        d_obs = -(latent[..., self.latent_dim:])
        d_pre = -tf.reshape(self.predict_tanh(tanhs), (samples, self.n_data))
        # print(self.data_std.flatten())
        if weighted:
            d_res = -tf.abs(d_obs - d_pre)*self.data_std.flatten()[None, :]
        else:
            d_res = -tf.abs(d_obs - d_pre)
        print('d_res', d_res.shape)
        # Why? I don't need to do this - for complex values
        print('d_res',d_res)
        data = d_res[...,1:self.n_time]
        print(data)
        plot_lines(data, save2file=save2file, filename=filename, step=step,
                   ylims=ylims, times=self.times[1:],x_label='Times (s)', y_label='Conductivity')




# Use numpy's complex numbers, because TF seems to be very slow
# https://stackoverflow.com/a/63583413/14134052

def plot_log(ax, log, depths=None):
    if depths is None:
        ax.semilogx(log, np.arange(len(log)))
    else:
        depth_centers = (depths[1:] + depths[:-1])/2
        plot_depths = np.r_[
            depth_centers[0] - (depth_centers[1] - depth_centers[0]),
            depth_centers,
            depth_centers[-1] + depth_centers[-1] - depth_centers[-2]
        ]
        ax.semilogx(log, plot_depths)
    ax.invert_yaxis()


def plot_logs(logs, save2file=False, filename='./model.png', step=None,
              xlims=(2e-2, 2e3), depths=None,
              x_label='Conductivity, S/m',
              y_label='Depth, m'
             ):
    # matplotlib.rc('font', size=8)
    fig = plt.figure(figsize=(14, 10))
    samples = logs.shape[0]
    subplot_rows = int(np.floor(np.sqrt(samples)))
    subplot_cols = int(np.ceil(samples/subplot_rows))
    for i in range(logs.shape[0]):
        ax = plt.subplot(subplot_rows, subplot_cols, i+1)
        log = logs[i, ...]
        ax.set_title("Sounding %d" % int(i+1), fontsize = 16)
        if depths is None:
            plt.semilogx(log, np.arange(len(log)))
        else:
            depth_centers = (depths[1:] + depths[:-1])/2
            plot_depths = np.r_[
                depth_centers[0] - (depth_centers[1] - depth_centers[0]),
                depth_centers,
                depth_centers[-1] + depth_centers[-1] - depth_centers[-2]
            ]
            if len(log) != len(plot_depths):
                log = np.delete(log, [-2,-1])
            plt.semilogx(log, plot_depths)
        plt.gca().invert_yaxis()
        plt.xlim(*xlims)
        if i%subplot_cols > 0:
            ax.axes.yaxis.set_ticks([])
        if i < (subplot_rows - 1)*subplot_cols:
            ax.axes.xaxis.set_ticks([])
    fig.text(0.5, 0.06, x_label, ha='center', va='center', size=20)
    fig.text(0.03, 0.5, y_label, ha='center', va='center',
             rotation='vertical', size=20)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if step is not None:
        plt.suptitle('Generated Validation Conductivty Logs')
    if save2file:
        plt.savefig(filename)
        plt.draw()
        plt.clf()
        plt.close('all')
    else:
        plt.show()


def plot_complex(data, **kwargs):    
    '''
    Plot complex data as real and imaginary lines
    '''
    assert data.shape[1]%2==0, 'Unequal number of real and imaginary values'
    nf = data.shape[1]//2
    stacked_data = np.stack((data[..., :nf], data[..., nf:]), axis=-1)
    plot_lines(stacked_data, **kwargs)


def plot_lines(data, save2file=False, filename='./data.png', step=None,
               ylims=(0, 1000), times=None,
               x_label='Conductivity, (S/m)',
               y_label='Depth',legend_labels=None):
    # matplotlib.rc('font', size=14)
    fig = plt.figure(figsize=(14, 10))
    samples = data.shape[0]
    subplot_rows = int(np.floor(np.sqrt(samples)))
    subplot_cols = int(np.ceil(samples/subplot_rows))
    for i in range(data.shape[0]):
        ax = plt.subplot(subplot_rows, subplot_cols, i+1)
        data_i = -data[i, ...]
        ax.set_title('Sounding %d' %int(i+1), fontsize = 16)
        if times is None:
            ax.semilogy(data_i)
        else:
            ax.loglog(times, data_i[:15])
        plt.ylim(*ylims)
        if i%subplot_cols > 0:
            ax.axes.yaxis.set_ticks([])
        if i < (subplot_rows - 1)*subplot_cols:
            ax.axes.xaxis.set_ticks([])
    fig.text(0.5, 0.06, x_label, ha='center', va='center', size=20)
    fig.text(0.03, 0.5, y_label, ha='center', va='center',
             rotation='vertical', size=20)
    if legend_labels is not None:
        plt.legend(legend_labels, bbox_to_anchor=(1.02, 0.1), loc='upper left', borderaxespad=0)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if step is not None:
        plt.suptitle("Time Decay Curves")
    if save2file:
        plt.savefig(filename)
        plt.draw()
        plt.clf()
        plt.close('all')
    else:
        plt.show()



def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


@tf.function
def compute_loss(network, xy, rel_noise=0):
    '''
    total loss function
    '''
    x = xy[0]
    d_input = tf.cast(xy[1], np.float32)
    tf.print('d_input:', d_input)
    d_true = network.input_to_data(d_input)
    if rel_noise > 0:
        d_input = network.data_input_noise(d_input, rel_noise)
    # d_true = -tf.exp(d_input)
    # d_true = tf.cast(network.predict_tanh(x), np.float32)
    # Add noise to d_true
    # eps = tf.random.normal(shape=d_true.shape)
    # d_true += eps*network.data_std
    # d_true_log = tf.math.log(-d_true)
    mean, logvar = network.encode(x)
    z = network.reparameterize(mean, logvar)
    zd = tf.concat((z, d_input), -1)
    print('zd:', zd)
    x_tanh = network.decode(zd, apply_tanh=True)
    print('x_tanh', x_tanh)
    d_pre = tf.cast(network.predict_tanh(x_tanh), np.float32)
    tf.print('d_true',d_true)
    tf.print('d_pre', d_pre)
    # d_pre = tf.math.log(-tf.cast(network.predict_tanh(x_tanh), np.float32))
    # print(d_true.shape, d_pre.shape, network.data_weights.shape)
    dme = network.data_mean_error(tf.transpose(d_true),
                                  tf.transpose(tf.reshape(d_pre, (-1, network.n_data))),
                                  sample_weight=network.data_weights)
    data_misfit = tf.reduce_mean(dme)
    tf.print('dme',dme)
    # data_misfit = tf.reduce_mean(
    #     network.data_mean_error(tf.transpose(d_true),
    #                             tf.transpose(d_pre),
    #                             sample_weight=network.data_weights))
#     x_tanh2 = torch.tensor()
    # dim = x.shape[0]
    # x_tanh1 = tf.slice(x_tanh, [0,0,0], [dim,32,1])
    logpx_z = tf.reduce_mean(
        network.model_mean_error(tf.transpose(tf.reshape(x_tanh, (-1, network.n_model))),
                                 tf.transpose(tf.reshape(x, (-1, network.n_model))),
                                 sample_weight=network.model_weights))
    # print(logpx_z)
    # logpx_z = tf.losses.mse(x_tanh, x)
    logpx_z = tf.cast(logpx_z, np.float32)
    logpz = tf.reduce_mean(log_normal_pdf(z, 0., 0.))
    logqz_x = tf.reduce_mean(log_normal_pdf(z, mean, logvar))
    # print(data_misfit.dtype, logpx_z.dtype, logpz.dtype, logqz_x.dtype)
    # print(data_misfit.shape, logpx_z.shape, logpz.shape, logqz_x.shape)
    # loss = -tf.reduce_mean(-data_misfit - logpx_z +
    #                        network.beta_vae*(logpz - logqz_x))
    # terms = (tf.reduce_mean(data_misfit),
    #          tf.reduce_mean(logpx_z),
    #          tf.reduce_mean(logqz_x - logpz))
    terms = (data_misfit, logpx_z, -network.beta_vae*(logpz - logqz_x))
    loss = data_misfit + logpx_z - network.beta_vae*(logpz - logqz_x)
    # print('loss',K.eval(loss))
    # print('data_misfit',K.eval(data_misfit))
    # print('logpx_z',K.eval(logpx_z))
    # print('beta_vae', network.beta_vae)
    # print('logqz_x',K.eval(logqz_x))
    # print('logpz',K.eval(logpz))
    return (loss, terms)


@tf.function
def compute_reconstruction_loss(network, xy, rel_noise=0):
    '''
    No data misfit
    rel_noise is noise relative to network.data_std
    '''
    x = xy[0]
    d_input = tf.cast(xy[1], np.float32)
    # d_true = -tf.exp(d_input)
    if rel_noise > 0:
        # d_input += tf.math.log(abs(1 + rel_noise*network.data_std.flatten()[None, :]*tf.random.normal(shape=d_input.shape)))
        # d_input += rel_noise*network.data_std.flatten()[None, :]*tf.random.normal(shape=y.shape)
        d_input = network.data_input_noise(d_input, rel_noise)
    # d_true = tf.math.log(-tf.cast(network.predict_tanh(x), np.float32))
    # d_true = tf.cast(network.predict_tanh(x), np.float32)
    mean, logvar = network.encode(x)
    z = network.reparameterize(mean, logvar)
    zd = tf.concat((z, d_input), -1)
    x_tanh = network.decode(zd, apply_tanh=True)
    print(network.n_model)
    print(x_tanh.shape)
    print(x.shape)

    dim = x.shape[0]
    x_tanh1 = tf.slice(x_tanh, [0,0,0], [dim,30,1])
    print('network n_model:',network.n_model-2)
    print('x_tanh:',x_tanh1.shape)
    print(x_tanh[0])
    print(x_tanh1[0])
    print('x:',x.shape)
    print('x_tanh tensor:',tf.transpose(tf.reshape(x_tanh1, (-1, network.n_model-2))))
    print('x tensor:',tf.transpose(tf.reshape(x, (-1, network.n_model-2))))
    logpx_z = tf.reduce_mean(
        network.model_mean_error(tf.reshape(x_tanh1, (-1, network.n_model)),
                                 tf.reshape(x, (-1, network.n_model)),
        # sample_weight=(network.n_model)/(network.model_std**2))
        sample_weight=network.model_weights))
    logpx_z = tf.cast(logpx_z, np.float32)
    logpz = tf.reduce_mean(log_normal_pdf(z, 0., 0.))
    logqz_x = tf.reduce_mean(log_normal_pdf(z, mean, logvar))
    loss =  logpx_z - network.beta_vae*(logpz - logqz_x)
    terms = (logpx_z, network.beta_vae*(logqz_x - logpz))
    return (loss, terms)


@tf.function
def compute_apply_gradients(network, xy, optimizer, use_data_misfit=True, rel_noise=0):
    with tf.GradientTape() as tape:
        if use_data_misfit:
            print('loss')
            print('network',network)
            print('xy',xy)
            loss, terms = compute_loss(network, xy, rel_noise=rel_noise)
        else:
            print('reconstrauction')
            loss, terms = compute_reconstruction_loss(network, xy, rel_noise=rel_noise)
    gradients = tape.gradient(loss, network.trainable_variables)
    # optimizer.apply_gradients(zip(gradients, network.trainable_variables))
    return (loss, terms)


def compute_losses(network, xy):
    '''
    compute each loss separately, for evaluating performance
    '''
    x = xy[0]
    d_input = tf.cast(xy[1], np.float32)
    d_true = network.input_to_data(d_input)
    # d_true = -tf.exp(d_input)
    # d_true = tf.cast(network.predict_tanh(x), np.float32)
    mean, logvar = network.encode(x)
    z = network.reparameterize(mean, logvar)
    zd = tf.concat((z, d_input), -1)
    x_tanh = network.decode(zd, apply_tanh=True)
    d_pre = tf.cast(network.predict_tanh(x_tanh), np.float32)
    # d_pre = tf.math.log(-tf.cast(network.predict_tanh(x_tanh), np.float32))
    # data_misfit = network.data_mean_error(
    #     d_true, d_pre, sample_weight=network.n_data/(network.data_std**2))
    data_misfit = network.data_mean_error(tf.transpose(d_true), tf.transpose(d_pre))
    logpx_z = network.model_mean_error(
        tf.reshape(x_tanh, (-1, network.n_model)),
        tf.reshape(x, (-1, network.n_model))
        # sample_weight=(network.n_model)/(network.model_std**2))
        # sample_weight=network.model_weights)
    )
    logpx_z = tf.cast(logpx_z, np.float32)
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    # print(data_misfit.dtype, logpx_z.dtype, logpz.dtype, logqz_x.dtype)
    # print(data_misfit.shape, logpx_z.shape, logpz.shape, logqz_x.shape)
    return(data_misfit, logpx_z, -logpz + logqz_x)


def compute_inversion_loss(network, xy, num_inversions):
    '''
    Invert data, compute loss
    '''
    x = xy[0]
    y = tf.cast(xy[1], np.float32)
    d_true = network.input_to_data(y)
    tanhs = network.invert(y, n=num_inversions)
    d_pre = tf.cast(network.predict_tanh(tanhs), np.float32)
    dme = network.data_mean_error(tf.transpose(d_true),
                                  tf.transpose(tf.reshape(d_pre, (-1, network.n_data))),
                                  sample_weight=network.data_weights)
    data_misfit = tf.reduce_mean(dme)
    return data_misfit


@tf.function
def decoder_reconstruction_loss(network, xy, rel_noise=0):
    '''
    No data misfit
    No encoding; just data to model
    rel_noise is noise relative to network.data_std
    '''
    x = xy[0]
    y = tf.cast(xy[1], np.float32)
    if rel_noise > 0:
        y += tf.math.log(abs(1 + rel_noise*network.data_std.flatten()[None, :]*tf.random.normal(shape=y.shape)))
        # y += rel_noise*network.data_std.flatten()[None, :]*tf.random.normal(shape=y.shape)
    z = tf.random.normal(shape=(y.shape[0], network.latent_dim))
    # d_true = tf.cast(network.predict_tanh(x), np.float32)
    # mean, logvar = network.encode(x)
    # z = network.reparameterize(mean, logvar)
    zd = tf.concat((z, y), -1)
    x_tanh = network.decode(zd, apply_tanh=True)
    logpx_z = tf.reduce_mean(network.model_mean_error(
        tf.reshape(x_tanh, (-1, network.n_model)),
        tf.reshape(x, (-1, network.n_model)),
        sample_weight=network.model_weights))
    logpx_z = tf.cast(logpx_z, np.float32)
    # logpz = tf.reduce_mean(log_normal_pdf(z, 0., 0.))
    # logqz_x = tf.reduce_mean(log_normal_pdf(z, mean, logvar))
    loss = logpx_z
    terms = (logpx_z, )
    return (loss, terms)


@tf.function
def decoder_apply_gradients(network, xy, optimizer, rel_noise=0):
    '''
    Train the decoder only to map from data to model
    '''
    with tf.GradientTape() as tape:
        loss, terms = decoder_reconstruction_loss(network, xy, rel_noise=rel_noise)
    gradients = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, network.trainable_variables))
    return (loss, terms)