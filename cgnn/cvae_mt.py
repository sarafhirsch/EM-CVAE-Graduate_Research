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

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.losses import Reduction
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.layers import (InputLayer, Dense, Flatten, Reshape,
                                     Conv1D, Conv1DTranspose)

# from .mt1d import forward_1_freq, gradient_Z_1_freq, gradient_Z_con_1_freq
from .mt1d import forward_vec_freq, gradient_Z, gradient_Z_con


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
                 frequencies=np.logspace(-4, 4, num=8), norm_pad=0.1,
                 channels=1, data_std=1, model_std=1, latent_dim=50,
                 beta_vae=1, model_loss_type='mse', data_loss_type='mse'):
        super(CVAE, self).__init__()

        self.depths = depths
        n_model = len(depths) + 1
        self.n_model = n_model
        self.channels = channels
        self.latent_dim = latent_dim
        self.frequencies = frequencies
        n_freqs = len(frequencies)
        self.n_freqs = n_freqs 
        n_data = 2*n_freqs
        self.n_data = n_data
        self.data_std = data_std
        self.model_std = model_std
        self.beta_vae = beta_vae
        # self.min_model = min_model
        # self.max_model = max_model
        # self.norm_pad = norm_pad
        self.model_shift = (max_model+min_model)/2
        self.model_scale = 2*(1-norm_pad)/(max_model-min_model)

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

        self.inference_net = Sequential([
            # 60 x 1, depth = 1
            InputLayer(input_shape=(n_model, channels)),
            Conv1D(filters=16,
                   kernel_size=5,
                   strides=(2,),
                   activation='relu'),
            # 30, depth=16
            Conv1D(filters=32,
                   kernel_size=5,
                   strides=(2,),
                   activation='relu'),
            # 15, depth=32
            Conv1D(filters=64,
                   kernel_size=5,
                   strides=(2,),
                   activation='relu'),
            # 8, depth=64
            Flatten(),
            # No activation
            Dense(latent_dim + latent_dim),
        ])

        self.generative_net = Sequential([
            InputLayer(input_shape=(latent_dim+n_data,)),
            Dense(units=15*64, activation=tf.nn.relu),
            Reshape(target_shape=(15, 64)),
            Conv1DTranspose(
                filters=32,
                kernel_size=5,
                strides=(2,),
                padding="SAME",
                activation='relu'),
            # 30, depth=32
            Conv1DTranspose(
                filters=16,
                kernel_size=5,
                strides=(2,),
                padding="SAME",
                activation='relu'),
            # 60, depth=16
            # No activation
            Conv1DTranspose(
                filters=1, kernel_size=5, strides=(1,), padding="SAME"),
        ])

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim+self.n_data))
        return self.decode(eps, apply_tanh=True)

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
            forward_np, [model, self.depths, self.frequencies],
            model.dtype)
        def mt_grad(ddata):
            '''
            Return J^T ddata
            '''
            return tf.numpy_function(
                gradient_np, [model, ys, ddata, self.depths, self.frequencies],
                model.dtype)
        return ys[..., 0], mt_grad

    @tf.custom_gradient
    def old_predict_data(self, model):
        '''
        Accepts conductivity model
        Outputs data, varying fastest in frequency and slowest in real/imag
        Returns data and gradient as a tuple, as per
        https://www.tensorflow.org/api_docs/python/tf/custom_gradient
        https://www.tensorflow.org/guide/advanced_autodiff#custom_gradients
        https://stackoverflow.com/questions/56657993/how-to-create-a-keras-layer-with-a-custom-gradient-in-tf2-0
        https://stackoverflow.com/questions/58223640/custom-activation-with-custom-gradient-does-not-work
        '''
        depths = self.depths
        frequencies = self.frequencies
        # nc = len(depths) + 1
        # nf = len(frequencies)
        nc = self.n_model
        nf = self.n_freqs
        # may need to reshape model, or extract values or loop or some such.
        con_array = tf.reshape(model, (-1, nc))#.numpy()
        # con_array = tf.complex(con_array, 0*con_array)
        #data_array = np.empty((con_array.shape[0], 2*nf))
        nb = con_array.shape[0]
        Zss = 1j*np.zeros((nb, nf, nc))
        ic = 0
        # for ic, c in enumerate(con_array):
        # for c in con_array:
        for ic in range(nb):
            #Zs = forward(c, depths, frequencies)
            for i_freq, frequency in enumerate(frequencies):
                # Zss[ic, i_freq, :] = forward_1_freq(c, depths, frequency)
                Zss[ic, i_freq, :] = tf.numpy_function(
                    forward_np, [con_array[ic], depths, frequency], Zss.dtype)
            # ic += 1

        Rs = np.real(Zss[:, :, 0])
        Is = np.imag(Zss[:, :, 0])
        data_array = np.c_[Rs, Is]
        data = tf.convert_to_tensor(data_array)

        #for frequency in frequencies:
            #Z = forward_1_freq(con_array, depths, frequency)
        def mt_grad(ddata):
            '''
            Return J^T ddata
            '''
            dd = tf.reshape(ddata, (-1, 2*nf))#.numpy()
            # nb = dd.shape[0]
            vJp = np.zeros((nb, nc))

            #Zss = data_array[:, :nf] + 1j*data_array[:, nf:]
            # for ib, (Zs, c) in enumerate(zip(Zss, con_array)):
            for ib, Zs in enumerate(Zss):
                for ifreq, (frequency, Z) in enumerate(zip(frequencies, Zs)):
                    print(Z.shape, c.shape, depths.shape)
                    dZdZ1 = gradient_Z_1_freq(Z, con_array[ib], depths,
                                              frequency)
                    dZidconi = gradient_Z_con_1_freq(Z, con_array[ib], depths,
                                                     frequency)
                    dZ1dZi = np.cumprod(dZdZ1)
                    dZdcon = dZidconi
                    dZdcon[1:] *= dZ1dZi
                    # print(vJp.shape, dZdcon.shape, dd.shape)
                    vJp[ib, :] += (np.real(dZdcon)*dd[ib, ifreq] + 
                                   np.imag(dZdcon)*dd[ib, nf + ifreq])
            # create a tensorflow array for output
            return tf.convert_to_tensor(vJp)
        return data, mt_grad

    # def predict_log(self, logs):
    #     '''
    #     Predict data, given log conductivities
    #     '''
    #     return self.predict_data(tf.exp(logs))

    def predict_tanh(self, tanhs):
        '''
        Predict data, given an output
        '''
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
        tanhs = np.reshape(tanhs, (samples, self.n_model))
        plot_logs(np.exp(self.tanhs_to_model(tanhs)), save2file=save2file,
                  filename=filename, step=step, depths=self.depths)

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
                  latent=None, step=None, ylims=(1e-3, 0.6)):
        if latent is None:
            latent = np.random.normal(
                0.0, 1.0, size=[samples, self.latent_dim+self.n_data])
        if step is None:
            filename = folder+"/data.png"
        else:
            filename = folder+"/data_%05d.png" % step
        tanhs = self.decode(latent, apply_tanh=True)
        samples = tanhs.shape[0]
        d_obs = -tf.exp(latent[..., self.latent_dim:])
        d_pre = tf.reshape(self.predict_tanh(tanhs), (samples, self.n_data))
        data = np.stack((d_obs[..., :self.n_freqs],
                         d_obs[..., self.n_freqs:],
                         d_pre[..., :self.n_freqs],
                         d_pre[..., self.n_freqs:]), axis=-1)
        plot_lines(data, save2file=save2file, filename=filename, step=step,
                   ylims=ylims, frequencies=self.frequencies,
                   legend_labels=['obs real', 'obs imaginary', 'pre real', 
                                  'pre imaginary'])

    def plot_residuals(self, save2file=False, folder='.', samples=16,
                       latent=None, step=None, ylims=(1e-6, 1e-1),
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
        d_obs = -tf.exp(latent[..., self.latent_dim:])
        d_pre = tf.reshape(self.predict_tanh(tanhs), (samples, self.n_data))
        if weighted:
            d_res = -tf.abs(d_obs - d_pre)*self.data_std.flatten()[None, :]
        else:
            d_res = -tf.abs(d_obs - d_pre)
        data = np.stack((d_res[..., :self.n_freqs],
                         d_res[..., self.n_freqs:]), axis=-1)
        plot_lines(data, save2file=save2file, filename=filename, step=step,
                   ylims=ylims, frequencies=self.frequencies,
                   legend_labels=['real residual', 'imaginary residual'])




# Use numpy's complex numbers, because TF seems to be very slow
# https://stackoverflow.com/a/63583413/14134052
def forward_np(x, depths, frequencies):
    '''
    Use numpy for forward modeling, return tensorflow object
    '''
    nb = x.shape[0]
    nc = x.shape[1]
    nf = len(frequencies)
    xn = tf.reshape(x, (-1, nc)).numpy()
    Zss = 1j*np.zeros((nb, nf, nc))

    ic = 0
    for ic, c in enumerate(xn):
        Zss[ic, :, :] = forward_vec_freq(c, depths, frequencies)
        # for i_freq, frequency in enumerate(frequencies):
        #     Zss[ic, i_freq, :] = forward_1_freq(c, depths, frequency)

    Rs = np.real(Zss)
    Is = np.imag(Zss)
    # data_array = np.c_[Rs, Is]
    data_array = np.concatenate((Rs, Is), axis=-2)
    return tf.cast(data_array, tf.float32)


def gradient_np(x, y, dy, depths, frequencies):
    '''
    Use numpy for gradient, return tensorflow object
    '''
    nb = x.shape[0]
    nc = x.shape[1]
    nf = len(frequencies)
    # print(type(x), type(y), type(dy))
    dd = np.reshape(dy, (-1, 2*nf))#.numpy()
    # nb = dd.shape[0]
    # xn = x#.numpy()
    xn = tf.reshape(x, (-1, nc)).numpy()
    # vJp = np.zeros((nb, nc))
    vJp = np.zeros(x.shape)

    # Zss = y.numpy()[:, :nf, :] + 1j*y.numpy()[:, nf:, :]
    Zss = y[:, :nf, :] + 1j*y[:, nf:, :]

    for ib, (Zs, c) in enumerate(zip(Zss, xn)):
    # for ib, Zs in enumerate(Zss):


        # Z = forward_vec_freq(c, depths, frequencies)
        dZdZ1 = gradient_Z(Zs, c, depths, frequencies)
        dZidconi = gradient_Z_con(Zs, c, depths, frequencies)
        dZ1dZi = np.cumprod(dZdZ1, axis=1)
        dZdcon = dZidconi
        dZdcon[:, 1:] *= dZ1dZi
        # return dZdcon
        vJp[ib, :, 0] = (np.dot(np.real(dZdcon.T), dd[ib, :nf]) +
                         np.dot(np.imag(dZdcon.T), dd[ib, nf:]))



#       for ifreq, (frequency, Z) in enumerate(zip(frequencies, Zs)):
#           #print(Z.shape, c.shape, depths.shape)
#           dZdZ1 = gradient_Z_1_freq(Z, c, depths, frequency)
#           dZidconi = gradient_Z_con_1_freq(Z, c, depths, frequency)
#           dZ1dZi = np.cumprod(dZdZ1)
#           dZdcon = dZidconi
#           dZdcon[1:] *= dZ1dZi
#           # print(vJp.shape, dZdcon.shape, dd.shape)
#           vJp[ib, :, 0] += (np.real(dZdcon)*dd[ib, ifreq] +
#                             np.imag(dZdcon)*dd[ib, nf + ifreq])

    # create a tensorflow array for output
    return tf.cast(vJp, tf.float32)


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
              xlims=(1e-4, 5), depths=None,
              x_label='Conductivity, S/m',
              y_label='Depth, m'
             ):
    # matplotlib.rc('font', size=8)
    fig = plt.figure(figsize=(14, 10))
    samples = logs.shape[0]
    subplot_rows = np.int(np.floor(np.sqrt(samples)))
    subplot_cols = np.int(np.ceil(samples/subplot_rows))
    for i in range(logs.shape[0]):
        ax = plt.subplot(subplot_rows, subplot_cols, i+1)
        log = logs[i, ...]
        if depths is None:
            plt.semilogx(log, np.arange(len(log)))
        else:
            depth_centers = (depths[1:] + depths[:-1])/2
            plot_depths = np.r_[
                depth_centers[0] - (depth_centers[1] - depth_centers[0]),
                depth_centers,
                depth_centers[-1] + depth_centers[-1] - depth_centers[-2]
            ]
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
        plt.suptitle('Epoch %d' % step)
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
               ylims=(1e-4, 1), frequencies=None,
               x_label='frequency, Hz',
               y_label='$Z_{xy}$',
               legend_labels=['real', 'imaginary']):
    # matplotlib.rc('font', size=14)
    fig = plt.figure(figsize=(14, 10))
    samples = data.shape[0]
    subplot_rows = np.int(np.floor(np.sqrt(samples)))
    subplot_cols = np.int(np.ceil(samples/subplot_rows))
    for i in range(data.shape[0]):
        ax = plt.subplot(subplot_rows, subplot_cols, i+1)
        data_i = -data[i, ...]
        if frequencies is None:
            ax.semilogy(data_i)
        else:
            ax.loglog(frequencies, data_i)
        plt.ylim(*ylims)
        if i%subplot_cols > 0:
            ax.axes.yaxis.set_ticks([])
        if i < (subplot_rows - 1)*subplot_cols:
            ax.axes.xaxis.set_ticks([])
    fig.text(0.5, 0.06, x_label, ha='center', va='center', size=20)
    fig.text(0.03, 0.5, y_label, ha='center', va='center',
             rotation='vertical', size=20)
    if legend_labels is not None:
        plt.legend(legend_labels)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if step is not None:
        plt.suptitle('Epoch %d' % step)
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
def compute_loss(network, xy):
    '''
    total loss function
    '''
    x = xy[0]
    d_true_log = tf.cast(xy[1], np.float32)
    d_true = -tf.exp(d_true_log)
    # d_true = tf.cast(network.predict_tanh(x), np.float32)
    # Add noise to d_true
    # eps = tf.random.normal(shape=d_true.shape)
    # d_true += eps*network.data_std
    # d_true_log = tf.math.log(-d_true)
    mean, logvar = network.encode(x)
    z = network.reparameterize(mean, logvar)
    zd = tf.concat((z, d_true_log), -1)
    x_tanh = network.decode(zd, apply_tanh=True)
    d_pre = tf.cast(network.predict_tanh(x_tanh), np.float32)
    # d_pre = tf.math.log(-tf.cast(network.predict_tanh(x_tanh), np.float32))
    data_misfit = tf.reduce_mean(
        network.data_mean_error(d_true, d_pre,
                                sample_weight=network.data_weights))
    logpx_z = tf.reduce_mean(
        network.model_mean_error(tf.reshape(x_tanh, (-1, network.n_model)),
                                 tf.reshape(x, (-1, network.n_model)),
                                 sample_weight=network.model_weights))
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
    return (loss, terms)


@tf.function
def compute_reconstruction_loss(network, xy):
    '''
    No data misfit
    '''
    x = xy[0]
    d_true_log = tf.cast(xy[1], np.float32)
    d_true = -tf.exp(d_true_log)
    # d_true = tf.math.log(-tf.cast(network.predict_tanh(x), np.float32))
    # d_true = tf.cast(network.predict_tanh(x), np.float32)
    # d_true_log = tf.math.log(-d_true)
    mean, logvar = network.encode(x)
    z = network.reparameterize(mean, logvar)
    zd = tf.concat((z, d_true_log), -1)
    x_tanh = network.decode(zd, apply_tanh=True)
    logpx_z = tf.reduce_mean(
        network.model_mean_error(tf.reshape(x_tanh, (-1, network.n_model)),
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
def compute_apply_gradients(network, xy, optimizer, use_data_misfit=True):
    with tf.GradientTape() as tape:
        if use_data_misfit:
            loss, terms = compute_loss(network, xy)
        else:
            loss, terms = compute_reconstruction_loss(network, xy)
    gradients = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, network.trainable_variables))
    return (loss, terms)


def compute_losses(network, xy):
    '''
    compute each loss separately, for evaluating performance
    '''
    x = xy[0]
    d_true_log = tf.cast(xy[1], np.float32)
    d_true = -tf.exp(d_true_log)
    # d_true = tf.cast(network.predict_tanh(x), np.float32)
    # d_true_log = tf.math.log(-d_true)
    mean, logvar = network.encode(x)
    z = network.reparameterize(mean, logvar)
    zd = tf.concat((z, d_true_log), -1)
    x_tanh = network.decode(zd, apply_tanh=True)
    d_pre = tf.cast(network.predict_tanh(x_tanh), np.float32)
    # d_pre = tf.math.log(-tf.cast(network.predict_tanh(x_tanh), np.float32))
    # data_misfit = network.data_mean_error(
    #     d_true, d_pre, sample_weight=network.n_data/(network.data_std**2))
    data_misfit = network.data_mean_error(d_true, d_pre)
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
