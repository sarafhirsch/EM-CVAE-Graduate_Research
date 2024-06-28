#!/usr/bin/env python
# coding: utf-8

# MT VAE training runs
# A good idea for how to organize
# No time to truly implement
# An organization system to keep in mind for future research
# Downside: it can be cumbersome to implement EVERY change as an argument to a function

import os
import time
import gc
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from . import cvae_mt64 as vae


def n(*args, **kwargs):
    '''
    No data misfit
    '''
    targs = {'use_data_misfit':False,
             'log_data':True,
             'norm_data':False
            }
    targs.update(*args, **kwargs)
    return targs


def fn(*args, **kwargs):
    '''
    "fixed" (normalized data rather than log-data)
    No data misfit
    '''
    targs = {'use_data_misfit':False,
             'log_data':False,
             'norm_data':True
            }
    targs.update(*args, **kwargs)
    return targs


def fd(*args, **kwargs):
    '''
    "fixed" (normalized data rather than log-data)
    No data misfit
    '''
    targs = {'use_data_misfit':True,
             'log_data':False,
             'norm_data':True
            }
    targs.update(*args, **kwargs)
    return targs


def fn4(*args, **kwargs):
    targs = {'run':'fn4',
             'beta_vae':0.1,
             'epochs':10000
            }
    targs = fn(**targs)
    targs.update(*args, **kwargs)
    train(**targs)


def train(run='test', beta_vae=1, epochs=100,
          log_data=True, norm_data=False,
          use_data_misfit=True,
          model_loss_type='se',
          data_loss_type='se',
          data_std_type='rel_std',
          model_std_type='train_std',
          latent_dim=20,
          initializer='he_normal',
          optimizer=tf.keras.optimizers.Adam(0.0002, 0.9)
         ):
    '''
    Do a training run (experiment)
    TODO: implement...
        data_std_type
        model_std_type
        initializer
        Allow pretrained networks to be loaded
    '''

    if not os.path.exists(run):
        os.makedirs(run)

    # Set up survey
    # remote reference frequencies
    #   7.680002e+02   5.120000e+02   3.840000e+02   2.560000e+02   1.920000e+02   1.280000e+02
    #   9.599997e+01   6.400000e+01   4.800001e+01   3.200000e+01   2.400000e+01   1.600000e+01
    #   1.200000e+01   8.000000e+00   5.999999e+00   4.000000e+00   3.000000e+00   2.000000e+00
    #   1.500000e+00   1.000000e+00   7.500002e-01   5.000000e-01   3.750000e-01   2.500000e-01
    #   1.875000e-01

    # mesh
    # 64 ft
    cell_size = 64*0.3048
    depth_to_top = 0*0.3048
    n_cells = 64
    # one fewer depth; last cell extends to inf
    depths = depth_to_top + np.arange(1,n_cells)*cell_size
    n_depths = len(depths)

    # data frequencies
    # conservative, lines up with remote referenced stations, minus one frequency to avoid extrapolation
    f_a = np.logspace(-2, 9, num=12, base=2)
    f_b = np.logspace(-4, 7, num=12, base=2)*3
    frequencies = np.sort(np.r_[f_a, f_b])
    #frequencies = np.logspace(-4, 10, 15, base=2)
    nf = len(frequencies)
    # data are real and imaginary Zxy at each frequency
    ndata = 2*nf

    # Load training models and Create neural network
    # normalize model parameters between -1 and 1
    # remember, min resistivity is 0.01, max is 1e5
    # Gaussian infill potentially allows for values outside this range, but not likely
    min_model = np.log(1e-5)
    max_model = np.log(1e2)
    # pad by norm_pad, so that a bunch of values don't end up at -1
    norm_pad = 0.1

    # create network
    network = vae.CVAE(depths,
                       min_model=min_model,
                       max_model=max_model,
                       frequencies=frequencies,
                       norm_pad=norm_pad,
                       #data_std=0.1,
                       #model_std=.01,
                       beta_vae=beta_vae,
                       model_loss_type=model_loss_type,
                       data_loss_type=data_loss_type
                      )

    def preprocess(filename):
        '''
        Read RILD values from npy file
        Convert to log conductivity
        Reshape to include channel dimension
        '''
        x = np.log(1/np.load(filename))
        x = x.reshape(-1, n_cells, 1)
        return x

    x_train_log = preprocess('KGS_RILD_64ft_train.npy')
    x_validate_log = preprocess('KGS_RILD_64ft_validate.npy')
    x_test1_log = preprocess('KGS_RILD_64ft_test1.npy')
    x_test2_log = preprocess('KGS_RILD_64ft_test2.npy')

    x_train = network.model_to_tanhs(x_train_log)
    x_validate = network.model_to_tanhs(x_validate_log)
    x_test1 = network.model_to_tanhs(x_test1_log)
    x_test2 = network.model_to_tanhs(x_test2_log)

    # compute stds
    model_std = np.std(x_train.flatten())
    #mt_data_file = 'KGS_MT.npy'
    #os.remove(mt_data_file)
    #try:
    #    all_data = tf.convert_to_tensor(np.load(mt_data_file))
    #except FileNotFoundError:
    #    all_data = network.predict_tanh(x_train)
    #    np.save(mt_data_file, all_data.numpy())

    train_data = network.predict_tanh(x_train)
    # log_train_data = tf.math.log(-train_data)
    std_train_data = np.std(train_data.numpy(), axis=0)
    mean_train_data = np.mean(train_data.numpy(), axis=0)
    norm_train_data = (train_data - mean_train_data)/std_train_data

    validate_data = network.predict_tanh(x_validate)
    norm_validate_data = (validate_data - mean_train_data)/std_train_data

    test1_data = network.predict_tanh(x_test1)
    norm_test1_data = (test1_data - mean_train_data)/std_train_data

    test2_data = network.predict_tanh(x_test2)
    norm_test2_data = (test2_data - mean_train_data)/std_train_data

    # Create batches and shuffle
    BATCH_SIZE = 79

    train_dataset = tf.data.Dataset.from_tensor_slices((
        tf.cast(x_train, tf.float32), 
        tf.cast(norm_train_data, tf.float32))).shuffle(10000).batch(BATCH_SIZE)

    validate_dataset = tf.data.Dataset.from_tensor_slices((
        tf.cast(x_validate, tf.float32),
        tf.cast(norm_validate_data, tf.float32))).shuffle(10000).batch(x_validate.shape[0])

    # though the data vary over several orders of magnitude, 
    # they don't vary so much within one frequency.
    # data_std_vec = np.std(train_data.numpy(), axis=0)
    # log_data_std_vec = np.std(log_train_data.numpy(), axis=0)
    # log_data_std = np.std(log_train_data.numpy().flatten())
    # average_log_data_std = np.std((log_train_data.numpy() - log_train_data.numpy().mean(axis=0)).flatten())
    # compute elementwise stds
    model_std_vec = np.std(x_train, axis=0)
    model_std_vec = np.reshape(model_std_vec, (n_cells))
    # compute relative std
    rel_data_std = np.abs(mean_train_data)

    # same for model, but mean over all
    mean_model_value = np.mean(x_train)
    rel_model_std = 0.5*mean_model_value

    # finalize network
    network = vae.CVAE(depths,
                       min_model=min_model,
                       max_model=max_model,
                       frequencies=frequencies,
                       norm_pad=norm_pad,
                       data_std=rel_data_std.reshape(1, -1, 1),
                       model_std=model_std,
                       #model_std=model_std_vec.reshape(1, -1, 1),
                       latent_dim=latent_dim,
                       beta_vae=beta_vae,
                       model_loss_type=model_loss_type,
                       data_loss_type=data_loss_type,
                       log_data=log_data,
                       norm_data=norm_data,
                       data_scale=1/std_train_data,
                       data_shift=mean_train_data
                      )

    print('data_std mean: {}'.format(np.mean(network.data_std)))
    print('model_std mean: {}'.format(np.mean(network.model_std)))

    # pick some random training models
    # i_random_train = np.random.randint(0, x_train.shape[0], 16)
    i_random_train = np.arange(16)
    random_train = x_train[i_random_train].reshape((16, n_cells))
    # predict their data
    random_data = network.predict_tanh(random_train.reshape(16, n_cells, 1))
    # random_log_data = tf.math.log(-random_data)
    random_norm_data = (random_data - mean_train_data)/std_train_data
    # Save data and latent space inputs for plots
    latent_input = tf.random.normal([16, network.latent_dim], seed=0, dtype=tf.float32)
    data_input = tf.reshape(random_norm_data,(16,network.n_data))
    zd_input = tf.concat((latent_input,data_input),axis=-1)

    # plot a few random training models and their data
    vae.plot_complex(random_data, frequencies=frequencies, save2file=True, 
                     filename=run+'/training_MT_data.png')
    vae.plot_logs(np.exp(network.tanhs_to_model(random_train)), depths=depths, save2file=True, 
                  filename=run + '/training_models.png')

    # Save starting plots
    network.plot_models(save2file=True,folder=run,samples=zd_input.shape[0],
                        latent=zd_input,step=0)
    network.plot_data(save2file=True,folder=run,latent=zd_input,step=0)
    network.plot_residuals(save2file=True,folder=run,latent=zd_input,step=0)

    plt.close('all')
    plt.clf()
    gc.collect()

    # network.inference_net = tf.keras.models.load_model('n6/encoder.h5')
    # network.generative_net = tf.keras.models.load_model('n6/decoder.h5')

    # optimizer = tf.keras.optimizers.Adam(0.0002, 0.9)

    # Train
    validate_terms = []
    train_terms = []
    #train_losses = []
    #ttest_losses = []
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:#.batch(BATCH_SIZE):
            train_loss, train_term = vae.compute_apply_gradients(network, train_x, optimizer, 
                                        use_data_misfit=use_data_misfit, rel_noise=0.005)
            #train_losses.append(train_loss.numpy())
            train_terms.append([tt.numpy() for tt in train_term])
            #for test_x in test_dataset:
            #    ttest_loss = vae.compute_losses(network, test_x)
            #    terms = [loss(l).numpy() for l in ttest_loss]
            #    ttest_losses.append(terms)
        end_time = time.time()

        if epoch % 100 == 0:
            # compute and save losses
            for val_x in validate_dataset:#.batch(BATCH_SIZE):
                val_loss, val_term = vae.compute_loss(network, val_x)
                #terms = [tf.reduce_mean(l).numpy() for l in losses]
                #loss(vae.compute_reconstruction_loss(network, test_x))
            #elbo = -loss.result()
            validate_terms.append([tt.numpy() for tt in val_term])
            print('Epoch: {}, Data misfit: {:.4}, '
                  'Reconstruction: {:.4}, '
                  'KL: {:.4}, '
                  'Elapsed {:.4} s'.format(epoch, val_term[0], val_term[1], val_term[2],
                                        #elbo,
                                        end_time - start_time))

        if epoch % 1e3 == 0:
            network.plot_models(save2file=True,folder=run,samples=zd_input.shape[0],
                     latent=zd_input,step=epoch)
            network.plot_data(save2file=True,folder=run,latent=zd_input,step=epoch)
            network.plot_residuals(save2file=True,folder=run,latent=zd_input,step=epoch)
            plt.close('all')
            gc.collect()

    network.inference_net.save(run+'/encoder.h5')
    network.generative_net.save(run+'/decoder.h5')
    np.save(run+'/optimizer_weights.npy', optimizer.get_weights())
    np.save(run+'/losses.npy', np.array(validate_terms))
    np.save(run+'/train_losses.npy', np.array(train_terms))

    network.plot_models(save2file=True,folder=run,samples=zd_input.shape[0],
             latent=zd_input,step=epoch)
    network.plot_data(save2file=True,folder=run,latent=zd_input,step=epoch)
    network.plot_residuals(save2file=True,folder=run,latent=zd_input,step=epoch)


