import numpy as np
import os
from matplotlib import pyplot as plt

from SimPEG import maps
import SimPEG.electromagnetics.time_domain as tdem
from SimPEG.utils import plot_1d_layer_model
from SimPEG.electromagnetics.utils.em1d_utils import ColeCole, LogUniform


def EM(times, thicknesses):
        # super(EM, self).__init__()
        eps = 1e-6
        # I used provided start and end times for first pulse to accurately create the correctly sized square-waveform
        ramp_on = np.r_[0.007, 0.007 + eps]
        ramp_off = np.r_[0.020-eps, 0.020]
        times_currents = np.r_[ramp_on, ramp_off]
        times_currents -= times_currents.max()
        currents = np.array([0,1,1,0])
        waveform = tdem.sources.PiecewiseLinearWaveform(times_currents, currents)

        receiver_location = np.array([[-108,0, 68]])  # Receiver location
        receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
            locations=receiver_location, times=times, orientation="z",)

        receiver_b = tdem.receivers.PointMagneticFluxDensity(
            locations=receiver_location, times=times, orientation="z",)

    # Sources
        source_location = np.array([0, 0, 120])  # Source location
        source = tdem.sources.MagDipole(
            receiver_list=[receiver, receiver_b],
            waveform=waveform,
            location=source_location,
            # radius = 7,  # Transmitter loop radius
            moment = 1, # usually the data are normalized by the Tx moment. 
            srcType = 'inductive')

    # Survey
        survey = tdem.Survey([source])

        model_mapping = maps.IdentityMap(nP=32)
        # print(len(thicknesses))
    # Simulate response for static conductivity
        simulation_conductive = tdem.Simulation1DLayered(
            survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping)
            
        return simulation_conductive

def forward_vec_freq(EM,con):
        dpred_conductive = EM.dpred(con)
        dpred = dpred_conductive*1e11 #Double check unit conversion
        # print('dpred',len(dpred))
        return dpred


def gradient(EM,con,v):
     return EM.Jtvec(con,v)

# def forward(con, thicknesses, freqs):
#         '''
#         Wrapper for forward calcs
#         Only return Zxy at surface
#         '''
#         # Zs = [forward_1_freq(con,thicknesses,freq)[0] for freq in freqs]
#         Zs = forward_vec_freq(con, thicknesses, freqs)[:, 0]
#         return Zs


# def forward_1_freq(con, thicknesses, freq):
#         '''
#         Compute 1D isotropic MT response
#         Return Zxy at each interface
#         con is an array of conductivities of length n, in S/m
#         thicknesses is an array of thicknesses of length n-1, in m
#         freq is frequency, in Hz
#         Note: in 1D isotropic, Zyx = -Zxy, Zxx=Zyy=0
#         TODO: allow multiple frequencies, broadcast
#         '''
#         omega = 2*np.pi*freq
#         alpha = np.sqrt(1j*omega*mu0*con)
#         n = len(con)
#         assert len(thicknesses)==n-1, 'con and thicknesses must be same length'
#         Z = 1j*np.zeros(len(con))
#         Z[-1] = -alpha[-1]/con[-1]
#     # iterate from next to last layer up to first layer
#         for ii in range(n-2,-1,-1):
#             Z[ii] = ((Z[ii+1] - alpha[ii]/con[ii]*np.tanh(alpha[ii]*thicknesses[ii]))/
#                      (1 - con[ii]/alpha[ii]*Z[ii+1]*np.tanh(alpha[ii]*thicknesses[ii])))
#         return Z

# def gradient_1_freq(con, thicknesses, freq):
#     '''
#     Get gradients of datum wrt model (conductivities)
#     '''
#     Z = forward_1_freq(con, thicknesses, freq)
#     dZdZ1 = gradient_Z_1_freq(Z, con, thicknesses, freq)
#     dZidconi = gradient_Z_con_1_freq(Z, con, thicknesses, freq)
#     dZ1dZi = np.cumprod(dZdZ1)
#     dZdcon = dZidconi
#     dZdcon[1:] *= dZ1dZi
#     return dZdcon


# def gradient(con, thicknesses, freqs):
#     '''
#     Get gradients of datum wrt model (conductivities)
#     '''
#     Z = forward_vec_freq(con, thicknesses, freqs)
#     dZdZ1 = gradient_Z(Z, con, thicknesses, freqs)
#     dZidconi = gradient_Z_con(Z, con, thicknesses, freqs)
#     dZ1dZi = np.cumprod(dZdZ1, axis=1)
#     dZdcon = dZidconi
#     dZdcon[:, 1:] *= dZ1dZi
#     return dZdcon


# def gradient_Z_1_freq_unstable(Z, con, thicknesses, freq):
#     '''
#     Get gradients of impedances wrt impedance at interface below
#     '''
#     omega = 2*np.pi*freq
#     alpha = np.sqrt(1j*omega*mu0*con)
#     dZdZ1 = (np.cosh(alpha[:-1]*thicknesses) - 
#              con[:-1]/alpha[:-1]*Z[1:]*np.sinh(alpha[:-1]*thicknesses)
#             )**(-2)
#     return dZdZ1


# def gradient_Z_1_freq(Z, con, thicknesses, freq):
#     '''
#     Get gradients of impedances wrt impedance at interface below
#     '''
#     omega = 2*np.pi*freq
#     alpha = np.sqrt(1j*omega*mu0*con)
#     dZdZ1 = (1 - np.tanh(alpha[:-1]*thicknesses)**2) / (
#         1 - con[:-1]/alpha[:-1]*Z[1:]*np.tanh(alpha[:-1]*thicknesses)
#     )**2
#     return dZdZ1


# def gradient_Z(Z, con, thicknesses, freqs):
#     '''
#     Get gradients of impedances wrt impedance at interface below
#     '''
#     # Is impedance relevant for EM gradients? Does it need to be w.r.t.?
#     omega = 2*np.pi*freqs
#     alpha = np.sqrt(1j*mu0*np.outer(omega, con))
#     ad = alpha[:, :-3]*thicknesses[None, :]
#     Z2 = Z[:, 1:]**2
#     tanh = np.tanh(ad)
#     tanh2 = tanh**2
#     sech2 = 1 - tanh2
#     dZdZ1 = sech2 / (1 - con[None, :-3]/alpha[:, :-3]*Z[:, 1:]*tanh)**2
#     return dZdZ1


# def gradient_Z_con_1_freq(Z, con, thicknesses, freq):
#     '''
#     Gradients of impedances wrt conductivity of their respective layer
#     '''
#     omega = 2*np.pi*freq
#     alpha = np.sqrt(1j*omega*mu0*con)
#     dZdcon = 1j*np.zeros(Z.shape)
#     # temp values used in big expression
#     ad = alpha[:-1]*thicknesses
#     Z2 = Z[1:]**2
#     tanh = np.tanh(ad)
#     tanh2 = tanh**2
#     sech2 = 1 - tanh2
#     ao2 = alpha[:-1]**2/(con[:-1]**2)
#     dZdcon[:-1] = thicknesses/2 * sech2 * (Z2 - ao2)
#     dZdcon[:-1] += 1/2/alpha[:-1] * tanh * (Z2 + ao2)
#     dZdcon[:-1] -= 1/con[:-1] * Z[1:] * tanh2
#     dZdcon[:-1] /= (1 - con[:-1]/alpha[:-1] * Z[1:] * tanh)**2
#     dZdcon[-1] = alpha[-1]/(2*con[-1]**2)
#     return dZdcon


# def gradient_Z_con(Z, con, thicknesses, freqs):
#     '''
#     Gradients of impedances wrt conductivity of their respective layer
#     '''
#     omega = 2*np.pi*freqs
#     alpha = np.sqrt(1j*mu0*np.outer(omega, con))
#     dZdcon = 1j*np.zeros(Z.shape)
#     # temp values used in big expression
#     ad = alpha[:, :-3]*thicknesses[None, :]
#     Z2 = Z[:, 1:]**2
#     tanh = np.tanh(ad)
#     tanh2 = tanh**2
#     sech2 = 1 - tanh2
#     ao2 = alpha[:, :-3]**2/(con[None, :-3]**2)
#     dZdcon[:, :-1] = thicknesses[None, :]/2 * sech2 * (Z2 - ao2)
#     dZdcon[:, :-1] += 1/2/alpha[:, :-3] * tanh * (Z2 + ao2)
#     dZdcon[:, :-1] -= 1/con[None, :-3] * Z[:, 1:] * tanh2
#     dZdcon[:, :-1] /= (1 - con[None, :-3]/alpha[:, :-3] * Z[:, 1:] * tanh)**2
#     dZdcon[:, -1] = alpha[:, -1]/(2*con[None, -1]**2)
#     return dZdcon

