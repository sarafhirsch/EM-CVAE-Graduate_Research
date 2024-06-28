import numpy as np
import os
from matplotlib import pyplot as plt

from SimPEG import maps
import SimPEG.electromagnetics.time_domain as tdem
from SimPEG.utils import plot_1d_layer_model
from SimPEG.electromagnetics.utils.em1d_utils import ColeCole, LogUniform


class EM:
    def __init__(self, source_location = np.array([0.0, 0.0, 120]), source_orientation = "z", current_amplitude = 1.0, source_radius = 10.0,
              receiver_location = np.array([0.0, 0.0, 120]), receiver_orientation = "z"):
        # super(EM, self).__init__()
        eps = 1e-6
        ramp_on = np.r_[0.007-eps, 0.007]
        ramp_off = np.r_[0.02,0.02+eps]
        self.waveform = tdem.sources.TrapezoidWaveform(
                    ramp_on=ramp_on, ramp_off=ramp_off)

        self.source_location = source_location
        self.source_orientation = source_orientation   # "x", "y" or "z"
        self.current_amplitude = current_amplitude  # maximum amplitude of source current
        self.source_radius = source_radius  # loop radius

        self.receiver_location = receiver_location
        self.receiver_orientation = receiver_orientation  # "x", "y" or "z"

    def forward_vec_freq(self, con, thicknesses, times):
        '''
        Compute 1D isotropic MT response
        Return Zxy at each interface
        con is an array of conductivities of length n, in S/m
        thicknesses is an array of thicknesses of length n-1, in m
        freqs is an array of frequencies, in Hz
        Note: in 1D isotropic, Zyx = -Zxy, Zxx=Zyy=0
        TODO: use cumsum to speed up even more
        '''
        if len(con) == 32:
            con = np.delete(con,[-2,-1])
#     if len(con) == 28:
#         con = np.append(con, [1])
    # print(len(con))
#     print(len(thicknesses))
   
        n = len(con)
        assert len(thicknesses)==n-1, 'con and thicknesses must be same length'

    # Receiver list
        receiver_list = []
        receiver_list.append(
            tdem.receivers.PointMagneticFluxDensity(
                self.receiver_location, times, orientation=self.receiver_orientation))
        receiver_list.append(
            tdem.receivers.PointMagneticFluxTimeDerivative(
                self.receiver_location, times, orientation=self.receiver_orientation))

    # Sources
        source_list = [
            tdem.sources.CircularLoop(
                receiver_list=receiver_list,
                location=self.source_location,
                waveform=self.waveform,
                current=self.current_amplitude,
                radius=self.source_radius,)]

    # Survey
        survey = tdem.Survey(source_list)

        model_mapping = maps.IdentityMap(nP=n)

    # Simulate response for static conductivity
        simulation_conductive = tdem.Simulation1DLayered(
            survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping)
    # points = len(con
        dpred_conductive = simulation_conductive.dpred(con)
        # print('dpred', dpred_conductive.shape)
    # print('dpred_conductive', dpred_conductive.shape)
    # omega = 2*np.pi*freqs
    # alpha = np.sqrt(1j*mu0*np.outer(omega, con))
    # tanh_ad = np.tanh(alpha[:, :-1] * thicknesses[None, :])
    # ac_tanh_ad = alpha[:, :-1]/con[None, :-1]*tanh_ad
    # ca_tanh_ad = con[None, :-1]/alpha[:, :-1]*tanh_ad
    # Z = 1j*np.zeros(alpha.shape)
    # Z[:, -1] = -alpha[:, -1]/con[-1]
    # # iterate from next to last layer up to first layer
    # for ii in range(n-2,-1,-1):
    #     # Z[:, ii] = ((Z[:, ii+1] - alpha[:, ii]/con[ii]*tanh_ad[:, ii]) /
    #     #             (1 - con[ii]/alpha[:, ii]*Z[:, ii+1]*tanh_ad[:, ii]))
    #     Z[:, ii] = ((Z[:, ii+1] - ac_tanh_ad[:, ii]) /
    #                 (1 - ca_tanh_ad[:, ii]*Z[:, ii+1]))
        return dpred_conductive


    def forward(con, thicknesses, freqs):
        '''
        Wrapper for forward calcs
        Only return Zxy at surface
        '''
        # Zs = [forward_1_freq(con,thicknesses,freq)[0] for freq in freqs]
        Zs = forward_vec_freq(con, thicknesses, freqs)[:, 0]
        return Zs


    def forward_1_freq(con, thicknesses, freq):
        '''
        Compute 1D isotropic MT response
        Return Zxy at each interface
        con is an array of conductivities of length n, in S/m
        thicknesses is an array of thicknesses of length n-1, in m
        freq is frequency, in Hz
        Note: in 1D isotropic, Zyx = -Zxy, Zxx=Zyy=0
        TODO: allow multiple frequencies, broadcast
        '''
        omega = 2*np.pi*freq
        alpha = np.sqrt(1j*omega*mu0*con)
        n = len(con)
        assert len(thicknesses)==n-1, 'con and thicknesses must be same length'
        Z = 1j*np.zeros(len(con))
        Z[-1] = -alpha[-1]/con[-1]
    # iterate from next to last layer up to first layer
        for ii in range(n-2,-1,-1):
            Z[ii] = ((Z[ii+1] - alpha[ii]/con[ii]*np.tanh(alpha[ii]*thicknesses[ii]))/
                     (1 - con[ii]/alpha[ii]*Z[ii+1]*np.tanh(alpha[ii]*thicknesses[ii])))
        return Z

def gradient_1_freq(con, thicknesses, freq):
    '''
    Get gradients of datum wrt model (conductivities)
    '''
    Z = forward_1_freq(con, thicknesses, freq)
    dZdZ1 = gradient_Z_1_freq(Z, con, thicknesses, freq)
    dZidconi = gradient_Z_con_1_freq(Z, con, thicknesses, freq)
    dZ1dZi = np.cumprod(dZdZ1)
    dZdcon = dZidconi
    dZdcon[1:] *= dZ1dZi
    return dZdcon


def gradient(con, thicknesses, freqs):
    '''
    Get gradients of datum wrt model (conductivities)
    '''
    Z = forward_vec_freq(con, thicknesses, freqs)
    dZdZ1 = gradient_Z(Z, con, thicknesses, freqs)
    dZidconi = gradient_Z_con(Z, con, thicknesses, freqs)
    dZ1dZi = np.cumprod(dZdZ1, axis=1)
    dZdcon = dZidconi
    dZdcon[:, 1:] *= dZ1dZi
    return dZdcon


def gradient_Z_1_freq_unstable(Z, con, thicknesses, freq):
    '''
    Get gradients of impedances wrt impedance at interface below
    '''
    omega = 2*np.pi*freq
    alpha = np.sqrt(1j*omega*mu0*con)
    dZdZ1 = (np.cosh(alpha[:-1]*thicknesses) - 
             con[:-1]/alpha[:-1]*Z[1:]*np.sinh(alpha[:-1]*thicknesses)
            )**(-2)
    return dZdZ1


def gradient_Z_1_freq(Z, con, thicknesses, freq):
    '''
    Get gradients of impedances wrt impedance at interface below
    '''
    omega = 2*np.pi*freq
    alpha = np.sqrt(1j*omega*mu0*con)
    dZdZ1 = (1 - np.tanh(alpha[:-1]*thicknesses)**2) / (
        1 - con[:-1]/alpha[:-1]*Z[1:]*np.tanh(alpha[:-1]*thicknesses)
    )**2
    return dZdZ1


def gradient_Z(Z, con, thicknesses, freqs):
    '''
    Get gradients of impedances wrt impedance at interface below
    '''
    # Is impedance relevant for EM gradients? Does it need to be w.r.t.?
    omega = 2*np.pi*freqs
    alpha = np.sqrt(1j*mu0*np.outer(omega, con))
    ad = alpha[:, :-3]*thicknesses[None, :]
    Z2 = Z[:, 1:]**2
    tanh = np.tanh(ad)
    tanh2 = tanh**2
    sech2 = 1 - tanh2
    dZdZ1 = sech2 / (1 - con[None, :-3]/alpha[:, :-3]*Z[:, 1:]*tanh)**2
    return dZdZ1


def gradient_Z_con_1_freq(Z, con, thicknesses, freq):
    '''
    Gradients of impedances wrt conductivity of their respective layer
    '''
    omega = 2*np.pi*freq
    alpha = np.sqrt(1j*omega*mu0*con)
    dZdcon = 1j*np.zeros(Z.shape)
    # temp values used in big expression
    ad = alpha[:-1]*thicknesses
    Z2 = Z[1:]**2
    tanh = np.tanh(ad)
    tanh2 = tanh**2
    sech2 = 1 - tanh2
    ao2 = alpha[:-1]**2/(con[:-1]**2)
    dZdcon[:-1] = thicknesses/2 * sech2 * (Z2 - ao2)
    dZdcon[:-1] += 1/2/alpha[:-1] * tanh * (Z2 + ao2)
    dZdcon[:-1] -= 1/con[:-1] * Z[1:] * tanh2
    dZdcon[:-1] /= (1 - con[:-1]/alpha[:-1] * Z[1:] * tanh)**2
    dZdcon[-1] = alpha[-1]/(2*con[-1]**2)
    return dZdcon


def gradient_Z_con(Z, con, thicknesses, freqs):
    '''
    Gradients of impedances wrt conductivity of their respective layer
    '''
    omega = 2*np.pi*freqs
    alpha = np.sqrt(1j*mu0*np.outer(omega, con))
    dZdcon = 1j*np.zeros(Z.shape)
    # temp values used in big expression
    ad = alpha[:, :-3]*thicknesses[None, :]
    Z2 = Z[:, 1:]**2
    tanh = np.tanh(ad)
    tanh2 = tanh**2
    sech2 = 1 - tanh2
    ao2 = alpha[:, :-3]**2/(con[None, :-3]**2)
    dZdcon[:, :-1] = thicknesses[None, :]/2 * sech2 * (Z2 - ao2)
    dZdcon[:, :-1] += 1/2/alpha[:, :-3] * tanh * (Z2 + ao2)
    dZdcon[:, :-1] -= 1/con[None, :-3] * Z[:, 1:] * tanh2
    dZdcon[:, :-1] /= (1 - con[None, :-3]/alpha[:, :-3] * Z[:, 1:] * tanh)**2
    dZdcon[:, -1] = alpha[:, -1]/(2*con[None, -1]**2)
    return dZdcon

