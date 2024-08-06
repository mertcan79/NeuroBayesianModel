import numpy as np
from scipy import signal

class EEGAugmenter:
    @staticmethod
    def add_gaussian_noise(eeg_data, std=0.1):
        return eeg_data + np.random.normal(0, std, eeg_data.shape)

    @staticmethod
    def time_warp(eeg_data, sigma=0.1, num_knots=4):
        orig_steps = np.arange(eeg_data.shape[1])
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(eeg_data.shape[0], num_knots+2))
        warp_steps = (np.cumsum(random_warps, axis=1) * (eeg_data.shape[1]-1)/(num_knots+1)).T
        warper = np.array([signal.CubicSpline(warp_steps[:, channel], orig_steps)(orig_steps) for channel in range(eeg_data.shape[0])])
        return np.array([np.interp(warper[channel], orig_steps, eeg_data[channel]) for channel in range(eeg_data.shape[0])])

    @staticmethod
    def frequency_shift(eeg_data, fs, max_shift=1.0):
        shift = np.random.uniform(-max_shift, max_shift)
        return signal.resample(eeg_data, int(eeg_data.shape[1] * (fs + shift) / fs), axis=1)

    @staticmethod
    def amplitude_scale(eeg_data, scale_range=(0.8, 1.2)):
        scale = np.random.uniform(scale_range[0], scale_range[1], size=(eeg_data.shape[0], 1))
        return eeg_data * scale