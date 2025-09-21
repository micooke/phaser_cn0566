#!/usr/bin/env python3

import pickle
import numpy as np
from scipy import signal

import matplotlib.pyplot as plt

nominal_hb100_freq_Hz: float = 10.525e9
Phaser_LO_LOW: float = 2.5e9
Phaser_LO_HIGH: float = 1.6e9  # 1.6e9

def print_vector(input, num_values:int = 5):
    print("["+f"{input[:num_values]}"+"..."+f"{input[-num_values:]}"+"]")

# @author: ADI
def spectrum_estimate(x, fs, ref=2**15):
    eps = 10**-20
    N = len(x)

    # Apply window
    window = signal.windows.kaiser(N, beta=38)
    window /= np.average(window)
    x = np.multiply(x, window)

    # Use FFT to get the amplitude of the spectrum
    ampl = (
        np.fft.fftshift(np.absolute(np.fft.fft(x, n=N))) / N
    )  # normalise to FFT length
    ampl = 20 * np.log10((ampl / ref) + eps)

    # FFT frequency bins
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs))

    return ampl, freqs


# N_CUT - the number of samples to cut from the max estimation
# window_length - the number of samples to use for estimating the max noise value
# @author: Mark Cooke
def noise_estimate(amplitude_dBFS, N_CUT: int = 5, window_length: int = 100):
    n_samples = len(amplitude_dBFS)
    # max hold over the samples
    max_hold_threshold = np.array([-np.Inf] * n_samples)
    half_window_length = window_length // 2
    for w in np.arange(half_window_length, n_samples, half_window_length):
        # get the window
        window_data = amplitude_dBFS[
            w - half_window_length : w + half_window_length + 1
        ]
        window_data = np.sort(window_data)[
            N_CUT:-N_CUT
        ]  # remove the top N_CUT and bottom N_CUT values
        window_mean = window_data.mean()
        window_std = window_data.std()

        # take the max of all values within the mean (+/-) std
        max_val = np.max(
            window_data[
                (window_data >= (window_mean - window_std))
                & (window_data <= (window_mean + window_std))
            ]
        )

        # print(f"{max_val:,.3f}:[{window_mean-window_std:,.3f}|{window_mean:,.3f}|{window_mean+window_std:,.3f}]")

        # max_val = window_data[(window_data >= (window_mean-window_std)) & (window_data <= (window_mean+window_std))].max()

        max_hold_threshold[w - half_window_length : w + half_window_length] = max_val
    return max_hold_threshold


# plot the FFT spectrum, signal threshold and plot legend
# @author: Mark Cooke
def plot_spectrum(full_freq_range, full_amp_range, signal_threshold):
    peak_index = np.argmax(full_amp_range)
    peak_freq = full_freq_range[peak_index]

    plt.figure(1)
    plt.title(f"Full Spectrum, peak at {full_freq_range[peak_index]*1e3:,.3f}MHz.")
    plt.plot(full_freq_range, full_amp_range, label="spectrum")
    plt.plot(full_freq_range, signal_threshold, label="signal threshold")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Signal Strength [dBFS]")
    plt.legend()
    plt.show()
    print("You may need to close this plot to continue...")


def save_pkl(data, filename="hb100_freq_val.pkl"):
    with open(filename, "wb") as fb:
        pickle.dump(data, fb)  # save calibrated gain value to a file


def load_pkl(filename="hb100_freq_val.pkl"):
    try:
        with open(filename, "rb") as fb:
            data = pickle.load(fb)  # Load gain cal values
    except Exception:
        print(f"[ERROR] File not found: {filename}")
        data = None
    return data
