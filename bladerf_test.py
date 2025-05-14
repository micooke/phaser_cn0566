import matplotlib.pyplot as plt
from bladerf import _bladerf
import numpy as np
import time
import sys
import math
from scipy import signal

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--adc_bits", type=int, default=16, help="Number of ADC bits")
parser.add_argument(
    "-b", "--buffer_bits", type=int, default=21, help="Buffer size = 2^buffer_bits"
)
parser.add_argument("-p", "--plot", type=int, default=1, help="Plot the samples")
parser.add_argument("-c", "--fc", default=None, help="Centre Frequency in MHz")
parser.add_argument(
    "-s", "--fs", type=float, default=40.0, help="Sampling Rate in Msps"
)
# "-s", "--fs", type=float, default=61.440, help="Sampling Rate in Msps"

## conversion reminder:
# 1 / Hz  =  s
# 1 / kHz = ms
# 1 / MHz = us


# add additional functions to the standard BladeRF class to enable or disable oversampling
class blade_RF(_bladerf.BladeRF):
    def enable_oversample(self):
        ret = _bladerf.libbladeRF.bladerf_enable_feature(self.dev[0], 1, True)
        _bladerf._check_error(ret)

    def disable_oversample(self):
        ret = _bladerf.libbladeRF.bladerf_enable_feature(self.dev[0], 0, True)
        _bladerf._check_error(ret)


def vector_string(v):
    return "[" + " ".join([f"{v_:,.1f}" for v_ in v]) + "]"


def shutdown(error=0, board=None):
    print("Shutting down with error code: " + str(error))
    if board is not None:
        board.close()
    sys.exit(error)


def rx_config(
    bladerf1,
    sample_rate,
    center_freq,
    bandwidth_Hz,
    gain,
    channel1,
    channel2,
    sample_format=_bladerf.Format.SC16_Q11,
):
    # ret = libbladeRF.bladerf_get_sample_rate_range(self.dev[0], ch, _range_ptr)
    # int ret = bladerf_get_sample_rate_range(dev, config.channel, &rf_range);

    if sample_format == _bladerf.Format.SC8_Q7:
        bladerf1.enable_oversample()
    else:
        bladerf1.disable_oversample()

    # activate channels
    ch1 = bladerf1.Channel(channel1)
    ch2 = bladerf1.Channel(channel2)

    # set sample rate
    ch1.sample_rate = sample_rate
    ch2.sample_rate = sample_rate

    # print("frequency_range:", ch1.frequency_range)
    # print("sample_rate_range:", ch1.sample_rate_range)
    # print("bandwidth_range:", ch1.bandwidth_range)

    ch1.frequency = center_freq
    ch2.frequency = center_freq

    # set bandwidth_Hz
    ch1.bandwidth_Hz = bandwidth_Hz
    ch2.bandwidth_Hz = bandwidth_Hz

    # output the selected sample rate and bandwidth
    print(
        f"Sample Rate (Msps): requested {sample_rate/1e6:,.2f} -> actual {ch1.sample_rate/1e6:,.2f}"
    )
    print(
        f"   Bandwidth (MHz): requested {bandwidth_Hz/1e6:,.2f} -> actual {ch1.bandwidth_Hz/1e6:,.2f}"
    )

    ch1.gain = gain
    ch2.gain = gain

    ch1.enable = True
    ch2.enable = True

    return ch1, ch2


def rx(bladerf1, num_samples, sample_format=_bladerf.Format.SC16_Q11):
    # Setup synchronous stream MIMO 2x2
    bladerf1.sync_config(
        layout=_bladerf.ChannelLayout.RX_X2,
        fmt=sample_format,
        num_buffers=32,
        buffer_size=32_768,  # must be a multiple of 1_024
        num_transfers=16,  # 4, 8, 16
        stream_timeout=1_000,
    )

    num_channels = 2

    time_swath_us = (
        num_samples * 1_000_000 / bladerf1.Channel(_bladerf.CHANNEL_RX(0)).sample_rate
    )

    if sample_format == _bladerf.Format.SC8_Q7:
        bytes_per_sample = 1
    elif sample_format == _bladerf.Format.SC16_Q11:
        bytes_per_sample = 2
    else:
        print(f"[ERROR] unknown sample format {sample_format}")
        return [], []

    rx_dtype = f"<i{bytes_per_sample}"

    total_bytes_per_sample = 2 * bytes_per_sample * num_channels  # 2 for I and Q

    buf = bytearray(num_samples * total_bytes_per_sample)

    num_samples_read = num_channels * num_samples

    # Read into buffer
    t0 = time.perf_counter_ns()
    bladerf1.sync_rx(buf, num_channels * num_samples)

    # data = np.frombuffer(buf[: len(buf) // 2], dtype=rx_dtype)
    data = np.frombuffer(buf, dtype=rx_dtype)
    t1 = time.perf_counter_ns()

    num_bytes = (
        sys.getsizeof(buf) / 2
    )  # Not sure why /2, but when i write the data out, thats what it is

    time_delta_us = (t1 - t0) / 1_000

    read_rate_Msps = num_samples_read / time_delta_us
    read_rate_MBph = (num_bytes / time_delta_us) * 60 * 60

    print(
        f"number of IQ samples read = {num_samples_read:,}| {num_bytes/1024_000:,.2f}MB, time swath: {time_swath_us:,.3f}us, rx time: {time_delta_us:,.3f}us, rate: {read_rate_Msps:,.3f}Ms/s | {read_rate_MBph:,.3f} MB/hour"
    )

    t0 = time.perf_counter_ns()
    # dump the data to a file
    data.tofile("test.bin")

    ## append to binary file
    # with open('test.bin', 'ab') as f:
    #    data.tofile(f)
    t1 = time.perf_counter_ns()

    time_delta_us = (t1 - t0) / 1_000
    write_rate_MBps = num_samples_read / time_delta_us
    print(f"data write time: {time_delta_us:,.0f}us, rate: {write_rate_MBps:,.1f}MB/s")

    t0 = time.perf_counter_ns()
    signal1_i = data[0:-3:4] / 2048  # 2048 = 2^(12-1) bits; where ADC bit number is 12b
    signal1_q = data[1:-2:4] / 2048

    signal2_i = data[2:-1:4] / 2048
    signal2_q = data[3::4] / 2048

    signal1 = signal1_i + 1j * signal1_q
    signal2 = signal2_i + 1j * signal2_q
    t1 = time.perf_counter_ns()

    time_delta_us = (t1 - t0) / 1_000
    convert_rate_Msps = num_samples_read / time_delta_us
    print(
        f"data conversion time: {time_delta_us:,.0f}us, rate: {convert_rate_Msps:,.1f}Ms/s"
    )

    print(
        f"num samples requested: {num_samples:,d}, signal1: {len(signal1):,d}, signal2: {len(signal2):,d}"
    )

    return signal1, signal2


def plot_time_freq(
    iq_signals,
    fc_MHz: float = 1_000,
    fs_MHz: float = 500,
    fft_size=None,
    n_time_samples: int = 100,
    f0_MHz=None,
    f1_MHz=None,
):
    if type(iq_signals) is not list:
        iq_signals = [iq_signals]

    if fft_size is None:
        fft_size = len(iq_signals[0])

    n_time_samples = min(n_time_samples, len(iq_signals[0]))

    ts_us = 1 / fs_MHz
    time_range_us = np.linspace(
        0,
        fft_size * ts_us,
        fft_size,
    )[:n_time_samples]

    freq_range_MHz = np.linspace(
        (fc_MHz - fs_MHz / 2),
        (fc_MHz + fs_MHz / 2),
        fft_size,
    )

    signal_count = len(iq_signals)
    _, axs = plt.subplots(nrows=signal_count, ncols=2)

    for row_ in range(signal_count):
        IQ_dBFS = 10 * np.log10(
            np.abs(np.fft.fftshift(np.fft.fft(iq_signals[row_], fft_size))) ** 2
        )

        axs[row_, 0].plot(
            time_range_us, iq_signals[row_][:n_time_samples]
        )  # time series
        axs[row_, 0].set_xlabel("Time (us)")
        axs[row_, 0].set_ylabel("Amplitude")

        axs[row_, 1].plot(freq_range_MHz, IQ_dBFS)  # FFT
        axs[row_, 1].set_xlabel("Frequency (MHz)")
        axs[row_, 1].set_ylabel("Amplitude (dBFS)")

        # show vertical lines around a frequency range
        if f0_MHz is not None and f1_MHz is not None:
            axs[row_, 1].axvline(x=f0_MHz, color="r", linestyle="dashed", linewidth=0.5)
            axs[row_, 1].axvline(x=f1_MHz, color="r", linestyle="dashed", linewidth=0.5)

    plt.show()


def fft_filtering(
    iq_signal, bpf_MHz: int = 10, fc_MHz: float = 500, fs_MHz: float = 1_000
):
    num_samples = len(iq_signal)

    freq_range_MHz = np.linspace(
        (fc_MHz - fs_MHz / 2),
        (fc_MHz + fs_MHz / 2),
        num_samples,
    )

    print(f"Freq range (MHz): {freq_range_MHz[0]:,.1f}->{freq_range_MHz[-1]:,.1f}")

    IQ_ = np.fft.fftshift(np.fft.fft(iq_signal))
    # np.fft.fftshift( np.fft.fft(iq_signal, n_fft) )
    IQ_dBFS = 10 * np.log10(np.abs(IQ_) ** 2)

    freq_bin_size_MHz = fs_MHz / len(IQ_)
    bpf_bins = bpf_MHz // freq_bin_size_MHz
    print(f"{bpf_bins}")

    frequency_mask = np.zeros(len(IQ_))

    # get the peak frequency, and filter "bpf_MHz" around that
    # peaks = [np.argmax(IQ_dBFS)]
    peaks, _ = signal.find_peaks(IQ_dBFS, distance=bpf_bins)

    for peak_idx in peaks:
        f0_MHz = freq_range_MHz[peak_idx] - bpf_MHz / 2
        f1_MHz = freq_range_MHz[peak_idx] + bpf_MHz / 2

        Istart = np.argmax(freq_range_MHz > f0_MHz)
        Iend = np.argmax(freq_range_MHz > f1_MHz)

        print(f"Peak signal at {freq_range_MHz[peak_idx]:,.1f} MHz")
        print(f"Freq filter  (MHz): {f0_MHz:,.1f}->{f1_MHz:,.1f}")
        print(f"Freq filter(index): {Istart}->{Iend}")

        frequency_mask[Istart:Iend] = 1

        # # zero everything outside that
        # IQ_[:Istart] = 0
        # IQ_[Iend:] = 0

    # element-wise multiplication
    IQ_ *= frequency_mask

    # inverse fft of the filtered signal
    iq_filtered = np.fft.ifft(np.fft.fftshift(IQ_))

    return iq_filtered, f0_MHz, f1_MHz, peaks


def test_freq_filtering():
    fs_MHz = 1020  # Sampling frequency in Hz
    fc_MHz = 0  # fs_MHz / 4
    T = 5  # Length of signal in seconds
    t = np.arange(0, T, 1 / fs_MHz)  # Time vector
    f1 = 150  # Frequency of sine wave in Hz
    f2 = 400  # Frequency of cosine wave in Hz
    a, b = 2, 5  # Constants

    print(f"Two signals at {f1}MHz and {f2}MHz")

    f_t = np.cos(2 * np.pi * f1 * t) + 1j * np.sin(2 * np.pi * f1 * t)  # 1st function
    g_t = np.cos(2 * np.pi * f2 * t) + 1j * np.sin(2 * np.pi * f2 * t)  # 2nd function
    iq_signal = a * f_t + b * g_t  # Linear combination of the functions

    # F_f_t = np.fft.fft(f_t)  # FT of the first function
    # F_g_t = np.fft.fft(g_t)  # FT of the second function
    # linear_combination_F = a * F_f_t + b * F_g_t  # Linear combination of the FTs
    # F_combination = np.fft.fft(combination)  # FT of the linear combination

    bpf_MHz = 100
    iq_filtered, f0_MHz, f1_MHz, peaks = fft_filtering(
        iq_signal, bpf_MHz, fc_MHz, fs_MHz
    )

    plot_time_freq(
        [iq_signal, iq_filtered], fc_MHz, fs_MHz, f0_MHz=f0_MHz, f1_MHz=f1_MHz
    )


## simple encoder - should work on burst DIF, but not continuous DIF
#
def rough_PDW_encoder(iq_signal, sample_time_us, toa_start_us: float = 0.0):
    smoothing_factor = 10  # envelope smoothing factor (filtering)
    num_samples = len(iq_signal)
    t_us = np.linspace(0, sample_time_us * num_samples, num_samples)

    _, axs = plt.subplots(nrows=1, ncols=1)

    ## envelope detector
    # envelope = np.sqrt(np.real(iq_signal)*np.real(iq_signal)+np.imag(iq_signal)*np.imag(iq_signal))
    envelope = np.abs(iq_signal)
    axs.plot(t_us, envelope, label="envelope")

    ## envelope smoothing (apply a moving average filter and ignore the first and last values)
    for idx in np.arange(
        smoothing_factor // 2, len(envelope) - smoothing_factor // 2 - 1
    ):
        envelope[idx] = (
            np.sum(envelope[idx : idx + smoothing_factor]) / smoothing_factor
        )
    axs.plot(t_us, envelope, label="smoothed")
    axs.legend()

    plt.show()

    ## find the peak amplitude
    peak_idx = np.argmax(envelope)
    amplitude = envelope[peak_idx]

    ## find the start of the pulse
    I_start = np.argmax(envelope[: peak_idx + 1] > (amplitude / 2))

    TOA_us = toa_start_us + I_start * sample_time_us

    ## find the end of the pulse
    I_end = np.argmax(envelope[peak_idx:] < (amplitude / 2))

    pulse_width_idx = I_end - I_start
    pulse_width_us = pulse_width_idx * sample_time_us

    ## frequency estimate
    zReal = 0.0
    zImag = 0.0
    I_ = np.real(iq_signal)
    Q_ = np.imag(iq_signal)
    for idx in np.arange(I_start, I_end):
        zReal += I_[idx + 1] * I_[idx] + Q_[idx + 1] * Q_[idx]
        zImag += Q_[idx + 1] * I_[idx] - I_[idx + 1] * Q_[idx]
    zReal /= pulse_width_idx
    zImag /= pulse_width_idx
    freq_MHz = 1_000_000 * math.atan(zImag / zReal) / (2 * np.pi)

    print(f"TOA (us): {TOA_us:,.3f}")
    print(f"amplitude: {amplitude:,.3f}")
    print(f"pulse width (us): {pulse_width_us:,.3f}")
    print(f"frequency (MHz): {freq_MHz:,.3f}")


def main(
    num_bits: int = 21,
    plot_data: bool = False,
    centre_freq=None,
    sample_rate=None,
    ADC_bits=None,
):
    real_FFT = False  # True

    if centre_freq is None:
        fc_wifi = np.append([2_407 + 5 * n for n in range(14)], 2_484) * 1e6
        # Note: channel 0 is not a valid channel, it's there for convenience

        center_freq = fc_wifi[8]  # wifi, channel 8

    print(f"[INFO] fc = {center_freq/1e6:,}MHz")

    if sample_rate is None:
        sample_rate = 61_440_000  # 6_250_000 -> 122_880_000

    bandwidth_Hz = sample_rate  # 56_000_000
    print(f"[INFO] Sample Rate: {sample_rate/1_000_000:,.3f}Msps")
    print(f"[INFO] Bandwidth: {bandwidth_Hz/1_000_000:,.3f}MHz")

    if ADC_bits is None:
        ADC_bits = 16
        sample_format = _bladerf.Format.SC16_Q11
    elif ADC_bits == 8:
        sample_format = _bladerf.Format.SC8_Q7
    else:
        sample_format = _bladerf.Format.SC16_Q11

    print(f"[INFO] ADC samples: {ADC_bits:}bits")

    try:
        devices = _bladerf.get_device_list()
        if len(devices) == 1:
            device = "{backend}:device={usb_bus}:{usb_addr}".format(
                **devices[0]._asdict()
            )
            print("[INFO] Found one bladeRF device: " + str(device))
        if len(devices) > 1:
            print("[INFO] Multiple bladeRFs detected:")
            print("\n".join([str(device_) for device_ in devices]))
    except _bladerf.BladeRFError:
        print("No bladeRF devices found.")
        shutdown(error=-1, board=None)  # chuck this in an endless loop

    # bladerf1 = _bladerf.BladeRF(devinfo=devices[0])
    bladerf1 = blade_RF(devinfo=devices[0])

    channels = rx_config(
        bladerf1,
        sample_rate=sample_rate,
        center_freq=center_freq,
        bandwidth_Hz=bandwidth_Hz,
        gain=int(0),
        channel1=_bladerf.CHANNEL_RX(0),
        channel2=_bladerf.CHANNEL_RX(1),
        sample_format=sample_format,
    )

    print(f"buffer size / requested number of samples: {int(2**num_bits):,}")

    RX1, RX2 = rx(bladerf1, num_samples=int(2**num_bits), sample_format=sample_format)

    # print(f"RX1[{len(RX1):,}]: {RX1[:10]}")
    # print(f"RX2[{len(RX2):,}]: {RX2[:10]}")

    # Create spectrogram
    ch1 = bladerf1.Channel(_bladerf.CHANNEL_RX(0))
    bandwidth_Hz = ch1.sample_rate  # ch1.bandwidth
    sampletime_us = 1_000_000 / ch1.sample_rate

    rough_PDW_encoder(RX1, sampletime_us, 0.0)

    fft_size = 2048

    num_rows = len(RX1) // fft_size  # // is an integer division which rounds down

    half_fft = fft_size // 2

    spectrogram = {0: [], 1: []}
    if real_FFT:
        spectrogram[0] = np.zeros((num_rows, 1 + half_fft))
        spectrogram[1] = np.zeros((num_rows, 1 + half_fft))
    else:
        spectrogram[0] = np.zeros((num_rows, fft_size))
        spectrogram[1] = np.zeros((num_rows, fft_size))

    for i in range(num_rows):
        if real_FFT:
            spectrogram[0][i, :] = 10 * np.log10(
                np.abs(np.fft.rfft(np.real(RX1[i * fft_size : (i + 1) * fft_size])))
                ** 2
            )
            spectrogram[1][i, :] = 10 * np.log10(
                np.abs(np.fft.rfft(np.real(RX2[i * fft_size : (i + 1) * fft_size])))
                ** 2
            )
        else:
            spectrogram[0][i, :] = 10 * np.log10(
                np.abs(
                    np.fft.fftshift(np.fft.fft(RX1[i * fft_size : (i + 1) * fft_size]))
                )
                ** 2
            )
            spectrogram[1][i, :] = 10 * np.log10(
                np.abs(
                    np.fft.fftshift(np.fft.fft(RX2[i * fft_size : (i + 1) * fft_size]))
                )
                ** 2
            )

    extent = []
    y0 = np.average(spectrogram[0][:, :], axis=0)
    y1 = np.average(spectrogram[1][:, :], axis=0)
    if real_FFT:
        extent = [
            center_freq / 1e6,
            (center_freq + bandwidth_Hz / 2) / 1e6,
            len(RX2) * sampletime_us / 1_000,
            0,
        ]
        freq_range = np.linspace(
            center_freq / 1e6, (center_freq + bandwidth_Hz / 2) / 1e6, 1 + half_fft
        )
    else:
        extent = [
            (center_freq - bandwidth_Hz / 2) / 1e6,
            (center_freq + bandwidth_Hz / 2) / 1e6,
            len(RX2) * sampletime_us / 1_000,
            0,
        ]
        freq_range = np.linspace(
            (center_freq - bandwidth_Hz / 2) / 1e6,
            (center_freq + bandwidth_Hz / 2) / 1e6,
            fft_size,
        )

    spacing_1MHz = int(0.90 * fft_size // (bandwidth_Hz / 1e6))
    # set it to 90% of the 1MHz bin spacing, allowing a minimum 1MHz between emitters
    threshold_height = 10
    num_peaks = 2

    y0_peaks, _ = signal.find_peaks(y0, distance=spacing_1MHz, height=threshold_height)
    y1_peaks, _ = signal.find_peaks(y1, distance=spacing_1MHz, height=threshold_height)

    # sort the peaks, and grab the highest <num_peaks> peaks
    y0_idx = np.flip(np.argsort(y0[y0_peaks]))[:num_peaks]
    y1_idx = np.flip(np.argsort(y1[y1_peaks]))[:num_peaks]
    y0_peaks = y0_peaks[y0_idx]
    y1_peaks = y1_peaks[y1_idx]

    print("Emitter peak locations")
    print(f"ch0: {vector_string(freq_range[y0_peaks])} MHz")
    print(f"ch1: {vector_string(freq_range[y1_peaks])} MHz")

    ## PLOT
    if plot_data == 1:
        samples_to_plot = 1000
        num_samples = len(RX1)

        t_us = np.linspace(0, sampletime_us * num_samples, len(RX1))

        _ = plt.figure()
        ax = dict()
        #                           shape=(row,col), loc=(row,col)
        ax["FFT_1"] = plt.subplot2grid(shape=(5, 2), loc=(0, 0))
        ax["FFT_2"] = plt.subplot2grid(shape=(5, 2), loc=(0, 1))
        ax["SPEC_1"] = plt.subplot2grid(shape=(5, 2), loc=(1, 0))
        ax["SPEC_2"] = plt.subplot2grid(shape=(5, 2), loc=(1, 1))
        ax["EYE_1"] = plt.subplot2grid(shape=(5, 2), loc=(4, 0))
        ax["EYE_2"] = plt.subplot2grid(shape=(5, 2), loc=(4, 1))
        ax["IQ_1"] = plt.subplot2grid(shape=(5, 2), loc=(2, 0), colspan=2)
        ax["IQ_2"] = plt.subplot2grid(shape=(5, 2), loc=(3, 0), colspan=2)

        # _, axs = plt.subplots(nrows=4, ncols=2)

        ax["FFT_1"].plot(freq_range, y0)
        ax["FFT_1"].plot(freq_range[y0_peaks], y0[y0_peaks], "x")
        ax["FFT_1"].set_xlim(extent[0], extent[1])
        ax["FFT_1"].set_title(f"ch0: {vector_string(freq_range[y0_peaks])} MHz")
        ax["FFT_1"].set_xlabel("Frequency (MHz)")
        ax["FFT_1"].set_ylabel("Amplitude (dB)")

        ax["FFT_2"].plot(freq_range, y1)
        ax["FFT_2"].plot(freq_range[y1_peaks], y1[y1_peaks], "x")
        ax["FFT_2"].set_xlim(extent[0], extent[1])
        ax["FFT_2"].set_title(f"ch1: {vector_string(freq_range[y1_peaks])} MHz")
        ax["FFT_2"].set_xlabel("Frequency (MHz)")
        ax["FFT_2"].set_ylabel("Amplitude (dB)")

        ax["SPEC_1"].imshow(spectrogram[0], aspect="auto", extent=extent)
        ax["SPEC_1"].set_xlabel("Frequency (MHz)")
        ax["SPEC_1"].set_ylabel("Time (ms)")

        ax["SPEC_2"].imshow(spectrogram[1], aspect="auto", extent=extent)
        ax["SPEC_2"].set_xlabel("Frequency (MHz)")
        ax["SPEC_2"].set_ylabel("Time (ms)")

        ax["EYE_1"].scatter(
            np.real(RX1[(num_samples // 2) : (num_samples // 2) + samples_to_plot]),
            np.imag(RX1[(num_samples // 2) : (num_samples // 2) + samples_to_plot]),
        )
        ax["EYE_1"].set_xlabel("Real")
        ax["EYE_1"].set_ylabel("Imag")
        ax["EYE_1"].set_aspect("equal", adjustable="box")

        ax["EYE_2"].scatter(
            np.real(RX2[(num_samples // 2) : (num_samples // 2) + samples_to_plot]),
            np.imag(RX2[(num_samples // 2) : (num_samples // 2) + samples_to_plot]),
        )
        ax["EYE_2"].set_xlabel("Real")
        ax["EYE_2"].set_ylabel("Imag")
        ax["EYE_2"].set_aspect("equal", adjustable="box")

        ax["IQ_1"].plot(
            t_us[:samples_to_plot], np.real(RX1)[:samples_to_plot], label="RX1"
        )
        ax["IQ_1"].plot(
            t_us[:samples_to_plot], np.real(RX2)[:samples_to_plot], label="RX2"
        )
        ax["IQ_1"].set_xlabel("Time (us)")
        ax["IQ_1"].set_ylabel("Amplitude")
        ax["IQ_1"].set_title("Inphase")
        ax["IQ_1"].legend()

        ax["IQ_2"].plot(
            t_us[(num_samples // 2) : (num_samples // 2) + samples_to_plot],
            np.real(RX1)[(num_samples // 2) : (num_samples // 2) + samples_to_plot],
            label="RX1",
        )
        ax["IQ_2"].plot(
            t_us[(num_samples // 2) : (num_samples // 2) + samples_to_plot],
            np.real(RX2)[(num_samples // 2) : (num_samples // 2) + samples_to_plot],
            label="RX2",
        )
        ax["IQ_2"].set_xlabel("Time (us)")
        ax["IQ_2"].set_ylabel("Amplitude")
        ax["IQ_2"].set_title("Quadrature")
        ax["IQ_2"].legend()

        plt.show()

    # close the board and exit without error
    shutdown(error=0, board=bladerf1)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    fs_Hz = int(args["fs"] * 1_000_000)

    if args["fc"] is not None:
        fc_Hz = int(args["fc"] * 1_000_000)
    else:
        fc_Hz = None

    # test_freq_filtering()

    main(args["buffer_bits"], args["plot"], fc_Hz, fs_Hz, args["adc_bits"])
