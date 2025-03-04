#!/usr/bin/env python3
import os
import time
import atexit

from codetiming import Timer

import faulthandler

faulthandler.enable()
# /usr/local/lib/python3.9/dist-packages/iio.py -> error line 995 in __del__
#
# need these calls in order. See https://github.com/analogdevicesinc/libiio/issues/1145
#
# # This order crashes
# iio_buffer_cancel(buffer);
# iio_channels_mask_destroy(mask);
# iio_buffer_destroy(buffer);
#
# # This order doesn't crash
# iio_buffer_cancel(buffer);
# iio_buffer_destroy(buffer);
# iio_channels_mask_destroy(mask);
#

import numpy as np

from phaser_CN0566 import phaser_CN0566
from phaser_PlutoSDR import phaser_PlutoSDR
from phaser_IO import phaser_SigMF, phaser_IO
from phaser_utils import *


# phaser class
# @author: Mark Cooke
class phaser:
    def __init__(self):
        self.fs_Hz = 30_000_000
        self.fc_Hz = nominal_hb100_freq_Hz
        self.rx_lo_Hz = Phaser_LO_HIGH
        self.plutoSDR_1 = None
        self.plutoSDR_2 = None
        self.cn0566_1 = None
        self.cn0566_2 = None
        self.sigmf_1_CH0 = phaser_SigMF()
        self.sigmf_1_CH1 = phaser_SigMF()
        self.sigmf_2_CH0 = phaser_SigMF()
        self.sigmf_2_CH1 = phaser_SigMF()
        self.two_receivers = False
        self.stop_plotting = False

    def setup(
        self,
        fs_Hz: int = 30_000_000,
        fc_Hz: float = nominal_hb100_freq_Hz,
        rx_lo_Hz: float = Phaser_LO_HIGH,
        two_receivers: bool = False,
    ):
        # First try to connect to a locally connected CN0566. On success, connect,
        # on failure, connect to remote CN0566
        self.fs_Hz = fs_Hz
        self.fc_Hz = fc_Hz
        self.rx_lo_Hz = rx_lo_Hz
        self.two_receivers = two_receivers

        self.plutoSDR_1 = phaser_PlutoSDR(
            self.fs_Hz, self.rx_lo_Hz, PlutoSDR_ip="192.168.2.11"
        )
        self.cn0566_1 = phaser_CN0566(
            self.fc_Hz, self.rx_lo_Hz, CN0566_ip="ip:phaser.local"
        )
        if self.two_receivers:
            self.plutoSDR_2 = phaser_PlutoSDR(
                self.fs_Hz, self.rx_lo_Hz, PlutoSDR_ip="192.168.2.12"
            )
            self.cn0566_2 = phaser_CN0566(
                self.fc_Hz, self.rx_lo_Hz, CN0566_ip="ip:phaser.local"
            )
        time.sleep(0.5)  # recommended by Analog Devices

    def set_rx_buffer_size(self, rx_buffer_size: int = 1024):
        self.plutoSDR_1.rx_buffer_size=rx_buffer_size
        if self.two_receivers:
            self.plutoSDR_2.rx_buffer_size=rx_buffer_size

    def set_rx_bandwidth_Hz(self, rx_bandwidth_Hz: int = 10_000_000):
        self.plutoSDR_1.rx_bandwidth_Hz=rx_bandwidth_Hz
        if self.two_receivers:
            self.plutoSDR_2.rx_bandwidth_Hz=rx_bandwidth_Hz

    def set_sample_frequency_Hz(self, fs_Hz: int = 30_000_000):
        if fs_Hz != self.fs_Hz:
            self.fs_Hz = fs_Hz
            self.plutoSDR_1.fs_Hz=fs_Hz
            if self.two_receivers:
                self.plutoSDR_2.fs_Hz=fs_Hz

    def set_frequency_Hz(self, fc_Hz: float = nominal_hb100_freq_Hz):
        if fc_Hz != self.fc_Hz:
            self.fc_Hz = fc_Hz
            # note: change the plutoSDR LO if required
            new_lo = self.cn0566_1.set_frequency_Hz(self.fc_Hz)
            if new_lo is not None:
                self.rx_lo_Hz = new_lo
                self.plutoSDR_1.fc_Hz=self.rx_lo_Hz

            if self.two_receivers:
                new_lo = self.cn0566_2.set_frequency_Hz(self.fc_Hz)
                if new_lo is not None:
                    self.rx_lo_Hz = new_lo
                    self.plutoSDR_2.fc_Hz=self.rx_lo_Hz

    def plot(self, fc_Hz: float = nominal_hb100_freq_Hz):
        self.plutoSDR_1.rx_gain=63  # 60 = 1500/2000

        self.set_frequency_Hz(fc_Hz)
        data = self.plutoSDR_1.read()

        # Take FFT
        PSD0 = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data[0]))) ** 2)
        PSD1 = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data[1]))) ** 2)
        f = np.linspace(-self.fs_Hz / 2, self.fs_Hz / 2, len(data[0]))

        # Time plot helps us check that we see the emitter and that we're not saturated (ie gain isnt too high)
        plt.subplot(2, 1, 1)
        plt.plot(data[0].real, label="ch0")  # Only plot real part
        plt.plot(data[1].real, label="ch1")
        plt.legend()
        plt.xlabel("Data Point")
        plt.ylabel("ADC output")

        # PSDs show where the emitter is and verify both channels are working
        plt.subplot(2, 1, 2)
        plt.plot(f / 1e6, PSD0, label="ch0")
        plt.plot(f / 1e6, PSD1, label="ch1")
        plt.legend()
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Signal Strength [dB]")
        plt.tight_layout()
        plt.show()

    @Timer(name="record", text="{name}: {milliseconds:,.3f}ms")
    def record(
        self,
        fc_Hz: float = nominal_hb100_freq_Hz,
        fs_Hz: int = 30_000_000,
        rx_buffer_size: int = 2_097_144,
        num_buffers: int = 1,
    ):
        if fc_Hz != self.fc_Hz:
            self.set_frequency_Hz(fc_Hz)
        if fs_Hz != self.fs_Hz:
            self.set_sample_frequency_Hz(fs_Hz)

        self.set_rx_buffer_size(rx_buffer_size)

        # print(f"Default Buffer Size: {io.DEFAULT_BUFFER_SIZE:,}")
        self.sigmf_1_CH0.open(0, fs_Hz, fc_Hz)
        self.sigmf_1_CH1.open(1, fs_Hz, fc_Hz)
        for n in range(num_buffers):
            data = self.plutoSDR_1.read()
            self.sigmf_1_CH0.write(data)
            self.sigmf_1_CH1.write(data)
        self.sigmf_1_CH0.close()
        self.sigmf_1_CH1.close()

        print(f"rx buffer: {rx_buffer_size:,.0f}", end=" ")

        # phaser_IO.delete(self.base_filename + ".sigmf-data")
        # phaser_IO.delete(self.base_filename + ".sigmf-meta")

    # @author: Mark Cooke
    def sweep(
        self, f_start: float = 8.0e9, f_stop: float = 10.6e9, interactive: bool = False
    ):
        # Set up range of frequencies to sweep. Sample rate is set to 30Msps,
        # for a total of 30MHz of bandwidth (quadrature sampling)
        # Filter is 20MHz LTE, so you get a bit less than 20MHz of usable
        # bandwidth. Set step size to something less than 20MHz to ensure
        # complete coverage.

        N_FFT = 1024  # from the FFT in spec_est()

        # sampling bandwidth is 30MHz wide
        freqs = np.linspace(-self.fs_Hz / 2, self.fs_Hz / 2, N_FFT)

        f_diff = np.diff(freqs)[1]
        tune_inc = int(10_000_000 // f_diff)
        f_step = float(tune_inc) * f_diff  # make f_step a multiple of the fft steps

        # window is 20MHz wide
        w0_freq = freqs[freqs >= -f_step][0]
        w1_freq = freqs[freqs >= f_step][0]
        window_inc = np.argmin(abs(freqs - w0_freq))
        window_length = int((w1_freq - w0_freq) // f_diff)
        half_window_length = window_length // 2

        # print(f"{f_diff/1e3:,.3f}kHz, {f_step/1e6:,.3f}MHz, {tune_inc}, {window_inc}, {window_length}")
        tune_freq_range = np.arange(f_start, f_stop, f_step)  # 10MHz steps
        # tune_freq_range = np.arange(f_start, f_stop, 2 * f_step)  # 20MHz steps

        full_freq_range = np.arange(
            tune_freq_range[0] + w0_freq, tune_freq_range[-1] + w1_freq, f_diff
        )
        full_amp_range = np.array([-np.Inf] * len(full_freq_range))

        idx = 0
        for freq in tune_freq_range:  # range(int(f_start), int(f_stop), int(f_step)):
            # print(f"frequency: {freq/1e6:,.3f}MHz")
            self.set_frequency_Hz(freq)

            data = self.plutoSDR_1.read()  # 3.982 ms

            data_sum = data[0] + data[1]
            #    max0 = np.max(abs(data[0]))
            #    max1 = np.max(abs(data[1]))
            #    print("max signals: ", max0, max1)
            ampl, freqs = spectrum_estimate(
                data_sum, self.fs_Hz, ref=2 ^ 12
            )  # 2.375 ms

            # only look at the inner 20MHz
            window = ampl[window_inc : window_inc + window_length + 1]
            swath = full_amp_range[idx : idx + window_length + 1]

            dwell = np.maximum(window, swath)

            full_amp_range[idx : idx + window_length + 1] = dwell

            idx += tune_inc

            # print(f"{freq/1e9:,.3f}GHz:[{full_freq_range[idx]/1e9:,.3f}GHz|{full_freq_range[idx+half_window_length]/1e9:,.3f}GHz|{full_freq_range[idx+window_length]/1e9:,.3f}GHz]")

            time.sleep(0.001)  # 100.164 ms

        full_freq_range /= 1e9  # Hz -> GHz

        # estimate the noise floor and set the max value to 6dB above that
        signal_threshold = 6 + noise_estimate(
            full_amp_range, N_CUT=16, window_length=512
        )

        # then set the noise floor as the average of all values down to 6dB below this value
        max_noise = signal_threshold.max()
        noise_floor = max_noise - 10
        max_signal = full_amp_range.max()

        print(f"max signal = {max_signal:.2f}dBFS")
        print(f"max noise = {max_noise:.2f}dBFS")
        print(f"noise floor = {noise_floor:.2f}dBFS")

        # full_amp_range = np.clip(full_amp_range, noise_floor, None)

        if max_signal > max_noise + 10:
            peak_index = np.argmax(full_amp_range)
            peak_freq = full_freq_range[peak_index]
            self.fc_Hz = peak_freq
            print(f"Peak frequency found at {full_freq_range[peak_index]*1e3:,.3f}MHz.")
            if interactive:
                plot_spectrum(full_freq_range, full_amp_range, signal_threshold)

                prompt = input("Save cal file? ([y] or n)")
                if prompt.upper() != "N":
                    save_pkl(peak_freq * 1e9)
            else:
                save_pkl(peak_freq * 1e9)
        elif interactive:
            prompt = input(
                "Emitter not found.\nWould you like to plot the received spectrum? (y or [n])"
            )
            if prompt.upper() == "Y":
                plot_spectrum(full_freq_range, full_amp_range, signal_threshold)

    def on_mpl_exit(self, event):
        if event.key == "q":
            self.stop_plotting = True

    def freq_tracker(
        self, fc_MHz: int = 10_000, BW_MHz: int = 300, update_time_s: float = 1.0
    ):
        N_FFT = 1024  # from the FFT in spec_est()
        freqs_Hz = np.linspace(-self.fs_Hz / 2, self.fs_Hz / 2, N_FFT)

        # make f_step a multiple of the fft steps
        f_diff = np.diff(freqs_Hz)[1]
        tune_inc = int(10_000_000 // f_diff)
        f_step = float(tune_inc) * f_diff

        w0_freq = freqs_Hz[freqs_Hz >= -f_step][0]
        w1_freq = freqs_Hz[freqs_Hz >= f_step][0]
        window_inc = np.argmin(abs(freqs_Hz - w0_freq))
        window_length = int((w1_freq - w0_freq) // f_diff)
        half_window_length = window_length // 2

        ## setup plot
        plt.ion()
        fig = plt.figure(1)
        fig.canvas.mpl_connect("key_press_event", self.on_mpl_exit)

        freq_centre_MHz = fc_MHz

        freqs = []
        amps = []
        snrs = []

        print(f"centre frequency: {freq_centre_MHz:,.0f}MHz")
        while True:
            if self.stop_plotting:
                break

            ## update freq_centre_MHz
            f_start = float(freq_centre_MHz - (BW_MHz // 2))
            f_stop = float(freq_centre_MHz + (BW_MHz // 2))
            tune_freq_range_MHz = np.arange(f_start, f_stop, float(f_step / 1.0e6))

            # print(f"Tune freq:{tune_freq_range_MHz[0]:,.0f}->{tune_freq_range_MHz[-1]:,.0f}MHz")

            full_freq_range_MHz = np.arange(
                tune_freq_range_MHz[0] + (w0_freq / 1e6),
                tune_freq_range_MHz[-1] + (w1_freq / 1e6),
                (f_diff / 1e6),
            )
            full_amp_range = np.array([-np.Inf] * len(full_freq_range_MHz))

            idx = 0
            for (
                freq_MHz
            ) in tune_freq_range_MHz:  # range(int(f_start), int(f_stop), int(f_step)):
                # print(f"frequency: {freq_MHz:,.0f}MHz")
                # self.cn0566_1.set_frequency_Hz(freq)  # 1.743 ms
                self.set_frequency_Hz(freq_MHz * 1e6)
                time.sleep(
                    0.08
                )  # 0.06s is the threshold for accurately measuring CW signals

                data = self.plutoSDR_1.read()  # 3.982 ms

                data_sum = data[0] + data[1]
                #    max0 = np.max(abs(data[0]))
                #    max1 = np.max(abs(data[1]))
                #    print("max signals: ", max0, max1)
                ampl, _ = spectrum_estimate(
                    data_sum, self.fs_Hz, ref=2 ^ 12
                )  # 2.375 ms

                window = ampl[window_inc : window_inc + window_length + 1]
                swath = full_amp_range[idx : idx + window_length + 1]

                dwell = np.maximum(window, swath)

                full_amp_range[idx : idx + window_length + 1] = dwell

                idx += tune_inc

                # print(f"{freq/1e9:,.3f}GHz:[{full_freq_range_MHz[idx]/1e9:,.3f}GHz|{full_freq_range_MHz[idx+half_window_length]/1e9:,.3f}GHz|{full_freq_range_MHz[idx+window_length]/1e9:,.3f}GHz]")

            # estimate the noise floor and set the max value to 6dB above that
            signal_threshold = 6 + noise_estimate(
                full_amp_range, N_CUT=16, window_length=512
            )

            SNR_dBFS = full_amp_range - signal_threshold

            if SNR_dBFS.max() > 10:
                peak_index = np.argmax(SNR_dBFS)

                # print(f"peak centre frequency: {full_freq_range_MHz[peak_index]:,.0f}MHz")
                # change the centre frequency if the difference is greater than 10MHz
                if abs(freq_centre_MHz - full_freq_range_MHz[peak_index]) > 10:
                    freq_centre_MHz = full_freq_range_MHz[peak_index]
                    if len(freqs) > 0:
                        print(freqs)
                        print(
                            f"[{len(freqs)}] collects average: Fc {np.average(freqs):,.3f}MHz, Amp {np.average(amps):,.3f}dBFS, SNR {np.average(snrs):,.3f}dB"
                        )
                    print(f"new centre frequency: {freq_centre_MHz:,.0f}MHz")
                    freqs = []
                    amps = []
                    snrs = []

                    self.fc_Hz = freq_centre_MHz * 1e6
                else:
                    freqs += [full_freq_range_MHz[peak_index]]
                    amps += [full_amp_range[peak_index]]
                    snrs += [SNR_dBFS[peak_index]]

            plt.clf()
            plt.xlabel("Frequency [MHz]")
            plt.ylabel("Signal Strength [dBFS]")
            plt.plot(full_freq_range_MHz, full_amp_range)
            plt.plot(full_freq_range_MHz, signal_threshold)
            plt.draw()
            plt.pause(0.00001)
            # plt.clf()
            # time.sleep(1)

    def find_peak_bin(self, cn0566=None):
        if (cn0566 is None) or (cn0566 == self.cn0566_1):
            cn0566 = self.cn0566_1
            sdr = self.plutoSDR_1
        else:
            cn0566 = self.cn0566_2
            sdr = self.plutoSDR_2

        win = np.blackman(sdr.plutoSDR.rx_buffer_size)
        # First, locate fundamental.
        cn0566.cn0566.set_all_gain(127)
        cn0566.cn0566.set_beam_phase_diff(0.0)
        data = sdr.read()  # read a buffer of data
        y_sum = (data[0] + data[1]) * win
        s_sum = np.fft.fftshift(np.absolute(np.fft.fft(y_sum)))
        return np.argmax(s_sum)

    def channel_calibration(self, cn0566=None, fc_Hz=None, verbose: bool = False):
        """ " Do this BEFORE gain_calibration.
        Performs calibration between the two ADAR1000 channels. Accounts for all
        sources of mismatch between the two channels: ADAR1000s, mixers, and
        the SDR (Pluto) inputs."""
        if (cn0566 is None) or (cn0566 == self.cn0566_1):
            cn0566 = self.cn0566_1
        else:
            cn0566 = self.cn0566_2

        if (fc_Hz is not None) and (fc_Hz != self.fc_Hz):
            self.set_frequency_Hz(fc_Hz)

        peak_bin = self.find_peak_bin()
        channel_levels, _ = self.measure_channel_gains(peak_bin, cn0566, verbose=False)
        ch_mismatch = 20.0 * np.log10(channel_levels[0] / channel_levels[1])
        if verbose is True:
            print("channel mismatch: ", ch_mismatch, " dB")
        if ch_mismatch > 0:  # Channel 0 higher, boost ch1:
            cn0566.cn0566.ccal = [0.0, ch_mismatch]
        else:  # Channel 1 higher, boost ch0:
            cn0566.cn0566.ccal = [-ch_mismatch, 0.0]
        pass

    def measure_channel_gains(
        self, peak_bin, cn0566=None, verbose=False
    ):  # Default to central element
        """Calculate all the values required to do different plots. It method calls set_beam_phase_diff and
        sets the Phases of all channel. All the math is done here.
        parameters:
            gcal_element: type=int
                        If gain calibration is taking place, it indicates element number whose gain calibration is
                        is currently taking place
            cal_element: type=int
                        If Phase calibration is taking place, it indicates element number whose phase calibration is
                        is currently taking place
            peak_bin: type=int
                        Peak bin to examine around for amplitude
        """
        if (cn0566 is None) or (cn0566 == self.cn0566_1):
            cn0566 = self.cn0566_1
            sdr = self.plutoSDR_1
        else:
            cn0566 = self.cn0566_2
            sdr = self.plutoSDR_2

        width = 10  # Bins around fundamental to sum
        win = signal.windows.flattop(sdr.plutoSDR.rx_buffer_size)
        win /= np.average(np.abs(win))  # Normalize to unity gain
        plot_data = []
        channel_level = []
        # cn0566.cn0566.set_rx_hardwaregain(6, False)
        sdr.set_rx_gain(63)
        for channel in range(0, 2):
            # Start with sdr CH0 elements
            cn0566.cn0566.set_all_gain(
                0, apply_cal=False
            )  # Start with all gains set to zero
            cn0566.cn0566.set_chan_gain(
                (1 - channel) * 4 + 0,
                127,
                apply_cal=False,  # 1-channel because wonky channel mapping!!
            )  # Set element to max
            cn0566.cn0566.set_chan_gain(
                (1 - channel) * 4 + 1, 127, apply_cal=False
            )  # Set element to max
            cn0566.cn0566.set_chan_gain(
                (1 - channel) * 4 + 2, 127, apply_cal=False
            )  # Set element to max
            cn0566.cn0566.set_chan_gain(
                (1 - channel) * 4 + 3, 127, apply_cal=False
            )  # Set element to max

            # todo - remove when driver fixed to compensate for ADAR1000 quirk
            time.sleep(1.0)

            if verbose:
                print("measuring channel ", channel)

            total_sum = 0
            # win = np.blackman(sdr.plutoSDR.rx_buffer_size)

            spectrum = np.zeros(sdr.plutoSDR.rx_buffer_size)

            for count in range(
                0, cn0566.cn0566.Averages
            ):  # repeatsnip loop and average the results
                data = sdr.read()  # todo - remove once confirmed no flushing necessary
                data = sdr.read()  # read a buffer of data
                y_sum = (data[0] + data[1]) * win

                s_sum = np.fft.fftshift(np.absolute(np.fft.fft(y_sum)))
                spectrum += s_sum

                # Look for peak value within window around fundamental (reject interferers)
                s_mag_sum = np.max(s_sum[peak_bin - width : peak_bin + width])
                total_sum += s_mag_sum

            spectrum /= cn0566.cn0566.Averages * sdr.plutoSDR.rx_buffer_size
            PeakValue_sum = total_sum / (
                cn0566.cn0566.Averages * sdr.plutoSDR.rx_buffer_size
            )
            plot_data.append(spectrum)
            channel_level.append(PeakValue_sum)

        return channel_level, plot_data


def test_buffer_size(my_phaser):
    rx_buffer_range = [1024 * 2 ** (n) for n in range(12)]
    rx_buffer_range[-1] = 2_097_144  # clip to the largest buffer size
    #     1,024:    10,717.950us = 10.467us/sample, max sample rate 0.096Msps
    #     2,048:    11,681.369us =  5.704us/sample, max sample rate 0.175Msps
    #     4,096:    13,274.672us =  3.241us/sample, max sample rate 0.309Msps
    #     8,192:    16,283.735us =  1.988us/sample, max sample rate 0.503Msps
    #    16,384:    25,210.135us =  1.539us/sample, max sample rate 0.650Msps
    #    32,768:    53,312.008us =  1.627us/sample, max sample rate 0.615Msps
    #    65,536:   112,330.682us =  1.714us/sample, max sample rate 0.583Msps
    #   131,072:   204,608.837us =  1.561us/sample, max sample rate 0.641Msps
    #   262,144:   359,953.976us =  1.373us/sample, max sample rate 0.728Msps
    #   524,288:   482,487.844us =  0.920us/sample, max sample rate 1.087Msps
    # 1,048,576:   888,097.655us =  0.847us/sample, max sample rate 1.181Msps
    # 2,097,144: 1,654,651.676us =  0.789us/sample, max sample rate 1.267Msps

    for rx_buff_size in rx_buffer_range:
        # set the buffer size
        my_phaser.plutoSDR_1.rx_buffer_size=rx_buff_size
        # my_phaser.plutoSDR_1.plutoSDR.rx_buffer_size = int(rx_buff_size)

        # time the receive
        t0 = time.perf_counter_ns()
        data = my_phaser.plutoSDR_1.read()
        delta_us = (time.perf_counter_ns() - t0) / 1e3
        max_sample_rate_Msps = (
            np.shape(data)[1] / delta_us
        )  # in mega samples per second
        print(
            f"{np.shape(data)[1]:9,d}: {delta_us:13,.3f}us = {1/max_sample_rate_Msps:6,.3f}us/sample, max sample rate {max_sample_rate_Msps:2.3f}Msps"
        )


if __name__ == "__main__":
    # plt.ion()
    # for i in range(50):
    #     y = np.random.random([10, 1])
    #     plt.plot(y)
    #     plt.draw()
    #     plt.pause(0.0001)
    #     plt.clf()

    my_phaser = phaser()
    my_phaser.setup()

    # test the tuning range
    # for freq_MHz in range(10_600, 8_000, -100):
    #     print(f"set fc_MHz -> {freq_MHz:,.0f}")
    #     my_phaser.set_frequency_Hz(freq_MHz * 1e6)

    # channel calibration
    my_phaser.set_frequency_Hz(9_400e6)
    # my_phaser.channel_calibration(verbose=True)
    # 0.5192586079313221 dB @ 9.4GHz, rx_gain=6dB
    # my_phaser.cn0566_1.cn0566.ccal = [0.0, 0.5192586079313221]
    # my_phaser.plutoSDR_1.channel_cal = [1.2807293215820739, 0.0]
    my_phaser.plutoSDR_1.channel_cal = [2.0, 0.0]

    my_phaser.plutoSDR_1.rx_gain=63  # 60 = 1500/2000

    test_buffer_size(my_phaser)

    # my_phaser.plot(9_400e6)

    # my_phaser.freq_tracker(9_400)

    # my_phaser.sweep(9.7e9, 10.6e9, interactive=True)

    # from line_profiler import LineProfiler

    # lp = LineProfiler()
    # lp_wrapper = lp(my_phaser.sweep)
    # lp_wrapper(10e9, 10.6e9)
    # lp.print_stats()

    print("[INFO] end")
