#!/usr/bin/env python3
import os
import time
import atexit

#from codetiming import Timer

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
from phaser_BladeRF import phaser_BladeRF
from phaser_IO import phaser_SigMF, phaser_IO
from phaser_utils import *

# phaser class
# @author: Mark Cooke
class phaser:
    def __init__(self):
        self.fs_Hz = 30_000_000
        self.fc_Hz = nominal_hb100_freq_Hz
        self.rx_lo_Hz = Phaser_LO_HIGH
        self.SDR_1 = None
        self.SDR_2 = None
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

        self.SDR_1 = phaser_BladeRF(self.fs_Hz, self.rx_lo_Hz, DeviceString="6b5d") # Note: change
        if self.SDR_1.sdr is None:
            self.SDR_1 = phaser_PlutoSDR(self.fs_Hz, self.rx_lo_Hz, DeviceString="192.168.2.11")
        self.cn0566_1 = phaser_CN0566(
            self.fc_Hz, self.rx_lo_Hz, DeviceString="ip:phaser.local"
        )
        if self.two_receivers:
            self.SDR_2 = phaser_BladeRF(self.fs_Hz, self.rx_lo_Hz, DeviceString="6b5d") # Note: change
            if self.SDR_2.sdr is None:
                self.SDR_2 = phaser_PlutoSDR(self.fs_Hz, self.rx_lo_Hz, DeviceString="192.168.2.12")
            self.cn0566_2 = phaser_CN0566(
                self.fc_Hz, self.rx_lo_Hz, DeviceString="ip:phaser.local"
            )
        time.sleep(0.5)  # recommended by Analog Devices

    def set_rx_buffer_size(self, rx_buffer_size: int = 1024):
        self.SDR_1.rx_buffer_size=rx_buffer_size
        if self.two_receivers:
            self.SDR_2.rx_buffer_size=rx_buffer_size

    def set_rx_bandwidth_Hz(self, rx_bandwidth_Hz: int = 10_000_000):
        self.SDR_1.rx_bandwidth_Hz=rx_bandwidth_Hz
        if self.two_receivers:
            self.SDR_2.rx_bandwidth_Hz=rx_bandwidth_Hz

    def set_sample_frequency_Hz(self, fs_Hz: int = 30_000_000):
        if fs_Hz != self.fs_Hz:
            self.fs_Hz = fs_Hz
            self.SDR_1.fs_Hz=fs_Hz
            if self.two_receivers:
                self.SDR_2.fs_Hz=fs_Hz

    def set_frequency_Hz(self, fc_Hz: float = nominal_hb100_freq_Hz):
        if fc_Hz != self.fc_Hz:
            self.fc_Hz = fc_Hz
            # note: change the plutoSDR LO if required
            new_lo = self.cn0566_1.set_frequency_Hz(self.fc_Hz)
            if new_lo is not None:
                self.rx_lo_Hz = new_lo
                self.SDR_1.fc_Hz=self.rx_lo_Hz

            if self.two_receivers:
                new_lo = self.cn0566_2.set_frequency_Hz(self.fc_Hz)
                if new_lo is not None:
                    self.rx_lo_Hz = new_lo
                    self.SDR_2.fc_Hz=self.rx_lo_Hz

    def plot(self, fc_Hz: float = nominal_hb100_freq_Hz):
        self.SDR_1.rx_gain=63  # 60 = 1500/2000

        self.set_frequency_Hz(fc_Hz)
        data = self.SDR_1.read()

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

    # @Timer(name="record", text="{name}: {milliseconds:,.3f}ms")
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
            data = self.SDR_1.read()
            self.sigmf_1_CH0.write(data)
            self.sigmf_1_CH1.write(data)
        self.sigmf_1_CH0.close()
        self.sigmf_1_CH1.close()

        print(f"rx buffer: {rx_buffer_size:,.0f}", end=" ")

        # phaser_IO.delete(self.base_filename + ".sigmf-data")
        # phaser_IO.delete(self.base_filename + ".sigmf-meta")

    def find_peak(self, f_start: float = 8.0e9, f_stop: float = 10.6e9):
        return self.sweep(f_start, f_stop, interactive = False)

    # @author: Mark Cooke
    def sweep(
        self, f_start: float = 8.0e9, f_stop: float = 10.6e9, interactive: bool = False
    ):
        # Set up range of frequencies to sweep. Sample rate is set to 30Msps,
        # for a total of 30MHz of bandwidth (quadrature sampling)
        # Filter is 20MHz LTE, so you get a bit less than 20MHz of usable
        # bandwidth. Set step size to something less than 20MHz to ensure
        # complete coverage.
        
        peak_freq = None

        N_FFT = 1024  # from the FFT in spec_est()

        # sampling bandwidth is 30MHz wide
        freqs = np.linspace(-self.fs_Hz / 2, self.fs_Hz / 2, N_FFT)

        # set the tuned frequency list to be a multiple of the FFT bin size
        f_diff = np.diff(freqs)[1] # FFT bin size
        tune_inc = int(10_000_000 // f_diff)
        f_step = float(tune_inc) * f_diff  # make f_step a multiple of the fft steps

        # window is 20MHz wide
        w0_freq = freqs[freqs >= -f_step][0]
        w1_freq = freqs[freqs >= f_step][0]
        window_inc = np.argmin(abs(freqs - w0_freq))
        window_length = int((w1_freq - w0_freq) // f_diff) + 1
        half_window_length = window_length // 2

        # print(f"{f_diff/1e3:,.3f}kHz, {f_step/1e6:,.3f}MHz, {tune_inc}, {window_inc}, {window_length}")
        tune_freq_range = np.arange(f_start, f_stop, f_step)  # 10MHz steps
        # tune_freq_range = np.arange(f_start, f_stop, 2 * f_step)  # 20MHz steps

        # add the sampling window to either side of the tuneable range as f_start and f_stop define the centre frequency
        full_freq_range = np.arange(
            tune_freq_range[0] + w0_freq, tune_freq_range[-1] + w1_freq, f_diff
        )
        full_amp_range = np.array([-np.Inf] * len(full_freq_range))

        idx = 0
        for freq in tune_freq_range:  # range(int(f_start), int(f_stop), int(f_step)):
            #print(f"frequency: {freq/1e6:,.3f}MHz")
            #t0 = time.perf_counter_ns()
            self.set_frequency_Hz(freq) # 1,505 us
            #delta_us = (time.perf_counter_ns() - t0) / 1e3
            #print(f'[INFO] set freq {delta_us}us')
            
            #t0 = time.perf_counter_ns()
            data = self.SDR_1.read(1024)  # 1,493 us
            #delta_us = (time.perf_counter_ns() - t0) / 1e3
            #print(f'[INFO] read {delta_us}us')
            
            #t0 = time.perf_counter_ns()
            data_sum = data[0] + data[1]
            #    max0 = np.max(abs(data[0]))
            #    max1 = np.max(abs(data[1]))
            #    print("max signals: ", max0, max1)
            ampl, _ = spectrum_estimate( data_sum, self.fs_Hz, ref=2 ^ 12 )  # 4,289 us
            #delta_us = (time.perf_counter_ns() - t0) / 1e3
            #print(f'[INFO] spectrum estimate {delta_us}us')
            
            #t0 = time.perf_counter_ns()
            # only look at the inner 20MHz
            window = ampl[window_inc : window_inc + window_length]
            swath = full_amp_range[idx : idx + window_length]

            dwell = np.maximum(window, swath)

            full_amp_range[idx : idx + window_length] = dwell
            #delta_us = (time.perf_counter_ns() - t0) / 1e3
            #print(f'[INFO] allocate data {delta_us}us') # 200us
                        
            idx += tune_inc

            #print(f"{freq/1e9:,.3f}GHz:[{full_freq_range[idx]/1e9:,.3f}GHz|{full_freq_range[idx+half_window_length]/1e9:,.3f}GHz|{full_freq_range[idx+window_length]/1e9:,.3f}GHz]")

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
        
        return peak_freq

    def on_mpl_exit(self, event):
        if event.key == "q":
            self.stop_plotting = True

    def freq_tracker(
        self, fc_MHz: int = 10_000, BW_MHz: int = 300, update_time_s: float = 1.0, samples_per_bin: int = 1
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
        window_length = int((w1_freq - w0_freq) // f_diff) + 1
        half_window_length = window_length // 2

        ## setup plot
        plt.ion()
        fig = plt.figure(1)
        fig.canvas.mpl_connect("key_press_event", self.on_mpl_exit)

        freq_centre_MHz = fc_MHz
        
        sdr = self.SDR_1

        freqs = []
        amps = []
        snrs = []
        
        data_sum = np.zeros(sdr.rx_buffer_size, dtype=complex)

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
            
            # print('length(full_freq_range_MHz): %d, length(full_amp_range): %d'%(len(full_freq_range_MHz), len(full_amp_range)))

            idx = 0
            for (
                freq_MHz
            ) in tune_freq_range_MHz:  # range(int(f_start), int(f_stop), int(f_step)):
                fc_MHz = full_freq_range_MHz[idx + 1 + window_length // 2]
                # print(f"frequency: {freq_MHz:,.0f}MHz")
                
                # self.cn0566_1.set_frequency_Hz(freq)  # 1.743 ms
                self.set_frequency_Hz(freq_MHz * 1e6)
                time.sleep(
                    0.08
                )  # 0.06s is the threshold for accurately measuring CW signals

                data = self.SDR_1.read()  # 3.982 ms

                for _ in range(samples_per_bin):
                    data = self.SDR_1.read()  # 3.982 ms
                    data_sum += data[0] + data[1]
                
                data_sum /= samples_per_bin # average the data over 'samples_per_bin'
                
                #    max0 = np.max(abs(data[0]))
                #    max1 = np.max(abs(data[1]))
                #    print("max signals: ", max0, max1)
                ampl, _ = spectrum_estimate(
                    data_sum, self.fs_Hz, ref=2 ^ 12
                )  # 2.375 ms

                ampl = np.fft.fftshift(ampl)
                ampl = np.flip(ampl)  # Just an experiment...
                #freqs = np.fft.fftshift(freqs)
                
                window = ampl[window_inc : window_inc + window_length]
                swath = full_amp_range[idx : idx + window_length]

                # print('[%d : %d]'%(idx, idx + window_length))
                # print('pre: %d, length: %d, size(window): %d, size(swath): %d, post: %d'%(window_inc,window_length,len(window), len(swath), len(ampl) - (window_inc+window_length)))
                # print('freq_range: %f to %f MHz>>%f MHz'%(full_freq_range_MHz[idx],full_freq_range_MHz[min(len(full_freq_range_MHz)-1, idx+window_length)],full_freq_range_MHz[min(len(full_freq_range_MHz)-1, idx+window_length+1)]))

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
                            f"[{len(freqs)}] collects average: Fc {np.average(freqs):,.3f}MHz, Amp {np.average(amps):,.3f}dBFS, SNR {np.average(SNR_dBFS):,.3f}dB"
                        )
                    # print(f"new centre frequency: {freq_centre_MHz:,.0f}MHz")
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
            sdr = self.SDR_1
        else:
            cn0566 = self.cn0566_2
            sdr = self.SDR_2

        win = np.blackman(sdr.rx_buffer_size)
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
            print("  channel_levels: [%f,%f] dB"%(channel_levels[0], channel_levels[1]))
            print("channel mismatch: ", ch_mismatch, " dB")
        if ch_mismatch > 0:  # Channel 0 higher, boost ch1:
            cn0566.cn0566.ccal = [0.0, ch_mismatch]
        else:  # Channel 1 higher, boost ch0:
            cn0566.cn0566.ccal = [-ch_mismatch, 0.0]
        pass

    def measure_element_gain(self, cn0566=None, cal=0, peak_bin=0, verbose=False):  # Default to central element
        """ Calculate all the values required to do different plots. It method calls set_beam_phase_diff and
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
        width = 10  # Bins around fundamental to sum
        cn0566.set_rx_hardwaregain(6)  # Channel calibration defaults to True
        cn0566.set_all_gain(0, apply_cal=False)  # Start with all gains set to zero
        cn0566.set_chan_gain(cal, 127, apply_cal=False)  # Set element to max
        time.sleep(1.0)  # todo - remove when driver fixed to compensate for ADAR1000 quirk
        if verbose:
            print("measuring element: ", cal)
        total_sum = 0
        # win = np.blackman(cn0566.sdr.rx_buffer_size)
        win = signal.windows.flattop(cn0566.sdr.rx_buffer_size)
        win /= np.average(np.abs(win))  # Normalize to unity gain
        spectrum = np.zeros(cn0566.sdr.rx_buffer_size)

        for count in range(0, cn0566.Averages):  # repeatsnip loop and average the results
            data = cn0566.sdr.rx()  # todo - remove once confirmed no flushing necessary
            data = cn0566.sdr.rx()  # read a buffer of data
            y_sum = (data[0] + data[1]) * win

            s_sum = np.fft.fftshift(np.absolute(np.fft.fft(y_sum)))
            spectrum += s_sum

            # Look for peak value within window around fundamental (reject interferers)
            s_mag_sum = np.max(s_sum[peak_bin - width : peak_bin + width])
            total_sum += s_mag_sum

        spectrum /= cn0566.Averages * cn0566.sdr.rx_buffer_size
        PeakValue_sum = total_sum / (cn0566.Averages * cn0566.sdr.rx_buffer_size)

        return PeakValue_sum, spectrum

    def gain_calibration(self, cn0566=None, verbose=False):
        """ Perform the Gain Calibration routine."""

        """Set the gain calibration flag and create an empty gcal list. Looping through all the possibility i.e. setting
            gain of one of the channel to max and all other to 0 create a zero-list where number of 0's depend on total
            channels. Replace only 1 element with max gain at a time. Now set gain values according to above Note."""

        if (cn0566 is None) or (cn0566 == self.cn0566_1):
            cn0566 = self.cn0566_1
        else:
            cn0566 = self.cn0566_2

        cn0566.cn0566.gain_cal = True  # Gain Calibration Flag
        gcalibrated_values = []  # Intermediate cal values list
        plot_data = []
        peak_bin = self.find_peak_bin()
        if verbose is True:
            print("Peak bin at ", peak_bin, " out of ", cn0566.sdr.rx_buffer_size)
        # gcal_element indicates current element/channel which is being calibrated
        for gcal_element in range(0, (cn0566.cn0566.num_elements)):
            if verbose is True:
                print("Calibrating Element " + str(gcal_element))

            gcal_val, spectrum = self.measure_element_gain(
                cn0566, gcal_element, peak_bin, verbose=True
            )
            if verbose is True:
                print("Measured signal level (ADC counts): " + str(gcal_val))
            gcalibrated_values.append(gcal_val)  # make a list of intermediate cal values
            plot_data.append(spectrum)

        """ Minimum gain of intermediated cal val is set to Maximum value as we cannot go beyond max value and gain
            of all other channels are set accordingly"""
        print("gcalibrated values: ", gcalibrated_values)
        for k in range(0, 8):
            #            x = ((gcalibrated_values[k] * 127) / (min(gcalibrated_values)))
            cn0566.cn0566.gcal[k] = min(gcalibrated_values) / (gcalibrated_values[k])

        cn0566.cn0566.gain_cal = (
            False  # Reset the Gain calibration Flag once system gain is calibrated
        )

        return plot_data
        # print(cn0566.gcal)

    def phase_calibration(self, cn0566=None, verbose=False):
        """ Perform the Phase Calibration routine."""

        """ Set the phase calibration flag and create an empty pcal list. Looping through all the possibility
            i.e. setting gain of two adjacent channels to gain calibrated values and all other to 0 create a zero-list
            where number of 0's depend on total channels. Replace gain value of 2 adjacent channel.
            Now set gain values according to above Note."""
        peak_bin = self.find_peak_bin()
        if verbose is True:
            print("Peak bin at ", peak_bin, " out of ", cn0566.sdr.rx_buffer_size)

        #        cn0566.phase_cal = True  # Gain Calibration Flag
        #        cn0566.load_gain_cal('gain_cal_val.pkl')  # Load gain cal val as phase cal is dependent on gain cal
        cn0566.pcal = [0, 0, 0, 0, 0, 0, 0, 0]
        cn0566.ph_deltas = [0, 0, 0, 0, 0, 0, 0]
        plot_data = []
        # cal_element indicates current element/channel which is being calibrated
        # As there are 8 channels and we take two adjacent chans for calibration we have 7 cal_elements
        for cal_element in range(0, 7):
            if verbose is True:
                print("Calibrating Element " + str(cal_element))

            PhaseValues, gain, = self.phase_cal_sweep(
                cn0566, peak_bin, cal_element, cal_element + 1
            )

            ph_delta = to_sup((180 - PhaseValues[gain.index(min(gain))]) % 360.0)
            if verbose is True:
                print("Null found at ", PhaseValues[gain.index(min(gain))])
                print("Phase Delta to correct: ", ph_delta)
            cn0566.ph_deltas[cal_element] = ph_delta

            cn0566.pcal[cal_element + 1] = to_sup(
                (cn0566.pcal[cal_element] - ph_delta) % 360.0
            )
            plot_data.append(gain)
        return PhaseValues, plot_data

    def phase_cal_sweep(self, cn0566 = None, peak_bin=0, ref=0, cal=1):
        """ Calculate all the values required to do different plots. It method
            calls set_beam_phase_diff and sets the Phases of all channel.
            parameters:
                gcal_element: type=int
                            If gain calibration is taking place, it indicates element number whose gain calibration is
                            is currently taking place
                cal_element: type=int
                            If Phase calibration is taking place, it indicates element number whose phase calibration is
                            is currently taking place
                peak_bin: type=int
                            Which bin the fundamental is in.
                            This prevents detecting other spurs when deep in a null.
        """

        cn0566.set_all_gain(0)  # Reset all elements to zero
        cn0566.set_chan_gain(ref, 127, apply_cal=True)  # Set two adjacent elements to zero
        cn0566.set_chan_gain(cal, 127, apply_cal=True)
        sleep(1.0)

        cn0566.set_chan_phase(ref, 0.0, apply_cal=False)  # Reference element
        # win = np.blackman(cn0566.sdr.rx_buffer_size)
        win = signal.windows.flattop(cn0566.sdr.rx_buffer_size)  # Super important!
        win /= np.average(np.abs(win))  # Normalize to unity gain
        width = 10  # Bins around fundamental to sum
        sweep_angle = 180
        # These are all the phase deltas (i.e. phase difference between Rx1 and Rx2, then Rx2 and Rx3, etc.) we'll sweep
        PhaseValues = np.arange(-(sweep_angle), (sweep_angle), cn0566.phase_step_size)

        gain = []  # Create empty lists
        for phase in PhaseValues:  # These sweeps phase value from -180 to 180
            # set Phase of channels based on Calibration Flag status and calibration element
            cn0566.set_chan_phase(cal, phase, apply_cal=False)
            total_sum = 0
            for count in range(0, cn0566.Averages):  # repeat loop and average the results
                data = cn0566.sdr.rx()  # read a buffer of data
                data = cn0566.sdr.rx()
                y_sum = (data[0] + data[1]) * win
                s_sum = np.fft.fftshift(np.absolute(np.fft.fft(y_sum)))

                # Pick (uncomment) one:
                # 1) RSS sum a few bins around max
                # s_mag_sum = np.sqrt(
                #     np.sum(np.square(s_sum[peak_bin - width : peak_bin + width]))
                # )

                # 2) Take maximum value
                # s_mag_sum = np.maximum(s_mag_sum, 10 ** (-15))

                # 3) Apparently the correct way, use flat-top window, look for peak
                s_mag_sum = np.max(s_sum[peak_bin - width : peak_bin + width])
                s_mag_sum = np.max(s_sum)
                total_sum += s_mag_sum
            PeakValue_sum = total_sum / (cn0566.Averages * cn0566.sdr.rx_buffer_size)
            gain.append(PeakValue_sum)

        return (
            PhaseValues,
            gain,
        )  # beam_phase, max_gain

    # plutoSDR range: 0 to 74.5 dB
    # bladeRF range: -15 to 60 dB
    def set_sdr_gain(self, cn0566=None, gain:float = 6.0, apply_calibration:bool = False):
        if (cn0566 is None) or (cn0566 == self.cn0566_1):
            cn0566 = self.cn0566_1
            sdr = self.SDR_1
        else:
            cn0566 = self.cn0566_2
            sdr = self.SDR_2

        sdr.set_rx_hardwaregain(gain, apply_calibration)

    def antenna_element_test(self, cn0566=None, fc_Hz=None, verbose: bool = False):
        if (cn0566 is None) or (cn0566 == self.cn0566_1):
            cn0566 = self.cn0566_1
            sdr = self.SDR_1
        else:
            cn0566 = self.cn0566_2
            sdr = self.SDR_2

        if (fc_Hz is not None) and (fc_Hz != self.fc_Hz):
            self.set_frequency_Hz(fc_Hz)

        peak_bin = self.find_peak_bin()
        
        # initial channel calibration
        channel_levels, _ = self.measure_channel_gains(peak_bin, cn0566, verbose=False)
        channel_mismatch = 20.0 * np.log10(channel_levels[0] / channel_levels[1])
        print("  channel_levels: [%f,%f] dB"%(channel_levels[0], channel_levels[1]))
        print("channel mismatch: ", channel_mismatch, " dB")
        if channel_mismatch > 0:  # Channel 0 higher, boost ch1:
            cn0566.cn0566.ccal = [0.0, channel_mismatch]
        else:  # Channel 1 higher, boost ch0:
            cn0566.cn0566.ccal = [-channel_mismatch, 0.0]
        
        if abs(channel_mismatch) > 3.0:
            print("[ERROR] channel mismatch is greater than 3dB, please check your connections")
        

        # check each element
        sdr.rx_gain = 63
        cn0566.cn0566.set_all_gain(0, apply_cal=False)  # Start with all gains set to zero
        element_levels = np.zeros(8)
        # enable each element 1 by 1
        for element in range(4):
            # Note: channel 1 is at 0->3, channel 2 is 4->7
            # Channel 1 - set element to max
            cn0566.cn0566.set_chan_gain(
                element,
                127,
                apply_cal=False,
            )  
            # Channel 2 - set element to max
            cn0566.cn0566.set_chan_gain(
                4 + element,
                127,
                apply_cal=False,
            )
            time.sleep(0.001)
            ch_lvl, _ = self.measure_channel_gains(peak_bin, cn0566, set_rx_gains=False, verbose=False)
            element_levels[element] = 20.0 * np.log10(ch_lvl[0])
            element_levels[4+element] = 20.0 * np.log10(ch_lvl[1])
            # Channel 1 - set element back to zero
            cn0566.cn0566.set_chan_gain(
                element,
                0,
                apply_cal=False,
            )  
            # Channel 2 - set element back to zero
            cn0566.cn0566.set_chan_gain(
                4 + element,
                0,
                apply_cal=False,
            )
            time.sleep(0.001)
        print("[INFO] Per element signal level (dBFS):\n%s"%(format_vector(element_levels)))


    def measure_channel_gains(
        self, peak_bin, cn0566=None, set_rx_gains=True, verbose=False
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
            sdr = self.SDR_1
        else:
            cn0566 = self.cn0566_2
            sdr = self.SDR_2

        width = 10  # Bins around fundamental to sum
        win = signal.windows.flattop(sdr.rx_buffer_size)
        win /= np.average(np.abs(win))  # Normalize to unity gain
        plot_data = []
        channel_level = []
        # cn0566.cn0566.set_rx_hardwaregain(6, False)
        
        if set_rx_gains:
            sdr.rx_gain = 63
            cn0566.cn0566.set_all_gain(
                127, apply_cal=False
            )
        
        spectrum_averaged_perchannel = [np.zeros(sdr.rx_buffer_size), np.zeros(sdr.rx_buffer_size)]
        peakvalue_averaged_perchannel = [0, 0]
        
        # data = sdr.read()  # todo - remove once confirmed no flushing necessary
        data = sdr.read()  # read a buffer of data

        spectrum_averaged_perchannel[0] = np.fft.fftshift(np.absolute(np.fft.fft(data[0] * win)))
        spectrum_averaged_perchannel[1] = np.fft.fftshift(np.absolute(np.fft.fft(data[1] * win)))

        # Look for peak value within window around fundamental (reject interferers)
        peakvalue_averaged_perchannel[0] = np.max(spectrum_averaged_perchannel[0][peak_bin - width : peak_bin + width])
        peakvalue_averaged_perchannel[1] = np.max(spectrum_averaged_perchannel[1][peak_bin - width : peak_bin + width])
        time.sleep(0.001)
            
        for count in range(
            0, cn0566.cn0566.Averages -1
        ):  # repeats loop and average the results
            
            data = sdr.read()  # read a buffer of data

            spectrum_averaged_perchannel[0] += np.fft.fftshift(np.absolute(np.fft.fft(data[0] * win)))
            spectrum_averaged_perchannel[1] += np.fft.fftshift(np.absolute(np.fft.fft(data[1] * win)))

            # Look for peak value within window around fundamental (reject interferers)
            peakvalue_averaged_perchannel[0] += np.max(spectrum_averaged_perchannel[0][peak_bin - width : peak_bin + width])
            peakvalue_averaged_perchannel[1] += np.max(spectrum_averaged_perchannel[1][peak_bin - width : peak_bin + width])
            time.sleep(0.001)

        spectrum_averaged_perchannel[0] /= (cn0566.cn0566.Averages * sdr.rx_buffer_size)
        spectrum_averaged_perchannel[1] /= (cn0566.cn0566.Averages * sdr.rx_buffer_size)
        peakvalue_averaged_perchannel[0] /= (cn0566.cn0566.Averages * sdr.rx_buffer_size)
        peakvalue_averaged_perchannel[1] /= (cn0566.cn0566.Averages * sdr.rx_buffer_size)
 
        return peakvalue_averaged_perchannel, spectrum_averaged_perchannel


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
    
    # 2,097,152:   824,189.696us =  0.393us/sample, max sample rate 2.545Msps
    # 2,097,152:   693,572.434us =  0.331us/sample, max sample rate 3.024Msps
    # 2,097,152:   643,028.126us =  0.307us/sample, max sample rate 3.261Msps
    # 2,097,152:   690,810.140us =  0.329us/sample, max sample rate 3.036Msps
    # 2,097,152:   565,238.214us =  0.270us/sample, max sample rate 3.710Msps
    # 2,097,152:   563,814.052us =  0.269us/sample, max sample rate 3.720Msps
    # 2,097,152:   573,696.296us =  0.274us/sample, max sample rate 3.656Msps
    
    rx_buffer_range = rx_buffer_range[1:] # remove 1024 as the BladeRF cannot support this

    for rx_buff_size in rx_buffer_range:
        # set the buffer size
        my_phaser.SDR_1.rx_buffer_size=rx_buff_size
        # my_phaser.SDR_1.plutoSDR.rx_buffer_size = int(rx_buff_size)

        # time the receive
        t0 = time.perf_counter_ns()
        #data = my_phaser.SDR_1.read()
        data = my_phaser.SDR_1.read_buffer()
        delta_us = (time.perf_counter_ns() - t0) / 1e3
        #num_samples = np.shape(data)[1]
        num_samples = len(data)//my_phaser.SDR_1.total_bytes_per_sample
        
        max_sample_rate_Msps = (
            num_samples / delta_us
        )  # in mega samples per second
        print(
            f"{rx_buff_size:9,d}: {num_samples:9,d}: {delta_us:13,.3f}us = {1/max_sample_rate_Msps:6,.3f}us/sample, max sample rate {max_sample_rate_Msps:2.3f}Msps"
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
    my_phaser.set_rx_buffer_size(2048)
    
    peak_freq = my_phaser.find_peak(10.4e9, 10.6e9)
    my_phaser.set_frequency_Hz(peak_freq)
    my_phaser.antenna_element_test()
    

#    # channel calibration
#    if True:
#        while True:
#            peak_freq = my_phaser.find_peak(9.3e9, 10.6e9)
#            if peak_freq is not None:
#                my_phaser.set_frequency_Hz(peak_freq)
#                my_phaser.channel_calibration(verbose=True)
#                my_phaser.set_sdr_gain() # set to 6dB
#                my_phaser.antenna_element_test()
#                break

#            time.sleep(5)
#        
#        peak_freq_MHz = int(peak_freq/1e6)
#    else:
#        peak_freq_MHz = 10_430

    # my_phaser.plot(peak_freq_MHz*1e6)

    # 0.5192586079313221 dB @ 9.4GHz, rx_gain=6dB
    # my_phaser.cn0566_1.cn0566.ccal = [0.0, 0.5192586079313221]
    # my_phaser.SDR_1.channel_cal = [1.2807293215820739, 0.0]
    #my_phaser.SDR_1.channel_cal = [2.0, 0.0]

    # my_phaser.SDR_1.rx_gain=63  # 60 = 1500/2000

    # my_phaser.set_frequency_Hz(peak_freq_MHz*1e6)
    # my_phaser.channel_calibration(verbose=True)

    # test_buffer_size(my_phaser)

    # my_phaser.plot(peak_freq_MHz*1e6)

    # my_phaser.freq_tracker(peak_freq_MHz)
    #my_phaser.freq_tracker(10_400, BW_MHz=100)

    # my_phaser.sweep(9.7e9, 10.6e9, interactive=True)
    # my_phaser.find_peak(9.3e9, 10.6e9)

    # from line_profiler import LineProfiler

    # lp = LineProfiler()
    # lp_wrapper = lp(my_phaser.sweep)
    # lp_wrapper(10e9, 10.6e9)
    # lp.print_stats()

    print("[INFO] end")
