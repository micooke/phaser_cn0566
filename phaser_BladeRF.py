#!/usr/bin/env python3

# USB 3.0 Control of the phaser BladeRF receiver
 
import os, sys
import time
import atexit

import numpy as np
from bladerf import _bladerf

from phaser_utils import Phaser_LO_HIGH, print_vector, plot_2channels
from phaser_IO import phaser_SigMF, phaser_IO


# add additional functions to the standard BladeRF class to enable or disable oversampling
class BladeRF(_bladerf.BladeRF):
    def enable_oversample(self):
        ret = _bladerf.libbladeRF.bladerf_enable_feature(self.dev[0], 1, True)
        _bladerf._check_error(ret)

    def disable_oversample(self):
        ret = _bladerf.libbladeRF.bladerf_enable_feature(self.dev[0], 0, True)
        _bladerf._check_error(ret)
        
    #def calibrate_dc(self): # bladeRF 1.0
    #    # BladeRF_Cal_Module = {"BLADERF_DC_CAL_INVALID": -1,
    #    #                       "BLADERF_DC_CAL_LPF_TUNING": 0,
    #    #                       "BLADERF_DC_CAL_TX_LPF": 1,
    #    #                       "BLADERF_DC_CAL_RX_LPF": 2,
    #    #                       "BLADERF_DC_CAL_RXVGA2": 3}
    #    #
    #    # Note: lms cal runs through 0,1,2,3 in order for calibration
    #    ret = _bladerf.libbladeRF.bladerf_calibrate_dc(self.dev[0], 2)
    #    _bladerf._check_error(ret)
 
# phaser class
# @author: Mark Cooke
class phaser_BladeRF:
    def __init__(
        self,
        fs_Hz: int = 30_000_000,
        fc_Hz: float = Phaser_LO_HIGH,
        # channel_number: int = [0,1],
        DeviceString: str = "6b5d",
        # serial: bdb3a74b4ba24a8e9b10cc14fd7f6b5d
    ):
        self.__name__ = "BladeRF"
        self.ccal = [0.0, 0.0]
        self.default_gain_dB = 0
        self.num_channels = 2
        self._bytes_per_sample = 2
        self.total_bytes_per_sample = 2 * self._bytes_per_sample * self.num_channels  # 2* for I and Q
        self.rx_dtype = f"<i{self._bytes_per_sample}"

        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        self._sample_format = _bladerf.Format.SC16_Q11
        self._channel = None
        # self.is_streaming: bool = False

        self.sdr = None
        self.sigmf = None

        self.buffer = bytearray(0)
        self._rx_buffer_size = 32_768
        
        self._dc_offset = np.complex64([0.0+1j*0.0, 0.0+1j*0.0])

        self.setup(
            fs_Hz, fc_Hz, self.default_gain_dB, self._rx_buffer_size, DeviceString
        )

        atexit.register(self.close)

    def close(self):
        for ch in self._channel:
            ch.enable = False
        self.sdr.close()    

    # @author: Mark Cooke
    def setup(
        self,
        fs_Hz: int = 30_000_000,
        fc_Hz: float = Phaser_LO_HIGH,
        rx_gain_dB: int = 10,
        rx_buffer_size: int = 32_768,
        DeviceString: str = "6b5d",
        sample_format=_bladerf.Format.SC16_Q11,
    ):
        self.sdr = None
        self.DeviceBytes = DeviceString.encode("utf-8")

        try:
            devices = _bladerf.get_device_list()

            for d in devices:
                if d.serial[-len(self.DeviceBytes) :] == self.DeviceBytes:
                    self.sdr = BladeRF(devinfo=d)

            if self.sdr is None:
                if len(devices) > 0:
                    self.sdr = BladeRF(devinfo=devices[0])
                else:
                    return None

        except _bladerf.BladeRFError:
            print("No bladeRF devices found.")
            self.close()

        time.sleep(0.5)

        # ['RX_X1', 'RX_X2', 'TX_X1', 'TX_X2']
        self._layout = _bladerf.ChannelLayout.RX_X2

        self._channel = [
            self.sdr.Channel(_bladerf.CHANNEL_RX(0)),
            self.sdr.Channel(_bladerf.CHANNEL_RX(1)),
        ]

        self.rx_buffer_size = int(rx_buffer_size)
        
        self.sample_format = sample_format

        #  Configure SDR parameters.
        self.fc_Hz = int(fc_Hz)  # Downconvert to ~2GHz | Receive Freq
        self.fs_Hz = int(fs_Hz)  # Sampling rate

        self.rx_bandwidth_Hz = int(10e6)  # int(fs_Hz // 2)# int(10e6)

        self.rx_gain = rx_gain_dB

        # enable the channels
        self._channel[0].enable = True
        self._channel[1].enable = True

        # Handy filter for fairly wideband measurements
        # self.sdr.filter = os.path.join(self.base_dir, "LTE20_MHz.ftr")

        # # Make sure the Tx channels are attenuated (or off) and their freq is far away from Rx
        # # this is a negative number between 0 and -88 (-90 according to another source)
        # tx_channels = [self.sdr.Channel(_bladerf.CHANNEL_TX(0)), self.sdr.Channel(_bladerf.CHANNEL_TX(1))]
        # for ch in tx_channels:
        #     ch.gain = -24 # [-24 to 66dB]
        #     ch.enable = False
        
        return True

    def calibrate_dc(self, fc_Hz = None, rx_gain_dB = None):
        if fc_Hz is not None:
            self.fc_Hz = fc_Hz
        if rx_gain_dB is not None:
            self.rx_gain = rx_gain_dB
        
        data = self.read()
        
        self._dc_offset = np.complex64([np.mean(data[0]), np.mean(data[1])])
        print(f"[INFO] DC offset = {self._dc_offset}")        
        

    # ['SC16_Q11', 'SC16_Q11_META', 'SC8_Q7', 'SC8_Q7_META']
    @property
    def sample_format(self):
        return self._sample_format

    @sample_format.setter
    def sample_format(self, value=_bladerf.Format.SC16_Q11):
        self._sample_format = value
        
        if self._sample_format == _bladerf.Format.SC8_Q7:
            self._bytes_per_sample = 1
        else:  # _bladerf.Format.SC16_Q11
            self._bytes_per_sample = 2
        
        self.total_bytes_per_sample = 2 * self._bytes_per_sample * self.num_channels  # 2* for I and Q
        self.rx_dtype = f"<i{self._bytes_per_sample}"

    @property
    def channel_cal(self):
        return self.ccal

    @channel_cal.setter
    def channel_cal(self, ccal):
        self.ccal = ccal

    @property
    def rx_gain_mode(self):
        return self._channel[0].gain_mode

    # _bladerf.GainMode.[Default, Manual, FastAttack_AGC, SlowAttack_AGC, Hybrid_AGC]
    # Default: This is BLADERF_GAIN_SLOWATTACK_AGC with reasonable default settings
    @rx_gain_mode.setter
    def rx_gain_mode(self, gain_mode: _bladerf.GainMode = _bladerf.GainMode.Manual):
        self._channel[0].gain_mode = gain_mode
        self._channel[1].gain_mode = gain_mode

    @property
    def rx_gain(self):
        return [self._channel[0].gain, self._channel[1].gain]

    # BladeRF range: -15 to 60 dB
    @rx_gain.setter
    def rx_gain(self, gain_dB: int = 0):
        if self._channel[0].gain_mode != _bladerf.GainMode.Manual:
            self._channel[0].gain_mode = _bladerf.GainMode.Manual
            self._channel[1].gain_mode = _bladerf.GainMode.Manual

        if self._channel[0].gain == int(gain_dB + self.ccal[0]):
            self._channel[0].gain = int(gain_dB + self.ccal[0])
            self._channel[1].gain = int(gain_dB + self.ccal[1])

    @property
    def rx_bandwidth_Hz(self):
        return self._channel[0].bandwidth

    @rx_bandwidth_Hz.setter
    def rx_bandwidth_Hz(self, rx_bandwidth_Hz: int = 10_000_000):
        if rx_bandwidth_Hz != self._channel[0].bandwidth:
            self._channel[0].bandwidth = rx_bandwidth_Hz
            self._channel[1].bandwidth = rx_bandwidth_Hz

    @property
    def fs_Hz(self):
        return self._channel[0].sample_rate

    @fs_Hz.setter
    def fs_Hz(self, fs_Hz: float = 30_000_000):
        if self._channel[0].sample_rate != fs_Hz:
            self._channel[0].sample_rate = fs_Hz
            self._channel[1].sample_rate = fs_Hz

        if self.rx_bandwidth_Hz > fs_Hz:
            self.rx_bandwidth_Hz = fs_Hz

    @property
    def rx_buffer_size(self):
        return self._rx_buffer_size

    @rx_buffer_size.setter
    def rx_buffer_size(self, rx_buffer_size: int = 32_768):
        print("[INFO] rx_buffer_size")
        if (rx_buffer_size % 1_024) != 0:
            _rx_buffer_size = ((rx_buffer_size // 1_024) + 1) * 1_024
            print(
                f"[WARNING] requested rx_buffer_size is not a multiple of 1_024. It will be set to {_rx_buffer_size:,}"
            )
        else:
            _rx_buffer_size = rx_buffer_size

        if _rx_buffer_size != self._rx_buffer_size:
            self._rx_buffer_size = _rx_buffer_size

        # print(f"[INFO] rx_buffer_size: {self._layout}, {self._sample_format} {self._rx_buffer_size:,}")

        # Setup synchronous stream MIMO 2x2
        self.sdr.sync_config(
            layout=self._layout,
            fmt=self._sample_format,
            num_buffers=32,
            buffer_size=self._rx_buffer_size,  # must be a multiple of 1_024
            num_transfers=16,  # 4, 8, 16
            stream_timeout=1_000,
        )

    @property
    def fc_Hz(self):
        if self._channel is not None:
            return self._channel[0].frequency
        return None

    @fc_Hz.setter
    def fc_Hz(self, fc_Hz: float = Phaser_LO_HIGH):
        self._channel[0].frequency = fc_Hz
        self._channel[1].frequency = fc_Hz

    def read_buffer(self, num_samples:int = int(2**21)):
        # print("[INFO] read_buffer")
        total_samples_read = self.num_channels * num_samples
        total_buffer_size = num_samples * self.total_bytes_per_sample

        if len(self.buffer) != total_buffer_size:
            print("[INFO] resize byte array")
            self.buffer = bytearray(total_buffer_size)

        # [INFO] buf length 16,777,216, read count 4,194,304
        # print(f"[INFO] buf length {num_samples * self.total_bytes_per_sample:,}, read count {total_samples_read:,}")

        # Read into buffer
        # t0 = time.perf_counter_ns()
        # self.sdr.sync_config(
        #     layout=self._layout,
        #     fmt=self._sample_format,
        #     num_buffers=32,
        #     buffer_size=self._rx_buffer_size,  # must be a multiple of 1_024
        #     num_transfers=16,  # 4, 8, 16
        #     stream_timeout=1_000,
        # )
        self.sdr.sync_rx(self.buffer, total_samples_read)
        # t1 = time.perf_counter_ns()
        # time_delta_us = (t1 - t0) / 1_000
        # print("[INFO] read {total_samples_read} into buffer in {time_delta_us:,.3f}us}")

        return self.buffer
    
    def read(self, num_samples:int = int(2048)):#int(2**21)):
        self.read_buffer(num_samples)
        data = np.frombuffer(self.buffer, dtype=self.rx_dtype)

        # 2048 = 2^(12-1) bits; where ADC bit number is 12b
        # int range = [-2048:2047]
        # convert int to complex64
        complex_data=(data/2047).astype(np.float32).view(np.complex64)
        # deinterleave
        signals = [complex_data[0::2] + self._dc_offset[0], complex_data[1::2] + self._dc_offset[1]]

        return signals
    
    def timed_read(self, num_samples:int = int(2**21)):
        print("[INFO] read")
        total_samples_read = self.num_channels * num_samples
        
        t0 = time.perf_counter_ns()
        self.read_buffer(num_samples)
        t1 = time.perf_counter_ns()
        
        time_readbytes_us = (t1 - t0) / 1_000
        
        # Read into buffer
        t1 = time.perf_counter_ns()
        data = np.frombuffer(self.buffer, dtype=self.rx_dtype)
        t2 = time.perf_counter_ns()

        time_byte2int_us = (t2 - t1) / 1_000
        
        t2 = time.perf_counter_ns()
        # 2048 = 2^(12-1) bits; where ADC bit number is 12b
        # int range = [-2048:2047]
        # convert int to complex64: 532.02us for 1024 samples
        complex_data=(data/2047).astype(np.float32).view(np.complex64)
        # deinterleave
        signals = [complex_data[0::2] + self._dc_offset[0], complex_data[1::2] + self._dc_offset[1]]
        ## convert int to complex128: 533.98us for 1024 samples
        #signals = [(data[0:-3:4] / 2047) + 1j*(data[1:-2:4] / 2047), 
        #           (data[2:-1:4] / 2047) + 1j*(data[3::4] / 2047)]
        t3 = time.perf_counter_ns()

        time_int2complex_us = (t3 - t2) / 1_000
        
        conversion_time_us = (t3 - t1) / 1_000
        total_time_us = (t3 - t0) / 1_000
        
        num_samples_read = len(signals[0])

        read_buffer_rate_MBps = len(self.buffer) / conversion_time_us
        convert_rate_Msps = num_samples_read / conversion_time_us
        total_read_rate_Msps = num_samples_read / total_time_us

        num_bytes = (
            sys.getsizeof(self.buffer) / 2
        )  # Not sure why /2, but when i write the data out, thats what it is

        print(f"[INFO] buffer_len: {len(self.buffer):,}, num_bytes: {num_bytes:,}, num_samples (per channel): {num_samples_read:,}")

        read_rate_MBph = (num_bytes / time_readbytes_us) * 60 * 60
        mins_to_record_1TB_hr = 60*(1_000_000)/read_rate_MBph

        print("[INFO]")
        print(f"* read bytes (us): {time_readbytes_us:,.2f}")
        print(f"* convert:bytes2int (us): {time_byte2int_us:,.2f}")
        print(f"* convert:int2complex (us): {time_int2complex_us:,.2f}")
        print(f"* read bytes (MB/s): {read_buffer_rate_MBps:,.2f}")
        print(f"* convert rate (MS/s): {convert_rate_Msps:,.2f}")
        print(f"* total read rate (MS/s): {total_read_rate_Msps:,.2f}")
        print(f"* data rate (MB/hr): {read_rate_MBph:,.2f}")
        print(f"* time to record 1TB (min): {mins_to_record_1TB_hr:,.2f}")

        return signals

    @staticmethod
    def list():
        print("[ERROR] 'list()' function is not implemented")

    @staticmethod
    def read_test(rx_buffer_size: int = 32_768):
        DeviceID = "6b5d".encode("utf-8")
        fc_Hz: float = Phaser_LO_HIGH
        fs_Hz: int = 10_000_000
        BW_Hz: int = fs_Hz
        gain_dB: int = 0 # -15 to 60 dB

        total_buffer_size = 16_777_216
        total_samples_read = 4_194_304
        buffer = bytearray(total_buffer_size)

        # [INFO] buf length 16,777,216|16,777,216, read count 4,194,304
        # print(f"[INFO] buf length {total_buffer_size:,}, read count {total_samples_read:,}")

        devices = _bladerf.get_device_list()
        
        sdr = None
        for d in devices:
            if d.serial[-len(DeviceID) :] == DeviceID:
                sdr = BladeRF(devinfo=d)

        if sdr is None:
            sdr = BladeRF(devinfo=devices[0])
        
        # Setup sdr
        # activate channels
        channels = [sdr.Channel(_bladerf.CHANNEL_RX(0)), sdr.Channel(_bladerf.CHANNEL_RX(1))]

        for ch in channels:
            ch.sample_rate = fs_Hz
            ch.frequency = fc_Hz
            ch.bandwidth = BW_Hz
            ch.gain = gain_dB
            ch.enable = True

        # Setup synchronous stream MIMO 2x2

        # Read into buffer
        sdr.sync_config(
            layout=_bladerf.ChannelLayout.RX_X2,
            fmt=_bladerf.Format.SC16_Q11,
            num_buffers=32,
            buffer_size=rx_buffer_size,  # must be a multiple of 1_024
            num_transfers=16,  # 4, 8, 16
            stream_timeout=1_000,
        )
        
        sdr.sync_rx(buffer, total_samples_read)
        
        print_vector(buffer)

    @staticmethod
    def test(
        num_reads: int = 1,
        rx_buffer_size: int = 32_768,
        num_samples: int = 2**21,
        DeviceString: str = "6b5d",
    ):
        print("[INFO] test")
        sdr = phaser_BladeRF(DeviceString=DeviceString)
        if sdr.sdr is None:
            print(f"[ERROR] {sdr.__name__} not found")
            return

        sdr.rx_buffer_size = rx_buffer_size
        data = [0] * num_samples
        sample_count = 0
        t0 = time.perf_counter_ns()
        for n in range(num_reads):
            data = sdr.read()
            sample_count += len(data)
        t1 = time.perf_counter_ns()
        sdr.close()
        print(data)

        time_delta_us = (t1 - t0) / 1_000
        read_rate_Msps = sample_count / time_delta_us
        print(
            f"read: sample_count = {sample_count:,}, time: {time_delta_us:,.3f}us, rate: {read_rate_Msps:,.3f}Msps"
        )

    @staticmethod
    def record(
        num_reads: int = 1,
        fc_Hz: float = Phaser_LO_HIGH,
        fs_Hz: int = 10_000_000,
        rx_buffer_size: int = 32_768,
        num_samples: int = 2**21,
        DeviceString: str = "6b5d",
    ):
        print("[INFO] record")
        sdr = phaser_BladeRF(fs_Hz=fs_Hz, fc_Hz=fc_Hz, DeviceString=DeviceString)
        if sdr.sdr is None:
            print(f"[ERROR] {sdr.__name__} not found")
            return

        sdr.rx_buffer_size = rx_buffer_size

        sdr.sigmf = [0,1]
        for ch in [0, 1]:
            sdr.sigmf[ch] = phaser_SigMF()
            sdr.sigmf[ch].open(ch, fs_Hz, fc_Hz)

            t0 = time.perf_counter_ns()
            sample_count = 0
            for n in range(num_reads):
                data = sdr.read()
                sample_count += len(data)
                sdr.sigmf[ch].write(data)
            t1 = time.perf_counter_ns()
            sdr.close()

        time_delta_us = (t1 - t0) / 1_000

        for ch in [0, 1]:
            sdr.sigmf[ch].close()
            phaser_IO.delete(sdr.sigmf[ch].base_filename + ".sigmf-data")
            phaser_IO.delete(sdr.sigmf[ch].base_filename + ".sigmf-meta")

        read_and_record_rate_Msps = sample_count / time_delta_us
        print(
            f"record: sample_count = {sample_count:,}, time: {time_delta_us:,.3f}us, rate: {read_and_record_rate_Msps:,.3f}Msps"
        )

    @staticmethod
    def plot(fc_Hz: float = Phaser_LO_HIGH, DeviceString: str = "6b5d"):
        print("[INFO] plot")
        import matplotlib.pyplot as plt

        sdr = phaser_BladeRF(fc_Hz=fc_Hz, DeviceString=DeviceString)
        if sdr.sdr is None:
            print(f"[ERROR] {sdr.__name__} not found")
            return

        sdr.rx_gain = 63  # [-15 to 60dB] 60 = 1500/2000
        sdr.sdr.calibrate_rx_dc(offset=True)
        data = sdr.read(1024)
        print("[INFO] read complete")
        fs_Hz = sdr.fs_Hz
        sdr.close()
        #print(type(data[0][0]))

        # Take FFT
        print("[INFO] FFT")
        PSD0 = 10 * np.log10(np.finfo(np.complex64).eps + np.abs(np.fft.fftshift(np.fft.fft(data[0]))) ** 2)
        PSD1 = 10 * np.log10(np.finfo(np.complex64).eps + np.abs(np.fft.fftshift(np.fft.fft(data[1]))) ** 2)
        f = np.linspace(-fs_Hz / 2, fs_Hz / 2, len(PSD0))

        print("[INFO] plot the spectral data")
        # Time plot helps us check that we see the emitter and that we're not saturated (ie gain isnt too high)
        plt.subplot(2, 1, 1)
        plt.plot(data[0].real, label="ch0")  # Only plot real part
        plt.plot(data[1].real, label="ch1")
        plt.legend(loc='upper right')
        plt.xlabel("Data Point")
        plt.ylabel("ADC output")

        # PSDs show where the emitter is and verify both channels are working
        plt.subplot(2, 1, 2)
        plt.plot(f / 1e6, PSD0, label="ch0")
        plt.plot(f / 1e6, PSD1, label="ch1")
        plt.legend(loc='upper right')
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Signal Strength [dB]")
        
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()
        
    @staticmethod
    def test_dc_calibration(fc_Hz: float = Phaser_LO_HIGH, rx_gain_dB: int = 0, DeviceString: str = "6b5d"):
        _sdr = phaser_BladeRF(fc_Hz = fc_Hz, DeviceString = DeviceString)
        _sdr.rx_gain = rx_gain_dB
        
        # there looks to be an initial spike in the data when the frequency is changed (uncleared buffer?)
        # the easier fix is to perform an initial read an discard it
        A = _sdr.read()
        A = _sdr.read()
        plot_2channels(A)
        _sdr.calibrate_dc()
        A = _sdr.read()
        #plot_2channels(A)
        _sdr.calibrate_dc()
        _sdr.close()

if __name__ == "__main__":
    phaser_BladeRF.list()
    num_reads = 5
    DeviceString = "6b5d"
    
    phaser_BladeRF.test_dc_calibration(DeviceString=DeviceString)
    # phaser_BladeRF.read_test()
    # phaser_BladeRF.plot(Phaser_LO_HIGH, DeviceString=DeviceString)
    # phaser_BladeRF.test(num_reads=num_reads, DeviceString=DeviceString)
    # phaser_BladeRF.record(num_reads=num_reads, DeviceString=DeviceString)
