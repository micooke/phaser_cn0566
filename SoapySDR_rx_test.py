#!/usr/bin/env python3

import os
import time
import atexit

import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32, SOAPY_SDR_CS16

from phaser_utils import Phaser_LO_HIGH
from phaser_IO import phaser_SigMF, phaser_IO


# PlutoSDR class
#
# DeviceString = {rtlsdr,hackrf,plutosdr}
# Note: serial number can be passed;
# 1. instead of the device: DeviceString='serial=675c62dc32526ccf'
# 2. with the driver string: DeviceString='driver=hackrf,serial=675c62dc32526ccf'
#
# @author: Mark Cooke
class phaser_SoapySDR:
    def __init__(
        self,
        fs_Hz: int = 10_000_000,
        fc_Hz: float = Phaser_LO_HIGH,
        channel_number: int = 0,
        DeviceString: str = "hackrf",
    ):
        self.__name__ = "SoapySDR"
        self.ccal = [0.0, 0.0]
        self.default_gain_dB = 0

        self.DeviceString = DeviceString
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        self._channel = None
        self.is_streaming: bool = False
        self.rxStream = None

        self.sdr = None
        self.sigmf = None

        self.default_buffer_size = 1_024
        self.buffer = np.empty(0, np.complex64)

        self.setup(
            fs_Hz,
            fc_Hz,
            channel_number,
            self.default_gain_dB,
            self.default_buffer_size,
            DeviceString,
        )

        atexit.register(self.close)

    def close(self):
        if self.is_streaming and (self.rxStream is not None):
            self.is_streaming = False
            self.sdr.deactivateStream(self.rxStream)  # stop streaming
            self.sdr.closeStream(self.rxStream)

    # original source
    # @author: Analog Devices
    # modification: move my_sdr to my_phaser and store my_phaser in the class object
    # @author: Mark Cooke
    def setup(
        self,
        fs_Hz: int = 30_000_000,
        fc_Hz: float = Phaser_LO_HIGH,
        channel_number: int = 0,
        rx_gain_dB: int = 10,
        rx_buffer_size: int = 1_024,
        DeviceString: str = "hackrf",
    ):
        if "=" not in DeviceString:
            self.DeviceString = f"driver={DeviceString}"

        try:
            print(f"Connecting to '{self.DeviceString}' via {self.__name__}..", end=" ")
            self.sdr = SoapySDR.Device(f"{{{self.DeviceString}}}")
            print("Connected")
        except RuntimeError:
            try:
                print("Failed. Try connecting to the first available device..", end=" ")
                sdr_list = SoapySDR.Device.enumerate()
                if len(sdr_list) == 0:
                    print("Failed. No SoapySDR devices present")
                else:
                    self.DeviceString = sdr_list[0]
                    print(f"'{self.DeviceString}'..", end=" ")
                    self.sdr = SoapySDR.Device({"driver": DeviceString})
                    print("Connected")
            except RuntimeError:
                print("Failed")
                self.sdr = None
                return False

        self.channel = channel_number
        self.fc_Hz = fc_Hz
        self.fs_Hz = fs_Hz
        self.rx_bandwidth_Hz = fs_Hz // 2
        self.rx_buffer_size = rx_buffer_size
        self.rx_gain = rx_gain_dB

        if (not self.is_streaming) and (self.sdr is not None):
            self.rxStream = self.sdr.setupStream(
                SOAPY_SDR_RX, SOAPY_SDR_CS16
            )  # SOAPY_SDR_CF32)
            self.sdr.activateStream(self.rxStream)  # start streaming
            self.is_streaming = True

        RFIC_TEMP = self.sdr.listSensors()[0]

        print(f"API version: {SoapySDR.getAPIVersion()}")
        print(f"Lib version: {SoapySDR.getLibVersion()}")

        print(f"Hardware Info: {self.sdr.getHardwareInfo()}")
        print(f"Clock Sources: {self.sdr.listClockSources()}")
        print(f"Time Sources: {self.sdr.listTimeSources()}")
        print(f"Sensors: {self.sdr.listSensors()}")
        print(f"{RFIC_TEMP}: {self.sdr.readSensor(RFIC_TEMP)}")
        print(f"Sample Rates: {self.sdr.listSampleRates(SOAPY_SDR_RX, self.channel)}")
        print(
            f"Stream Formats: {self.sdr.getStreamFormats(SOAPY_SDR_RX, self.channel)}"
        )
        # fullScale = 1.
        # print(f"Native Stream Format: {self.sdr.getNativeStreamFormat(SOAPY_SDR_RX, self.channel, fullScale)}")

        # print(f"Stream Args: {self.sdr.getStreamArgsInfo(SOAPY_SDR_RX, self.channel)}")

        return True

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value: int = 0):
        self._channel = value

    @property
    def channel_cal(self):
        return self.ccal

    @channel_cal.setter
    def channel_cal(self, ccal):
        self.ccal = ccal

    @property
    def rx_gain(self):
        return self.sdr.getGain(SOAPY_SDR_RX, self._channel)

    # hackRF range: 0 to 116.0 dB
    @rx_gain.setter
    def rx_gain(self, gain_dB: int = 0):
        if (gain_dB + self.ccal[0]) != self.rx_gain:
            self.sdr.setGain(SOAPY_SDR_RX, self._channel, gain_dB + self.ccal[0])

    @property
    def rx_bandwidth_Hz(self):
        return self.sdr.getBandwidth(SOAPY_SDR_RX, self._channel)

    @rx_bandwidth_Hz.setter
    def rx_bandwidth_Hz(self, rx_bandwidth_Hz: int = 10_000_000):
        self.sdr.setBandwidth(SOAPY_SDR_RX, self._channel, int(rx_bandwidth_Hz))

    @property
    def fs_Hz(self):
        return self.sdr.getSampleRate(SOAPY_SDR_RX, self._channel)

    @fs_Hz.setter
    def fs_Hz(self, fs_Hz: float = 30_000_000):
        self.sdr.setSampleRate(SOAPY_SDR_RX, self._channel, fs_Hz)

        if self.rx_bandwidth_Hz > fs_Hz:
            self.rx_bandwidth_Hz = fs_Hz

    @property
    def rx_buffer_size(self):
        return len(self.buffer)

    @rx_buffer_size.setter
    def rx_buffer_size(self, rx_buffer_size: int = 1_024_000):
        self.buffer = np.empty(rx_buffer_size, np.complex64)

    @property
    def fc_Hz(self):
        return self.sdr.getFrequency(SOAPY_SDR_RX, self._channel)

    @fc_Hz.setter
    def fc_Hz(self, fc_Hz: float = Phaser_LO_HIGH):
        self.sdr.setFrequency(SOAPY_SDR_RX, self._channel, fc_Hz)

    def buffer_read(self):
        _read_status = self.sdr.readStream(
            self.rxStream, [self.buffer], self.rx_buffer_size
        )
        # num samples or error code, flags set by receive operation, timestamp for receive buffer
        # print(f"{_read_status.ret}, {_read_status.flags}, {_read_status.timeNs}")
        return _read_status.ret

    def read(self):
        buffer_size = self.buffer_read()

        return self.buffer[:buffer_size]

    @staticmethod
    def list():
        sdr_list = SoapySDR.Device.enumerate()
        for sdr_ in sdr_list:
            print(sdr_)

    @staticmethod
    def test(
        num_reads: int = 1,
        rx_buffer_size: int = 1_024_000,  # 2_097_144
        fc_Hz: float = Phaser_LO_HIGH,
        fs_Hz: int = 10_000_000,
        DeviceString: str = "hackrf",
    ):
        print("[INFO] test")
        sdr = phaser_SoapySDR(DeviceString=DeviceString)
        if sdr.sdr is None:
            print(f"[ERROR] {sdr.__name__} not found")
            return

        sdr.rx_buffer_size = rx_buffer_size
        data = [0] * rx_buffer_size
        num_samples = 0
        t0 = time.perf_counter_ns()
        for n in range(num_reads):
            data = sdr.read()
            num_samples += len(data)
        t1 = time.perf_counter_ns()
        sdr.close()
        print(data)

        time_delta_us = (t1 - t0) / 1_000
        read_rate_Msps = num_samples / time_delta_us
        print(
            f"read: num_samples = {num_samples:,}, time: {time_delta_us:,.3f}us, rate: {read_rate_Msps:,.3f}Msps"
        )

    @staticmethod
    def record(
        num_reads: int = 1,
        fc_Hz: float = Phaser_LO_HIGH,
        fs_Hz: int = 10_000_000,
        rx_buffer_size: int = 1_024_000,  # 2_097_144,
        DeviceString: str = "hackrf",
    ):
        print("[INFO] record")
        sdr = phaser_SoapySDR(fs_Hz=fs_Hz, fc_Hz=fc_Hz, DeviceString=DeviceString)
        if sdr.sdr is None:
            print(f"[ERROR] {sdr.__name__} not found")
            return

        sdr.rx_buffer_size = rx_buffer_size

        if sdr.sigmf is None:
            if type(sdr._channel) is list:
                sdr.sigmf = sdr._channel
                for ch in sdr._channel:
                    sdr.sigmf[ch] = phaser_SigMF()
                    sdr.sigmf[ch].open(ch, fs_Hz, fc_Hz)

                    t0 = time.perf_counter_ns()
                    num_samples = 0
                    for n in range(num_reads):
                        data = sdr.read()
                        num_samples += len(data)
                        sdr.sigmf[0].write(data[0])
                        sdr.sigmf[1].write(data[1])
                    t1 = time.perf_counter_ns()
                    sdr.close()
            else:
                ch = sdr._channel
                sdr.sigmf = [ch]
                sdr.sigmf[ch] = phaser_SigMF()
                sdr.sigmf[ch].open(ch, fs_Hz, fc_Hz)

                t0 = time.perf_counter_ns()
                num_samples = 0
                for n in range(num_reads):
                    data = sdr.read()
                    num_samples += len(data)
                    sdr.sigmf[ch].write(data)
                t1 = time.perf_counter_ns()
                sdr.close()

        time_delta_us = (t1 - t0) / 1_000

        if type(sdr._channel) is list:
            for ch in sdr._channel:
                sdr.sigmf[ch].close()
                phaser_IO.delete(sdr.sigmf[ch].base_filename + ".sigmf-data")
                phaser_IO.delete(sdr.sigmf[ch].base_filename + ".sigmf-meta")
        else:
            ch = sdr._channel
            sdr.sigmf[ch].close()
            phaser_IO.delete(sdr.sigmf[ch].base_filename + ".sigmf-data")
            phaser_IO.delete(sdr.sigmf[ch].base_filename + ".sigmf-meta")

        read_and_record_rate_Msps = num_samples / time_delta_us
        print(
            f"record: num_samples = {num_samples:,}, time: {time_delta_us:,.3f}us, rate: {read_and_record_rate_Msps:,.3f}Msps"
        )

    @staticmethod
    def plot(
        fc_Hz: float = Phaser_LO_HIGH,
        fs_Hz: int = 10_000_000,
        DeviceString: str = "hackrf",
    ):
        print("[INFO] plot")
        import matplotlib.pyplot as plt

        sdr = phaser_SoapySDR(fc_Hz=fc_Hz, fs_Hz=fs_Hz, DeviceString=DeviceString)
        if sdr.sdr is None:
            print(f"[ERROR] {sdr.__name__} not found")
            return

        sdr.rx_gain = 63  # 60 = 1500/2000
        data = sdr.read()
        sdr.close()

        # Take FFT
        PSD0 = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data))) ** 2)
        f = np.linspace(-sdr.fs_Hz / 2, sdr.fs_Hz / 2, len(data))
        # PSD0 = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data[0]))) ** 2)
        # PSD1 = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data[1]))) ** 2)
        # f = np.linspace(-sdr.fs_Hz / 2, sdr.fs_Hz / 2, len(data[0]))

        # Time plot helps us check that we see the emitter and that we're not saturated (ie gain isnt too high)
        plt.subplot(2, 1, 1)
        plt.plot(data.real, label="ch0")  # Only plot real part
        # plt.plot(data[0].real, label="ch0")  # Only plot real part
        # plt.plot(data[1].real, label="ch1")
        plt.legend()
        plt.xlabel("Data Point")
        plt.ylabel("ADC output")

        # PSDs show where the emitter is and verify both channels are working
        plt.subplot(2, 1, 2)
        plt.plot(f / 1e6, PSD0, label="ch0")
        # plt.plot(f / 1e6, PSD1, label="ch1")
        plt.legend()
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Signal Strength [dB]")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    sample_rate = 10e6
    center_freq = 100e6
    gain = 50  # -15 to 60 dB
    num_samples = int(1e6)

    phaser_SoapySDR.list()
    num_reads = 4

    DeviceString = "bladerf"
    a = phaser_SoapySDR(DeviceString="driver=bladerf,instance=0")
    # a = phaser_SoapySDR(DeviceString="driver=bladerf,sample_format=8,feature=oversample") # How do you set 8bit mode??

    phaser_SoapySDR.plot(
        fc_Hz=center_freq, fs_Hz=sample_rate, DeviceString=DeviceString
    )
    # phaser_SoapySDR.test(fc_Hz=center_freq, fs_Hz=sample_rate, num_reads=num_reads, DeviceString=DeviceString)
    # phaser_SoapySDR.record(fc_Hz=center_freq, fs_Hz=sample_rate, num_reads=num_reads, DeviceString=DeviceString)


# [INFO] record
# plutosdr issues
# [WARNING] Unable to scan local: -19
# [WARNING] Unable to scan ip: -19
#
# Connecting to 'driver=hackrf' via SoapySDR.. Connected
# [0.+0.j 0.+0.j 0.+0.j ... 0.+0.j 0.+0.j 0.+0.j]
# read: num_samples = 655,360, time: 64,950.900us, rate: 10.090Msps
# record: num_samples = 655,360, time: 63,205.100us, rate: 10.369Msps
#
