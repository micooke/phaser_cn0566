#!/usr/bin/env python3

# USB/USB-Ethernet Control of the phaser receiver: PlutoSDR

import os
import time
import atexit

import numpy as np
from adi import ad9361

from phaser_utils import Phaser_LO_HIGH
from phaser_IO import phaser_SigMF, phaser_IO

# phaser class
# @author: Mark Cooke
class phaser_PlutoSDR:
    def __init__(
        self,
        fs_Hz: int = 30_000_000,
        fc_Hz: float = Phaser_LO_HIGH,
        channel_number: int = [0,1],
        DeviceString: str = "192.168.2.1",
    ):
        self.__name__ = "PlutoSDR"
        self.ccal = [0.0, 0.0]
        self.default_gain_dB = 0
        self.rx_gain_limit = [0.0, 74.5]
        self.tx_gain_limit = [-90.0, 0]
                
        self.monitor_ch_names = [
            "Temp (deg C): ",
            "CH1 RSSI(dB): ",
            "CH2 RSSI(dB): ",
        ]

        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        self._channel = None
        # self.is_streaming: bool = False

        self.sdr = None
        self.sigmf = None
        
        self.default_buffer_size = 1_024
        # self.buffer = np.empty(0, np.complex64)
        
        self.setup(fs_Hz, fc_Hz, channel_number, self.default_gain_dB, self.default_buffer_size, DeviceString)

        atexit.register(self.close)

    def close(self):
        # self.sdr.rx_destroy_buffer()
        # self.sdr.tx_destroy_buffer()
        pass

    # original source
    # @author: Analog Devices
    # modification: move my_sdr to my_phaser and store my_phaser in the class object
    # @author: Mark Cooke
    def setup(
        self,
        fs_Hz: int = 30_000_000,
        fc_Hz: float = Phaser_LO_HIGH,
        channel_number: int = [0,1],
        rx_gain_dB: int = 10,
        rx_buffer_size: int = 1_024,
        DeviceString: str = "ip:192.168.2.1",
    ):
        self.DeviceString = DeviceString
        if DeviceString != "ip:phaser.local:50901":
            alt_DeviceString = "ip:phaser.local:50901"
        else:
            alt_DeviceString = "ip:192.168.2.1"
       
        # First try to connect to a locally connected device. On success, connect,
        # on failure, connect to remote device
        try:
            print(f"Connecting to '{self.__name__}' via '{self.DeviceString}'...", end=" ")
            self.sdr = ad9361(uri=self.DeviceString)
            print("Connected")
        except Exception:  # pyadi-iio raises this error
            try:
                self.DeviceString = alt_DeviceString
                print(f"Failed. Connecting via '{self.DeviceString}'...", end=" ")
                self.sdr = ad9361(uri=self.DeviceString)
                print("Connected")
            except Exception:
                print("Failed")
                self.sdr = None
                return False

        time.sleep(0.5)
        self._channel = channel_number

        #  Configure SDR parameters.
        self.sdr._ctrl.debug_attrs[
            "adi,frequency-division-duplex-mode-enable"
        ].value = "1"
        self.sdr._ctrl.debug_attrs[
            "adi,ensm-enable-txnrx-control-enable"
        ].value = "0"  # Disable pin control so spi can move the states
        self.sdr._ctrl.debug_attrs["initialize"].value = "1"

        self.sdr.rx_enabled_channels = channel_number
         # enable Rx1 (voltage0) and Rx2 (voltage1)
        self.sdr._rxadc.set_kernel_buffers_count(1)  # No stale buffers to flush
        rx = self.sdr._ctrl.find_channel("voltage0")
        rx.attrs["quadrature_tracking_en"].value = "1"  # enable quadrature tracking

        self.fc_Hz = int(fc_Hz)  # Downconvert to ~2GHz | Receive Freq
        self.fs_Hz = int(fs_Hz)  # Sampling rate
        self.rx_bandwidth_Hz = int(10e6) # int(fs_Hz // 2)# int(10e6)
        self.rx_buffer_size = int(rx_buffer_size)
        self.rx_gain = rx_gain_dB

        # Handy filter for fairly wideband measurements
        self.sdr.filter = os.path.join(self.base_dir, "LTE20_MHz.ftr")

        # Make sure the Tx channels are attenuated (or off) and their freq is far away from Rx
        # this is a negative number between 0 and -88 (-90 according to pysdr.org)
        self.sdr.tx_hardwaregain_chan0 = int(-88)
        self.sdr.tx_hardwaregain_chan1 = int(-88)

        return True

    @property
    def channel_cal(self):
        return self.ccal
    
    @channel_cal.setter
    def channel_cal(self, ccal):
        self.ccal = ccal

    @property
    def rx_gain(self):
        return [self.sdr.rx_hardwaregain_chan0, self.sdr.rx_hardwaregain_chan1]

    # plutoSDR range: 0 to 74.5 dB
    @rx_gain.setter
    def rx_gain(self, gain_dB: float = 0, gain_mode:str = 'manual', apply_cal:bool = True):
        gain_dB = np.clip(gain_dB, self.rx_gain_limit[0], self.rx_gain_limit[1]) # bound the gain

        if gain_mode in ['manual', 'slow_attack', 'fast_attack']:
            if self.sdr.gain_control_mode_chan0 != gain_mode:
                self.sdr.gain_control_mode_chan0 = gain_mode
                self.sdr.gain_control_mode_chan1 = gain_mode
        else:
            print(f"[ERROR] unknown gain control mode {gain_mode}")
            return

        if gain_mode == 'manual':
            if apply_cal:
                self.sdr.rx_hardwaregain_chan0 = gain_dB + self.ccal[0]
                self.sdr.rx_hardwaregain_chan1 = gain_dB + self.ccal[1]
            else:
                self.sdr.rx_hardwaregain_chan0 = gain_dB
                self.sdr.rx_hardwaregain_chan1 = gain_dB
    @property
    def rx_bandwidth_Hz(self):
        return self.sdr.rx_rf_bandwidth

    @rx_bandwidth_Hz.setter
    def rx_bandwidth_Hz(self, rx_bandwidth_Hz: int = 10_000_000):
        if rx_bandwidth_Hz != self.sdr.rx_rf_bandwidth:
            self.sdr.rx_rf_bandwidth = int(rx_bandwidth_Hz)
    
    @property
    def fs_Hz(self):
        return self.sdr.sample_rate

    @fs_Hz.setter
    def fs_Hz(self, fs_Hz: float = 30_000_000):
        self.sdr.sample_rate = int(fs_Hz)  # Sampling rate
        
        if self.rx_bandwidth_Hz > fs_Hz:
            self.rx_bandwidth_Hz = fs_Hz

    @property
    def rx_buffer_size(self):
        return self.sdr.rx_buffer_size
    
    @rx_buffer_size.setter
    def rx_buffer_size(self, rx_buffer_size: int = 1024):
        if rx_buffer_size > 2_097_144:
            rx_buffer_size = 2_097_144
            print(
                f"[WARNING] requested rx_buffer_size exceeds the max. It will be set to {rx_buffer_size:,}"
            )
        elif rx_buffer_size < 32:
            rx_buffer_size = 32
            print(
                f"[WARNING] requested rx_buffer_size is too low. It will be set to {rx_buffer_size:,}"
            )

        if rx_buffer_size != self.sdr.rx_buffer_size:
            # A call to rx() sets the buffer size until it is destroyed
            self.sdr.rx_destroy_buffer()  # need to destroy the buffer, prior to resizing
            self.sdr.rx_buffer_size = int(rx_buffer_size)

    @property
    def fc_Hz(self):
        return self.rx_lo_Hz

    # in our context, the LO is actually the IF.
    # The PlutoSDR AD936x chip digitizes this at baseband after downconversion
    @fc_Hz.setter
    def fc_Hz(self, rx_lo_Hz: float = Phaser_LO_HIGH):
        self.rx_lo_Hz = rx_lo_Hz
        self.sdr.rx_lo = int(self.rx_lo_Hz)
        # print(f"PlutoSDR [LO] (MHz) = [{int(self.rx_lo_Hz)/1e6:,.0f}]")

    def read(self, rx_buffer_size = None):
        if rx_buffer_size is not None:
            self.rx_buffer_size = rx_buffer_size
        return self.sdr.rx()
    
    @staticmethod
    def list():
        print("[ERROR] 'list()' function is not implemented")

    @staticmethod
    def test(num_reads: int = 1, rx_buffer_size: int = 1_024_000, #2_097_144
             DeviceString: str = "192.168.2.1"
             ):
        print("[INFO] test")
        sdr = phaser_PlutoSDR(DeviceString = DeviceString)
        if sdr.sdr is None:
            print(f"[ERROR] {sdr.__name__} not found")
            return
        
        sdr.rx_buffer_size = rx_buffer_size
        data = [0]*rx_buffer_size
        num_samples = 0
        t0 = time.perf_counter_ns()
        for n in range(num_reads):
            data = sdr.read()
            num_samples += len(data)
        t1 = time.perf_counter_ns()
        sdr.close()
        print(data)

        time_delta_us = (t1-t0)/1_000
        read_rate_Msps = num_samples / time_delta_us
        print(f"read: num_samples = {num_samples:,}, time: {time_delta_us:,.3f}us, rate: {read_rate_Msps:,.3f}Msps")
    
    @staticmethod
    def record(
        num_reads: int = 1,
        fc_Hz: float = Phaser_LO_HIGH,
        fs_Hz: int = 10_000_000,
        rx_buffer_size: int = 1_024_000, #2_097_144,
        DeviceString: str = "192.168.2.1"
    ):
        print("[INFO] record")
        sdr = phaser_PlutoSDR(fs_Hz = fs_Hz, fc_Hz = fc_Hz, DeviceString = DeviceString)
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

        time_delta_us = (t1-t0)/1_000

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
        print(f"record: num_samples = {num_samples:,}, time: {time_delta_us:,.3f}us, rate: {read_and_record_rate_Msps:,.3f}Msps")
       
    @staticmethod
    def plot(fc_Hz: float = Phaser_LO_HIGH, DeviceString: str = "192.168.2.1"):
        print("[INFO] plot")
        import matplotlib.pyplot as plt
        sdr = phaser_PlutoSDR(fc_Hz=fc_Hz, DeviceString = DeviceString)
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

    def status(self, verbose:bool = False):
        # note: get attributes list from:
        # https://wiki.analog.com/resources/tools-software/linux-drivers/iio-transceiver/ad9361
        # e.g. in_temp0_input = sdr._get_iio_attr('temp0','input',False)
        if verbose:
            print("Reading status...")
        monitor_vals = []
        monitor_vals.append(float(self.sdr._get_iio_attr('temp0','input',False)) / 1000) # as an integer: 12345 = 12.345 deg C
        monitor_vals.append(self.sdr._get_iio_attr('voltage0','rssi',False)) # channel 1 received signal strength indicator (in dB)
        monitor_vals.append(self.sdr._get_iio_attr('voltage1','rssi',False)) # channel 2 received signal strength indicator (in dB)

        if verbose:
            print(monitor_vals)
        
        return monitor_vals

if __name__ == '__main__':
    phaser_PlutoSDR.list()
    num_reads: int=5
    DeviceString: str = "192.168.2.1" # "192.168.2.2"
    phaser_PlutoSDR.plot(1.6e9, DeviceString=DeviceString)
    # phaser_PlutoSDR.test(num_reads=num_reads, DeviceString=DeviceString)
    # phaser_PlutoSDR.record(num_reads=num_reads, DeviceString=DeviceString)
    phaser_PlutoSDR.status()

    # default 192.168.1.50