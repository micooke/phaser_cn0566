#!/usr/bin/env python3

# USB/USB-Ethernet Control of the phaser receiver: PlutoSDR

import os
import time
import atexit

import numpy as np
from adi import ad9361

from phaser_utils import *


# phaser class
# @author: Mark Cooke
class phaser_PlutoSDR:
    def __init__(
        self,
        fs_Hz: int = 30_000_000,
        rx_lo_Hz: float = Phaser_LO_HIGH,
        PlutoSDR_ip: str = "192.168.2.11",
    ):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.fs_Hz: int = fs_Hz
        self.rx_lo_Hz: float = rx_lo_Hz
        self.rx_gain_dB: int = 0

        self.plutoSDR = None

        self.ccal = [0.0, 0.0]

        self.setup(fs_Hz, rx_lo_Hz, self.rx_gain_dB, PlutoSDR_ip)

        atexit.register(self.close)

    def close(self):
        # self.plutoSDR.rx_destroy_buffer()
        # self.plutoSDR.tx_destroy_buffer()
        pass

    # original source
    # @author: Analog Devices
    # modification: move my_sdr to my_phaser and store my_phaser in the class object
    # @author: Mark Cooke
    def setup(
        self,
        fs_Hz: int = 30_000_000,
        rx_lo_Hz: float = Phaser_LO_HIGH,
        rx_gain_dB: int = 10,
        PlutoSDR_ip: str = "192.168.2.11",
    ):
        self.fs_Hz = fs_Hz
        self.rx_lo_Hz = rx_lo_Hz

        try:
            print(f"Attempting to connect to PlutoSDR via {PlutoSDR_ip}..")
            self.plutoSDR = ad9361(uri=f"ip:{PlutoSDR_ip}")
            print(f"PlutoSDR connected on '{PlutoSDR_ip}'")
        except:
            try:
                print("Failed. Connecting via ip:phaser.local:50901...")
                self.plutoSDR = ad9361(uri="ip:phaser.local:50901")
                print("PlutoSDR connected on 'phaser.local:50901'")
            except:
                print(f"[ERROR] cannot connect to PlutoSDR on {PlutoSDR_ip}")
                return False

        time.sleep(0.5)

        #  Configure SDR parameters.
        self.plutoSDR._ctrl.debug_attrs[
            "adi,frequency-division-duplex-mode-enable"
        ].value = "1"
        self.plutoSDR._ctrl.debug_attrs[
            "adi,ensm-enable-txnrx-control-enable"
        ].value = "0"  # Disable pin control so spi can move the states
        self.plutoSDR._ctrl.debug_attrs["initialize"].value = "1"

        self.plutoSDR.rx_enabled_channels = [
            0,
            1,
        ]  # enable Rx1 (voltage0) and Rx2 (voltage1)
        self.plutoSDR._rxadc.set_kernel_buffers_count(1)  # No stale buffers to flush
        rx = self.plutoSDR._ctrl.find_channel("voltage0")
        rx.attrs["quadrature_tracking_en"].value = "1"  # enable quadrature tracking
        self.plutoSDR.sample_rate = int(self.fs_Hz)  # Sampling rate
        self.plutoSDR.rx_buffer_size = int(1024)
        self.plutoSDR.rx_rf_bandwidth = int(10e6)
        # We must be in manual gain control mode (otherwise we won't see the peaks and nulls!)
        self.plutoSDR.gain_control_mode_chan0 = (
            "manual"  # Disable AGC | 'slow_attack', 'fast_attack'
        )
        self.plutoSDR.gain_control_mode_chan1 = "manual"

        self.set_rx_gain(rx_gain_dB)

        self.plutoSDR.rx_lo = int(self.rx_lo_Hz)  # Downconvert to 2GHz  # Receive Freq

        # Handy filter for fairly wideband measurements
        self.plutoSDR.filter = os.path.join(self.base_dir, "LTE20_MHz.ftr")

        # Make sure the Tx channels are attenuated (or off) and their freq is far away from Rx
        # this is a negative number between 0 and -88 (-90 according to another source)
        self.plutoSDR.tx_hardwaregain_chan0 = int(-88)
        self.plutoSDR.tx_hardwaregain_chan1 = int(-88)

        return True

    def set_channel_cal(self, ccal):
        if ccal != self.ccal:
            self.ccal = ccal

    # range: 0 to 74.5 dB (its actually attenuation)
    def set_rx_gain(self, gain_dB: int = 0):
        if gain_dB != self.rx_gain_dB:
            self.rx_gain_dB = gain_dB
            self.plutoSDR.rx_hardwaregain_chan0 = gain_dB + self.ccal[0]
            self.plutoSDR.rx_hardwaregain_chan1 = gain_dB + self.ccal[1]

    def set_rx_bandwidth_Hz(self, rx_bandwidth_Hz: int = 10_000_000):
        if rx_bandwidth_Hz != self.plutoSDR.rx_rf_bandwidth:
            self.plutoSDR.rx_rf_bandwidth = int(rx_bandwidth_Hz)

    def set_sample_frequency_Hz(self, fs_Hz: float = 30_000_000):
        if fs_Hz != self.fs_Hz:
            self.fs_Hz = fs_Hz
            self.plutoSDR.sample_rate = int(self.fs_Hz)  # Sampling rate
            nyquist_rate = self.fs_Hz // 2
            if self.plutoSDR.rx_rf_bandwidth > nyquist_rate:
                self.set_rx_bandwidth_Hz(nyquist_rate)

    def set_rx_buffer_size(self, rx_buffer_size: int = 1024):
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

        if rx_buffer_size != self.plutoSDR.rx_buffer_size:
            # A call to rx() sets the buffer size until it is destroyed
            self.plutoSDR.rx_destroy_buffer()  # need to destroy the buffer, prior to resizing
            self.plutoSDR.rx_buffer_size = int(rx_buffer_size)

    # in our context, the LO is actually the IF.
    # The PlutoSDR AD936x chip digitizes this at baseband after downconvesion
    def set_LO_Hz(self, rx_lo_Hz: float = Phaser_LO_HIGH):
        self.rx_lo_Hz = rx_lo_Hz
        self.plutoSDR.rx_lo = int(self.rx_lo_Hz)
        # print(f"PlutoSDR [LO] (MHz) = [{int(self.rx_lo_Hz)/1e6:,.0f}]")

    def rx(self):
        return self.plutoSDR.rx()
