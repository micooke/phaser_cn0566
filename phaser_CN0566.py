#!/usr/bin/env python3

# SPI Control of the phaser beamsteering & X-band DDC: CN0566 (including 2xADAR1000)
# For the rpi4

import os
import time
import atexit

import numpy as np
from adi import ad9361
from adi.cn0566 import CN0566

from phaser_utils import *

nominal_hb100_freq_Hz: float = 10.525e9


# phaser class
# @author: Mark Cooke
class phaser_CN0566:
    def __init__(
        self,
        fc_Hz: float = nominal_hb100_freq_Hz,
        rx_lo_Hz: float = Phaser_LO_HIGH,
        CN0566_ip: str = "ip:phaser.local",
    ):
        #                    temp   1.8V 3.0    3.3   4.5   15?   USB   curr. Vtune
        self.monitor_hi_limits = [60.0, 1.85, 3.15, 3.45, 4.75, 16.0, 5.25, 1.6, 14.0]
        self.monitor_lo_limts = [20.0, 1.75, 2.850, 3.15, 4.25, 13.0, 4.75, 1.2, 1.0]
        self.monitor_ch_names = [
            "Board temperature: ",
            "1.8V supply: ",
            "3.0V supply: ",
            "3.3V supply: ",
            "4.5V supply: ",
            "Vtune amp supply: ",
            "USB C input supply: ",
            "Board current: ",
            "VTune: ",
        ]

        self.fc_Hz = fc_Hz
        self.cn0566 = None
        self.taper_type = "box"

        self.rx_lo_offset = 0  # 100e3

        if self.setup(rx_lo_Hz, CN0566_ip):
            self.set_frequency_Hz(fc_Hz)

        atexit.register(self.close)

    def close(self):
        pass

    def setup(
        self, rx_lo_Hz: float = Phaser_LO_HIGH, CN0566_ip: str = "ip:phaser.local"
    ):
        # First try to connect to a locally connected CN0566. On success, connect,
        # on failure, connect to remote CN0566
        try:
            print(f"Attempting to connect to CN0566 via {CN0566_ip}...")
            self.cn0566 = CN0566(uri=f"ip:{CN0566_ip}")
            print(f"CN0566 connected on '{CN0566_ip}'")
        except:
            try:
                print("Failed. Connecting via ip:localhost...")
                self.cn0566 = CN0566(uri="ip:localhost")
                print("CN0566 connected on 'ip:localhost'")
            except:
                print(f"[ERROR] cannot connect to CN0566 on {CN0566_ip}")
                return False

        time.sleep(0.5)  # needed?

        # By default device_mode is "rx"
        self.cn0566.configure(device_mode="rx")

        self.rx_lo_Hz = int(rx_lo_Hz)  # Receive Freq

        # Set initial PLL frequency to HB100 nominal
        self.set_frequency_Hz(self.fc_Hz)

        gain_list = [64] * 8  # (64 is about half scale)
        for i in range(0, len(gain_list)):
            self.cn0566.set_chan_gain(i, gain_list[i], apply_cal=False)

        # Aim the beam at boresight (zero degrees). Place HB100 right in front of array.
        self.cn0566.set_beam_phase_diff(0.0)

        # Averages decide number of time samples are taken to plot and/or calibrate system. By default it is 1.
        self.cn0566.Averages = 8

        return True

    # @author: Mark Cooke
    # set the RF frequency
    def set_frequency_Hz(
        self,
        fc_Hz: float = nominal_hb100_freq_Hz,
        limit_freq: bool = False,
        verbose: bool = False,
    ):
        new_lo = None
        freq_changed: bool = True

        if limit_freq:
            if fc_Hz > 10.6e9:
                self.fc_Hz = 10.6e9
                print(
                    "[WARNING] CN0566 frequency range is 8GHz to 10.6GHz. Limiting Fc to 10.6GHz"
                )
            elif fc_Hz < 8e9:
                self.fc_Hz = 8e9
                print(
                    "[WARNING] CN0566 frequency range is 8GHz to 10.6GHz. Limiting Fc to 8GHz"
                )

        if fc_Hz != self.fc_Hz:
            self.fc_Hz = fc_Hz
        else:
            freq_changed = False

        if (self.fc_Hz <= 9.0e9) and (self.rx_lo_Hz != Phaser_LO_LOW):
            new_lo = Phaser_LO_LOW
            if verbose:
                print(
                    f"[WARNING] PlutoSDR Rx LO needs to be set to {self.rx_lo_Hz/1e9:.1f}GHz for {self.fc_Hz/1e9:.1f}GHz tuned frequency"
                )
        elif (self.fc_Hz > 9.0e9) and (self.rx_lo_Hz != Phaser_LO_HIGH):
            new_lo = Phaser_LO_HIGH
            if verbose:
                print(
                    f"[WARNING] PlutoSDR Rx LO needs to be set to {self.rx_lo_Hz/1e9:.1f}GHz for {self.fc_Hz/1e9:.1f}GHz tuned frequency"
                )

        if new_lo is not None:
            self.rx_lo_Hz = new_lo
            freq_changed = True

        # print(f"[fc, LO] (MHz) = [{int(self.fc_Hz)/1e6:,.0f}, {int(self.rx_lo_Hz)/1e6:,.0f}]")

        if freq_changed:
            ## depending on the ADI example, either the lo or the frequency is set.
            ## Until further investigation is undertaken, its a roll of the dice on which is required

            # Change the ADF4159 PLL
            self.cn0566.lo = int(self.fc_Hz + self.rx_lo_Hz + self.rx_lo_offset)

            # Change the HMC735 VCO feeback to the ADF4159
            # self.cn0566.frequency = (
            #     int(self.fc_Hz + self.rx_lo_Hz + self.rx_lo_offset) // 4
            # )

        return new_lo

    # @author: Mark Cooke
    # set the ADAR1000 taper (default is no tapering)
    def set_taper(self, taper_type: str = "box"):
        print(f"[INFO] set_taper '{taper_type}'")

        if taper_type != self.taper_type:
            self.taper_type = taper_type

            # set the taper
            if str.lower(taper_type) == "chebychev":
                self.rx_gain = [4, 23, 62, 100, 100, 62, 23, 4]
            elif str.lower(taper_type) == "hamming":
                self.rx_gain = [9, 27, 67, 100, 100, 67, 27, 9]
            elif str.lower(taper_type) == "hanning":
                self.rx_gain = [12, 43, 77, 100, 100, 77, 43, 12]
            elif str.lower(taper_type) == "blackman":
                self.rx_gain = [6, 27, 66, 100, 100, 66, 27, 6]
            else:
                self.rx_gain = [100] * 8

            for i, gain_ in enumerate(self.rx_gain):
                self.cn0566.set_chan_gain(i, gain_, apply_cal=True)

    # @author: Mark Cooke
    # TODO: log this to a file, periodically call it
    def status(self):
        print("Reading voltage monitor...")
        monitor_vals = self.cn0566.read_monitor()

        for i in range(0, len(monitor_vals)):
            if not (
                self.monitor_lo_limts[i] <= monitor_vals[i] <= self.monitor_hi_limits[i]
            ):
                print("Fails ", monitor_ch_names[i], ": ", monitor_vals[i])
                failures.append(
                    "Monitor fails "
                    + self.monitor_ch_names[i]
                    + ": "
                    + str(monitor_vals[i])
                )
            else:
                print("Passes ", self.monitor_ch_names[i], monitor_vals[i])

    # @author: Mark Cooke
    # beamsteering - set the angle of the mainlobe
    def set_angle_deg(self):
        print("[TODO]")


if __name__ == "__main__":
    cn0566 = phaser_CN0566()
    cn0566.setup()
    cn0566.set_taper("hanning")
    # cn0566.find_emitter()
    cn0566.set_frequency_Hz()
