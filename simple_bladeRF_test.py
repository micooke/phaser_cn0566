## bladeRF supports 8b 122.88Msps mode from 2023.02 build
# * FPGA bitstream                v0.15.0
# * FX3 firmware                  v2.4.0
# * libbladeRF                    v2.5.0
# * bladeRF-cli                   v1.9.0
# * MATLAB & Simulink bindings    v1.0.5
# * Python bindings               v1.3.0
#

from bladerf import _bladerf
import time
import numpy as np
import matplotlib.pyplot as plt

## SETUP
sdr = _bladerf.BladeRF()

print("Device info:", _bladerf.get_device_list()[0])
print("libbladeRF version:", _bladerf.version())  # v2.5.0
print("Firmware version:", sdr.get_fw_version())  # v2.4.0
print("FPGA version:", sdr.get_fpga_version())  # v0.15.3

rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(0))  # give it a 0 or 1
print("sample_rate_range:", rx_ch.sample_rate_range)
print("bandwidth_range:", rx_ch.bandwidth_range)
print("frequency_range:", rx_ch.frequency_range)
print("gain_modes:", rx_ch.gain_modes)
print("manual gain range:", sdr.get_gain_range(_bladerf.CHANNEL_RX(0)))  # ch 0 or 1

buffer_size = 1_024 * 8
dont_plot = True

sample_rate = 10e6
center_freq = 100e6
gain = 50  # -15 to 60 dB
num_samples = int(1e6)

rx_ch.frequency = center_freq
rx_ch.sample_rate = sample_rate
rx_ch.bandwidth = sample_rate / 2
rx_ch.gain_mode = _bladerf.GainMode.Manual
rx_ch.gain = gain

## RECEIVE
# Setup synchronous stream
sdr.sync_config(
    layout=_bladerf.ChannelLayout.RX_X1,  # or RX_X2
    fmt=_bladerf.Format.SC16_Q11,  # int16s
    num_buffers=16,
    buffer_size=buffer_size,
    num_transfers=8,
    stream_timeout=3500,
)

# Create receive buffer
bytes_per_sample = 4  # don't change this, it will always use int16s
buf = bytearray(1024 * bytes_per_sample)

# Enable module
print("Starting receive")
rx_ch.enable = True

# Receive loop
x = np.zeros(num_samples, dtype=np.complex64)  # storage for IQ samples
num_samples_read = 0
t0 = time.perf_counter_ns()
while True:
    if num_samples > 0 and num_samples_read == num_samples:
        break
    elif num_samples > 0:
        num = min(len(buf) // bytes_per_sample, num_samples - num_samples_read)
    else:
        num = len(buf) // bytes_per_sample
    sdr.sync_rx(buf, num)  # Read into buffer
    samples = np.frombuffer(buf, dtype=np.int16)
    samples = samples[0::2] + 1j * samples[1::2]  # Convert to complex type
    samples /= 2048.0  # Scale to -1 to 1 (its using 12 bit ADC)
    x[num_samples_read : num_samples_read + num] = samples[
        0:num
    ]  # Store buf in samples array
    num_samples_read += num
t1 = time.perf_counter_ns()
print("Stopping")

time_delta_us = (t1 - t0) / 1_000
read_rate_Msps = num_samples_read / time_delta_us
print(
    f"read: num_samples = {num_samples_read:,}, time: {time_delta_us:,.3f}us, rate: {read_rate_Msps:,.3f}Msps"
)

rx_ch.enable = False
print(x[0:10])  # look at first 10 IQ samples
print(
    np.max(x)
)  # if this is close to 1, you are overloading the ADC, and should reduce the gain

## PLOT
# Create spectrogram
if not dont_plot:
    fft_size = 2048
    num_rows = len(x) // fft_size  # // is an integer division which rounds down
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i, :] = 10 * np.log10(
            np.abs(np.fft.fftshift(np.fft.fft(x[i * fft_size : (i + 1) * fft_size])))
            ** 2
        )
    extent = [
        (center_freq + sample_rate / -2) / 1e6,
        (center_freq + sample_rate / 2) / 1e6,
        len(x) / sample_rate,
        0,
    ]
    plt.imshow(spectrogram, aspect="auto", extent=extent)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Time [s]")
    plt.show()
