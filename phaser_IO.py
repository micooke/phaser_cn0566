#!/usr/bin/env python3

# USB/USB-Ethernet Control of the phaser receiver: PlutoSDR

import io
import os
import time
import atexit

import numpy as np

from codetiming import Timer

import pickle
import json
from sigmf import SigMFFile
from sigmf.utils import get_data_type_str, get_sigmf_iso8601_datetime_now

dummy_library = {
    "name": ["emitter1", "emitter2", "hb100", "gravity_hb100"],
    "freq_MHz": [9_400, 10_000, 10_525, 10_420],
}


class phaser_IO:
    def __init__(self):
        pass

    @staticmethod
    def base_dir():
        return os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def save_txt(data, filename: str = "dummy.txt"):
        with open(filename, "w") as fb:
            fb.writelines(data)

    @staticmethod
    def load_txt(filename: str = "dummy.txt"):
        with open(filename, "r") as fb:
            data = fb.readlines()
        return data

    @staticmethod
    def save_json(data, filename: str = "config.json"):
        # Serializing json
        json_object = json.dumps(data, indent=3)

        with open(filename, "w") as fb:
            fb.write(json_object)

    @staticmethod
    def load_json(filename: str = "config.json"):
        with open(filename, "r") as fb:
            data = json.load(fb)
        return data

    @staticmethod
    def save_pkl(data, filename="hb100_freq_val.pkl"):
        with open(filename, "wb") as fb:
            pickle.dump(data, fb)  # save calibrated gain value to a file

    @staticmethod
    def load_pkl(filename="hb100_freq_val.pkl"):
        try:
            with open(filename, "rb") as fb:
                data = pickle.load(fb)  # Load gain cal values
        except Exception:
            print(f"[ERROR] File not found: {filename}")
            data = None
        return data

    @staticmethod
    def delete(filename):
        if os.path.isfile(filename):
            os.remove(filename)
        else:
            print(f"[ERROR] file not found '{filename}'")

    @staticmethod
    def test():
        print("[TEST] save/load pkl")
        phaser_IO.save_pkl(dummy_library, "dummy.pkl")
        print(phaser_IO.load_pkl("dummy.pkl"))
        phaser_IO.delete("dummy.pkl")
        print("[TEST] save/load json")
        phaser_IO.save_json(dummy_library, "dummy.json")
        print(phaser_IO.load_json("dummy.json"))
        phaser_IO.delete("dummy.json")


class phaser_SigMF:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.first_record = None

        self.channel_num = 0
        self.fs_Hz = 30_000_000
        self.fc_Hz = 10.525e9

        self.base_filename = ""
        self.capture_index = 0
        self.sigmf_data_fb = None
        self.sigmf_metadata = None

        atexit.register(self.close)

    def __str__(self):
        # handle = sigmf.sigmffile.fromfile(self.base_filename + ".sigmf-data")
        handle = SigMFFile.fromfile(self.base_filename + ".sigmf-data")
        handle.read_samples()

        print(f"filename: {self.base_filename}.sigmf-data")
        print(f"total samples: {len(handle)}")
        print(f"num frames: {len(handle)/2048}")
        print(handle.get_global_info())

    def open(
        self,
        channel_num: int = 0,
        fs_Hz: float = 30_000_000,
        fc_Hz: float = 10.525e9,
        buffer_size: int = 8192,
    ):
        self.channel_num = channel_num
        self.fs_Hz = fs_Hz
        self.fc_Hz = fc_Hz

        self.first_record = get_sigmf_iso8601_datetime_now()
        self.base_filename = os.path.join(
            self.base_dir, f"phaser_CH{self.channel_num}_{self.first_record}"
        )
        self.sigmf_data_fb = open(
            self.base_filename + ".sigmf-data", "wb", buffering=buffer_size
        )
        iq_data = np.array([0 + 0j], dtype=np.complex64)
        # write empty data for the first sample (due to error 'cannot mmap to empty file')
        self.sigmf_data_fb.write(iq_data)
        self.sigmf_data_fb.close()

        # @TODO(cookem): do this manually, otherwise the meta file has to be written when closing the data file
        # either that or just write sigmf-data as binary, and sigmf-meta
        # as a dict manually without the sigmf function calls
        self.sigmf_metadata = SigMFFile(
            data_file=self.base_filename + ".sigmf-data",
            global_info={
                SigMFFile.DATATYPE_KEY: get_data_type_str(iq_data),
                SigMFFile.SAMPLE_RATE_KEY: self.fs_Hz,
                SigMFFile.AUTHOR_KEY: "Phaser_CN0566",
                SigMFFile.DESCRIPTION_KEY: f"recording taken {self.first_record}",
            },
            skip_checksum=True,
        )
        # self.sigmf_data_fb = open(self.base_filename + ".sigmf-data", "wb", buffering=2*2048)
        self.sigmf_data_fb = open(self.base_filename + ".sigmf-data", "wb")

    def close(self):
        if self.sigmf_metadata is not None:
            self.sigmf_data_fb.close()
            self.sigmf_metadata.tofile(self.base_filename + ".sigmf-meta")

    def write(self, data):
        dt_now = get_sigmf_iso8601_datetime_now()

        # write to the binary file
        self.sigmf_data_fb.write(data)

        # add the capture
        self.sigmf_metadata.add_capture(
            self.capture_index,
            metadata={
                SigMFFile.FREQUENCY_KEY: self.fc_Hz,
                SigMFFile.DATETIME_KEY: dt_now,
                # note: can add lat,lon etc here
            },
        )
        self.capture_index += 1

    #  example on how to profile a class function
    def profile_write(self, BufferSize: int = 8_196):
        from line_profiler import LineProfiler

        print(f"Default Buffer Size: {io.DEFAULT_BUFFER_SIZE:,}")
        self.open()
        data = np.zeros((2, 1024), dtype=np.float64)

        lp = LineProfiler()
        lp_wrapper = lp(self.write)
        lp_wrapper(data[self.channel_num])
        lp.print_stats()

        self.close()

        phaser_IO.delete(self.base_filename + ".sigmf-data")
        phaser_IO.delete(self.base_filename + ".sigmf-meta")

    @Timer(name="write_test", text="{name}: {milliseconds:,.3f}ms")
    def write_test(self, BufferSize: int = 8_196):
        io.DEFAULT_BUFFER_SIZE = BufferSize

        print(f"Buffer Size: {io.DEFAULT_BUFFER_SIZE:,}", end=" ")
        self.open()
        data = np.zeros((2, 1024), dtype=np.float64)
        self.write(data[self.channel_num])
        self.close()

        phaser_IO.delete(self.base_filename + ".sigmf-data")
        phaser_IO.delete(self.base_filename + ".sigmf-meta")

    def buffer_test(self):
        for n in range(31):
            self.write_test((n + 1) * 1024)

    @Timer(name="threaded_test", text="{name}: {milliseconds:,.3f}ms")
    def threading_test(self):
        import threading

        t1 = threading.Thread(None)
        t1.start()
        print(f"Default Buffer Size: {io.DEFAULT_BUFFER_SIZE:,}")
        self.open()
        for n in range(1_000):
            data = np.zeros((2, 1024), dtype=np.float64)
            t1.join()
            t1 = threading.Thread(target=self.write, args=(data[self.channel_num],))
            t1.start()
        t1.join()
        self.close()

        phaser_IO.delete(self.base_filename + ".sigmf-data")
        phaser_IO.delete(self.base_filename + ".sigmf-meta")


def profile_test():
    from line_profiler import LineProfiler

    lp = LineProfiler()
    lp_wrapper = lp(_sigmf.test)
    lp_wrapper()
    lp.print_stats()


def time_write_per_buffer_size():
    #     1,024 buffer_size: 49,588,237.877us =  0.236us/sample, max data rate 4.229Msps
    #     2,048 buffer_size: 47,220,267.783us =  0.225us/sample, max data rate 4.441Msps
    #     4,096 buffer_size: 52,771,696.096us =  0.252us/sample, max data rate 3.974Msps
    #     8,192 buffer_size: 47,015,790.757us =  0.224us/sample, max data rate 4.461Msps
    #    16,384 buffer_size: 46,828,950.226us =  0.223us/sample, max data rate 4.478Msps
    #    32,768 buffer_size: 47,981,193.013us =  0.229us/sample, max data rate 4.371Msps
    #    65,536 buffer_size: 50,079,534.548us =  0.239us/sample, max data rate 4.188Msps
    #   131,072 buffer_size: 46,074,479.300us =  0.220us/sample, max data rate 4.552Msps
    #   262,144 buffer_size: 46,971,092.565us =  0.224us/sample, max data rate 4.465Msps
    #   524,288 buffer_size: 46,553,195.041us =  0.222us/sample, max data rate 4.505Msps
    # 1,048,576 buffer_size: 47,128,251.482us =  0.225us/sample, max data rate 4.450Msps
    # 2,097,144 buffer_size: 49,820,579.872us =  0.238us/sample, max data rate 4.209Msps
    DataSize = 2_097_144 // 2

    io.DEFAULT_BUFFER_SIZE = 128 * 1024  # should be optimal for current SSD

    wr_buffer_range = [1024 * 2 ** (n) for n in range(12)]
    wr_buffer_range[-1] = 2_097_144  # clip to the largest buffer size

    _repeats = 10
    _sigmf = phaser_SigMF()

    data = np.zeros((1, DataSize), dtype=np.float64)
    print(f"io.DEFAULT_BUFFER_SIZE: {io.DEFAULT_BUFFER_SIZE}")
    for wr_buff_size in wr_buffer_range:
        _sigmf.open(wr_buff_size)
        t0 = time.perf_counter_ns()
        for i_ in range(_repeats):
            _sigmf.write(data[0])
        delta_us = (time.perf_counter_ns() - t0) / 1e3
        max_data_rate_Msps = (_repeats * DataSize) / delta_us
        print(
            f"{wr_buff_size:9,d} buffer_size: {delta_us:13,.3f}us = {1/max_data_rate_Msps:6,.3f}us/sample, max data rate {max_data_rate_Msps:2.3f}Msps"
        )
        _sigmf.close()
        phaser_IO.delete(_sigmf.base_filename + ".sigmf-data")
        phaser_IO.delete(_sigmf.base_filename + ".sigmf-meta")


if __name__ == "__main__":
    # phaser_IO.test()

    # _sigmf = phaser_SigMF()
    # _sigmf.test()  # 7.3779s

    # _sigmf.profile_write()

    # _sigmf.buffer_test() # showed no real difference
    # _sigmf.threading_test()  # 10.0960s

    # profile_test()

    time_write_per_buffer_size()
