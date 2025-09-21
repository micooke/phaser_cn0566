# bladeRF
sudo add-apt-repository ppa:nuand/bladerf
sudo apt-get update
sudo apt-get install bladerf

sudo apt-get install libbladerf-dev

sudo apt update
sudo apt install cmake python3-pip libusb-1.0-0

cd ~
git clone --depth 1 https://github.com/Nuand/bladeRF.git
pushd bladeRF/host
mkdir build && cd build
cmake ..
make -j8
sudo make install
sudo ldconfig
popd

python3 -m venv phaser
source phaser/bin/activate
pip install -r requirements.txt
python ~/bladeRF/host/libraries/libbladeRF_bindings/python/setup.py install
deactivate

conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

pipx install uv
uv init phaser
cd phaser
uv add -r requirements.txt
uv run phaser_BladeRF.py
uvx black phaser_BladeRF.py
# uv export -o requirements.txt 

sudo cp 88-nuand.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

bladerf-tool probe
bladerf-tool info

bladeRF-cli --flash-firmware /usr/share/Nuand/bladeRF/bladeRF_fw.img

bladerf-tool flash_fpga /usr/share/Nuand/bladeRF/hostedxA9.rbf

#cd ~/Downloads
#wget https://www.nuand.com/fx3/bladeRF_fw_latest.img
#bladerf-tool flash_fw bladeRF_fw_latest.img

## for xA9
#wget https://www.nuand.com/fpga/hostedxA9-latest.rbf
#bladerf-tool flash_fpga hostedxA9-latest.rbf

# BladeRF Notes
If you forget to enable the Rx channel, requesting a read will result in a timeout

Note: both Tx and Rx channels are disabled by default

Tx Gain: -24 to 66dB
Rx Gain: -15 to 60dB
Gain modes: Manual, Fast AGC, Slow AGC, Hybrid AGC, Default (for the micro v2 this is Slow AGC)

Error: "Using legacy message size. Consider upgrading firmware >= v2.5.0 and fpga >= v0.16.0"
"""python
from bladerf import _bladerf
_bladerf.version() # 2.6.0-git-7d7d87f
d = _bladerf.BladeRF()
d.get_fpga_version() # 0.14.0
d.close()
"""
