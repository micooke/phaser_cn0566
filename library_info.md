#### Using the outdated PothosSDR install ####
```bash
>>SoapySDRUtil --info
######################################################
##     Soapy SDR -- the SDR abstraction library     ##
######################################################

Lib Version: v0.8.1-PothosSDR-2021.07.25-vc16-x64
API Version: v0.8.0
ABI Version: v0.8
Install root: C:\Program Files\PothosSDR
Search path:  C:\Program Files\PothosSDR/lib/SoapySDR/modules0.8
Module found: ./bladeRFSupport.dll  (0.4.1-70505a5)
Module found: ./HackRFSupport.dll   (0.3.3-8d2e7be)
Module found: ./PlutoSDRSupport.dll (0.2.1-c880222)
Module found: ./airspyhfSupport.dll (0.2.0-d682533)
Module found: ./airspySupport.dll   (0.2.0-411f73e)
Module found: ./audioSupport.dll    (0.1.1-91080cb)
Module found: ./IrisSupport.dll     (2020.02.0.1-f100723)
Module found: ./LMS7Support.dll     (20.10.0-a45e482d)
Module found: ./miriSupport.dll     (0.2.6-585c012)
Module found: ./netSDRSupport.dll   (0.1.0-51516db)
Module found: ./osmosdrSupport.dll  (0.2.6-585c012)
Module found: ./RedPitaya.dll       (0.1.1-3d576f8)
Module found: ./remoteSupport.dll   (0.6.0-c09b2f1)
Module found: ./rtlsdrSupport.dll   (0.3.2-53ee8f4)
Module found: ./sdrPlaySupport.dll  (0.3.0-206b241)
Module found: ./uhdSupport.dll      (0.4.1-9a738c3)
Module found: ./volkConverters.dll  (0.1.0-62ac7f5)
Available factories... airspy, airspyhf, audio, bladerf, hackrf, iris, lime, miri, netsdr, osmosdr, plutosdr, redpitaya, remote, rtlsdr, sdrplay, uhd
Available converters...
 -  CF32 -> [CF32, CF64, CS16, CS32, CS8, CU16, CU8]
 -  CF64 -> [CF32, CS16, CS32, CS8]
 -  CS16 -> [CF32, CF64, CS16, CS8, CU16, CU8]
 -  CS32 -> [CF32, CF64, CS32]
 -   CS8 -> [CF32, CF64, CS16, CS8, CU16, CU8]
 -  CU16 -> [CF32, CS16, CS8]
 -   CU8 -> [CF32, CS16, CS8]
 -   F32 -> [F32, F64, S16, S32, S8, U16, U8]
 -   F64 -> [F32, S16, S32, S8]
 -   S16 -> [F32, F64, S16, S8, U16, U8]
 -   S32 -> [F32, F64, S32]
 -    S8 -> [F32, F64, S16, S8, U16, U8]
 -   U16 -> [F32, S16, S8]
 -    U8 -> [F32, S16, S8]

>>iio_info -s
Library version: 0.19 (git tag: 5f5af2e)
Compiled with backends: xml ip usb
No contexts found.

#>>iio_info --version # not available in this version
```

#### Using the soapysdr install from anaconda ####
```bash
>>SoapySDRUtil --info
######################################################
##     Soapy SDR -- the SDR abstraction library     ##
######################################################

Lib Version: v0.8.1-5
API Version: v0.8.0
ABI Version: v0.8
Install root: C:\Users\micooke\.conda\envs\soapysdr\Library
Search path:  C:\Users\micooke\.conda\envs\soapysdr\Library/lib/SoapySDR/modules0.8
Module found: ./bladeRFSupport.dll  (0.4.1)
Module found: ./HackRFSupport.dll   (0.3.4)
Module found: ./PlutoSDRSupport.dll (0.2.1)
Available factories... bladerf, hackrf, plutosdr
Available converters...
 -  CF32 -> [CF32, CS16, CS8, CU16, CU8]
 -  CS16 -> [CF32, CS16, CS8, CU16, CU8]
 -  CS32 -> [CS32]
 -   CS8 -> [CF32, CS16, CS8, CU16, CU8]
 -  CU16 -> [CF32, CS16, CS8]
 -   CU8 -> [CF32, CS16, CS8]
 -   F32 -> [F32, S16, S8, U16, U8]
 -   S16 -> [F32, S16, S8, U16, U8]
 -   S32 -> [S32]
 -    S8 -> [F32, S16, S8, U16, U8]
 -   U16 -> [F32, S16, S8]
 -    U8 -> [F32, S16, S8]

>>iio_info -s
Unable to create Local IIO context : Function not implemented (40)
No IIO context found.

>>iio_info --version
iio_info version: 0.26 (git tag:v0.26)
Libiio version: 0.26 (git tag: v0.26) backends: xml ip usb
```