# STGx control API 
This guide is for installing the ctrl-stg4000 package and making it work for STG5.

### Install ctrl-stg4 repo
```bash
git submodule add https://github.com/pyreiz/ctrl-stg4000.git
cd .\ctrl-stg4000\
pip install -r requirements.txt
pip install -e .
#download and install the dll from mulitchannelsystems
python -m stg.install
```
> Changes bellow are for STG5, if you are using STG4 you can skip them.
If the changes were made once, you can skip bellow steps and run third and fourth commands from the above code snippet.

### Chnages in order to make it work for STG5:
(Files after changes attached)
1. Use the updated McsUsbNet.dll from:
   multichannel systems (MCS) library to control MCS devices e.g. stimulus generator devices with examples. (
   URL: https://github.com/multichannelsystems/McsUsbNet_Examples.git).

* Make sure to use the right DLL_PATH in install.py

2. in stg/_wrapper/dll.py
    1. chnage def avialabl() to:
   ```
    def available() -> List[DeviceInfo]:
        "list all available MCS STGs connected over USB with this PC"
        deviceList = CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB)
        print("found %d devices" % (deviceList.Count))
    
        devices = []
        for dev_num in range(deviceList.Count):
            devices.append(deviceList.GetUsbListEntry(dev_num))
        return devices
   ````

    2. change in def _collect_properties(self):
       from:
       ```
       soft, hard = interface.GetStgVersionInfo("", "")
       ```
       to:
       ```
       _, soft, hard = interface.GetStgVersionInfo("", "")
       ```

3. in stg/_wrapper/downloadned.py, change in def download(), from:
   ```
         amplitudes = [System.Int32(a * 1000_000) for a in amplitudes_in_mA]
         durations = [System.UInt64(s * 1000) for s in durations_in_ms]
   ```
   to:
   ````
         amplitudes = [System.Int32(int(a * 1000_000)) for a in amplitudes_in_mA]
         durations = [System.UInt64(int(s * 1000)) for s in durations_in_ms]
   ````

* If not recognizing the stg module make sure to mark ctrl-stg4000 directory as Source Root. 
(Right-click->Mark directory as->Source Root)

RUN example_of_use to make sure everything is running!

Good Luck!