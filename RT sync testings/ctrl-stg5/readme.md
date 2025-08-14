[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://en.wikipedia.org/wiki/MIT_License) [![pytest-status](https://github.com/pyreiz/ctrl-stg4000/workflows/pytest/badge.svg)](https://github.com/pyreiz/ctrl-stg4000/actions) [![Coverage Status](https://coveralls.io/repos/github/pyreiz/ctrl-stg4000/badge.svg?branch=develop)](https://coveralls.io/github/pyreiz/ctrl-stg4000?branch=develop) [![Documentation Status](https://readthedocs.org/projects/ctrl-stg4000/badge/?version=latest)](https://ctrl-stg4000.readthedocs.io/en/latest/?badge=latest)

ctrl-stg4000
============

This documentation explains the python package ctrl-stg4000 wrapping the [C# .dll](https://www.multichannelsystems.com/software/mcsusbnetdll) offered by
multichannelsystems to control their STG4000 range of electrical stimulators. See also their gthub [repo](https://github.com/multichannelsystems/McsUsbNet).

At the time of development of this toolbox (and still), the company website offers the dll in version 3.2.45. In the meantime, the github repo was published and apparently already offers version 5.0.12. Please note that this toolbox is tested against version 3.2.45.

Installation
------------

#### Windows

ctrl-stg4000 wraps the [C# .dll](https://www.multichannelsystems.com/software/mcsusbnetdll) offered by
multichannelsystems to control their STG4000 range of electrical stimulators.  Therefore, the python package only works on Windows, because the STG and the dll are only supported for Windows by multichannelsystems.

``` bash
    git clone https://github.com/pyreiz/ctrl-stg4000
    cd ctrl-stg4000
    pip install -r requirements.txt
    pip install -e .
    #download and install the dll from mulitchannelsystems
    python -m stg.install
```

#### Linux

The package can be installed on Linux though, just skip the installation of pythonnet and downloading the dll.


``` bash

    git clone https://github.com/pyreiz/ctrl-stg4000
    cd ctrl-stg4000
    pip install -e .
```

On linux, the package automatically mocks the interface to the STG4000. This allows to run tests and build documentation, and can help when you write scripts for your experiments.

Testing
-------

Connect your oscilloscope and start the following example:

``` python

   from stg.api import PulseFile, STG4000
   stg = STG4000()
   stg.download(0, *PulseFile().compile())
   stg.start_stimulation([0])
```
You can run full tests using pytest, mypy or everything with :code: `make test` from the root of the package. By default, downloading the dll is not tested, but can be turned on with :code:`pytest -m "install"`.

## Notes: By @Rawan

This repository is a fork of the original [ctrl-stg4000](https://github.com/pyreiz/ctrl-stg4000) repository.\
The original repository was developed for the STG4000 stimulator. \
This fork is modified to work with the STG5 stimulator. 

The main changes made to the original repository are mentioned in the [README.md](api_stgx_install_guidlines/README.md) file in the api_stgx_install_guidlines directory.
Additional changes were made to get the right parameters for the STG5 stimulator (some default parameters are not right, see changes made since 2024/05).

API documentation can be find in [ctrl-stg4000 documentation](https://ctrl-stg4000.readthedocs.io/en/latest/index.html#)


1. [ ] TODO: Check this for parralel processing:\
Because at any time, only one process can be connected with a specific STG the connection is implemented using
a with ... as idiom. This should therefore be relatively safe. It is still possible that the STG can get into
a weird state. In that case, try turning it off and on again.

The Streaming mode works by use of two ring buffers which hold data. One is in PC memory and managed by the DLL,
and one is in on-board STG memory. Data is transferred from PC memory to the STG via the USB bus in time slices
of one millisecond.

### **Benchmark**
There are several possible ways how you can start stimulation on the STG. Depending on how you do that, different
jitters and latencies apply.

All the following examples use our in-house Arduino interface to generate TTL of 1ms duration. This trigger signal
is shown in the oscilloscope figures as blue trace. Above each figure, you can see the code which caused this
behaviour, and each code required the following snippet for initialiazation.

*Trigger via TTL*\
The fastest and most reliable by far is triggering via TTL when a stimulation has already been downloaded
with download(). In that regard, the latency from the TTL arriving at the STGs BNC to the actual stimulation
occurring is around 65µs.

*Trigger via USB*\
Compared to the fastest possible, triggering via start_stimulation() is still quite good. It comes with a
latency of 2 to 2.5 ms.

*Download on the fly*\
Ongoing download(NOT RECOMMENDED)
Downloading takes a long time, and latency and jitter are drastically increased to around 200 to 250ms.
Additionally, any ongoing stimulation at this channel will be interrupted while the download is going on.
This means adaptation of e.g. the amplitude of an ongoing repetitive stimulation is not possible, if that stimulation
is at a frequency faster than 4 Hz.

Instead use streaming mode.
