<h1 align="center">DM-VIO: Delayed Marginalization<br/> Visual-Inertial Odometry </h1>

<p align="center">
<a href="https://vision.in.tum.de/_media/research/vslam/dm-vio/dm-vio.pdf">Paper</a> | <a href="https://youtu.be/7iep3BvcJPU">Video</a> | <a href="http://vision.in.tum.de/dm-vio">Project Page</a>
</p>

When using this project in academic work, please consider citing:

    @article{stumberg22dmvio,
      author = {L. von Stumberg and D. Cremers},
      title = {{DM-VIO}: Delayed Marginalization Visual-Inertial Odometry},
      journal = {{IEEE} Robotics and Automation Letters ({RA-L})},
      year = {2022},
      volume = {7},
      number = {2},
      pages = {1408-1415},
      doi = {10.1109/LRA.2021.3140129}
    }

### 1. Related Papers
* **[DM-VIO: Delayed Marginalization Visual-Inertial Odometry](https://vision.in.tum.de/dm-vio)**, L. von Stumberg and D. Cremers, In IEEE Robotics and Automation Letters (RA-L), volume 7, 2022
* **[Direct Sparse Visual-Inertial Odometry using Dynamic Marginalization](https://vision.in.tum.de/vi-dso)**, L. von Stumberg, V. Usenko and D. Cremers, In International Conference on Robotics and Automation (ICRA), 2018
* **[Direct Sparse Odometry](https://vision.in.tum.de/dso)**, *J. Engel, V. Koltun, D. Cremers*, In  TPAMI, vol. 40, 2018

### 2. Installation

	git clone https://github.com/lukasvst/dm-vio.git

The following instructions have been tested with Ubuntu 20.04.
The system is also known to work well on Ubuntu 16.04, 18.04 and MacOS Big Sur (only Intel Macs have been tested so far).

#### 2.1 Required Dependencies 

##### Suitesparse, Eigen3, Boost, yaml-cpp (required).
Required, install with

    sudo apt-get install cmake libsuitesparse-dev libeigen3-dev libboost-all-dev libyaml-cpp-dev

On MacOS we recommend Homebrew to install the dependencies. It might be necessary
to install boost@1.60 instead of the newest boost, in order for the used GTSAM version to work.


##### GTSAM (required).
Build from source with

    sudo apt install libtbb-dev
    git clone https://github.com/borglab/gtsam.git
    cd gtsam
    git checkout a738529af9754c7a085903f90ae8559bbaa82e75
    mkdir build && cd build
    cmake -DGTSAM_POSE3_EXPMAP=ON -DGTSAM_ROT3_EXPMAP=ON -DGTSAM_USE_SYSTEM_EIGEN=ON ..
    make -j
    sudo make install

##### OpenCV.
Used to read, write and display images.
Install with

	sudo apt-get install libopencv-dev


##### Pangolin.
Like for DSO, this is used for the GUI. You should install v0.6.
Install from [https://github.com/stevenlovegrove/Pangolin](https://github.com/stevenlovegrove/Pangolin)


	sudo apt install libgl1-mesa-dev libglew-dev pkg-config libegl1-mesa-dev libwayland-dev libxkbcommon-dev wayland-protocols
	git clone https://github.com/stevenlovegrove/Pangolin.git
	cd Pangolin
	git checkout v0.6
	mkdir build
	cd build
	cmake ..
	cmake --build .
	sudo make install
	
	 

#### 2.2 Recommended Dependencies

##### GTest (optional).
For running tests, install with `git submodule update --init`.

##### ziplib (optional).
Used to read datasets with images as .zip.
See [src/dso/README.md](src/dso/README.md) for instructions.

##### sse2neon (required for ARM builds).
After cloning, run `git submodule update --init` to include this. 

#### 2.3 Build

    cd dm-vio
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
    make -j

This compiles `dmvio_dataset` to run DM-VIO on datasets (needs both OpenCV and Pangolin installed).
It also compiles the library `libdmvio.a`, which other projects can link to.

#### Trouble-Shooting
The project is based on DSO and only has two additional dependencies with GTSAM and yaml-cpp.
In case of problems with compilation we recommend trying to compile https://github.com/JakobEngel/dso 
first and seeing if it works. 

### 3 Running
Download a TUM-VI sequence (download in the format `Euroc / DSO 512x512`) at https://vision.in.tum.de/data/datasets/visual-inertial-dataset

    bin/dmvio_dataset
        files=XXXX/datasetXXXX/dso/cam0/images              
        vignette=XXXX/datasetXXXX/dso/cam0/vignette.png
        imuFile=XXXX/datasetXXXX/dso/imu.txt
        gtFile=XXXX/datasetXXXX/dso/gt_imu.csv
        calib=PATH_TO_DMVIO/configs/tumvi_calib/camera02.txt
        gamma=PATH_TO_DMVIO/configs/tumvi_calib/pcalib.txt
        imuCalib=PATH_TO_DMVIO/configs/tumvi_calib/camchain.yaml
        mode=0
        use16Bit=1
        preset=0                                                        # use 1 for realtime
        nogui=0                                                         # use 1 to enable GUI
        resultsPrefix=/PATH_TO_RESULTS/
        settingsFile=PATH_TO_DMVIO/configs/tumvi.yaml
        start=2                                                         

Instead of typing this long command you can use the [python tools](https://github.com/lukasvst/dm-vio-python-tools).

#### Running on EuRoC, 4Seasons, reproducing paper results, etc.
We **strongly recommend** using the python-dm-vio tools published at: https://github.com/lukasvst/dm-vio-python-tools

They can be used to
* prepare the EuRoC and 4Seasons sequences for usage with DM-VIO.
* run on all (or some) sequences of EuRoC, TUM-VI and 4Seasons and gather the results.
* create a Python evaluation script for inspecting the results and generating the plots shown in the paper.

#### Commandline arguments
There are two types of commandline arguments:
1. Main arguments defined `in main_dmvio_dataset.cpp` (see `parseArgument`). Most of these are derived from 
DSO, so you can read [src/dso/README.md](src/dso/README.md) for documentation on them. 
2. Lots of additional settings are defined using the `SettingsUtil`. They can be set either using comandline
or by placing them in the yaml file defined with the commandline argument `settingsFile`.
All of them are printed to commandline when the program starts (and also into the file `usedSettingsdso.txt`).
Most of these are documented in the header file they are defined in 
(see `src/IMU/IMUSettings.h`, `src/IMUInitialization/IMUInitSettings.h`).

#### Running on your own datasets
To run on your own dataset you need
* to pass the folder containing files with `files=...`
* an accurate camera calibration! For tips on calibration and the format of camera.txt see 
[src/dso/README.md](src/dso/README.md).
* to set the `mode=1` unless you have a photometric calibration (vignette.png and pcalib.txt).
* a file times.txt which contains **exactly** one timestamp for each image in the image folder.

When enabling IMU data you also need

* IMU calibration (transformation between camera and IMU) as a `camchain.yaml`. Note that only the field `cam0/T_cam_imu`
and optionally the noise values are read from this file.
* a file containing IMU data. For each image it **must** contain an IMU 'measurement' with exactly the same timestamp. 
If the sensor does not output this, a fake measurement with this timestamp has to be interpolated in advance.
    The [DM-VIO python tools](https://github.com/lukasvst/dm-vio-python-tools) contain a script to do this.
* You should also set the IMU noise values (see `configs/tumvi.yaml`, `configs/euroc.yaml`, and `configs/4seasons.yaml`).
You can read them from an Allan-Variance plot (either computed yourself or taken from datasheet of IMU). 
Note that often times these values are too small in practice and should be inflated by a large factor for optimal results.

You can first set `useimu=0` to try the visual-only system (basically DSO). If this does not work well for 
comparably slow motions, there is likely a problem with camera calibration which should be addressed first.


### 4 License
DM-VIO is based on Direct Sparse Odometry (DSO), which was developed by Jakob Engel 
at the Technical University of Munich and Intel.
Like DSO, DM-VIO is licensed under the GNU General Public License
Version 3 (GPLv3).
