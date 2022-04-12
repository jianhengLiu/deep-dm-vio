/**
 * This file is based on the file main_dso_pangolin.cpp of the project DSO written by Jakob Engel.
 * It has been modified by Lukas von Stumberg for the inclusion in DM-VIO (http://vision.in.tum.de/dm-vio).
 *
 * Copyright 2022 Lukas von Stumberg <lukas dot stumberg at tum dot de>
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

// Main file for running on datasets, based on the main file of DSO.

#define BACKWARD_HAS_DW 1
#include "backward.hpp"
namespace backward{
    backward::SignalHandling sh;
}

#include <locale.h>
#include <opencv2/highgui.hpp>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h>

#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/Output3DWrapper.h"

#include "dso/util/DatasetReader.h"
#include "dso/util/globalCalib.h"
#include "dso/util/globalFuncs.h"
#include "dso/util/settings.h"
#include "util/TimeMeasurement.h"
#include <boost/thread.hpp>

#include "FullSystem/FullSystem.h"
#include "FullSystem/PixelSelector2.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "dso/util/NumType.h"

#include <util/SettingsUtil.h>

#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"

std::string gtFile = "";
std::string vignette = "";
std::string gammaCalib = "";
std::string source = "";
std::string calib = "";
std::string imuFile = "";
std::string imuCalibFile = "";
bool reverse = false;
int start = 0; // 开始帧
int end = 100000; // 结束帧
float playbackSpeed = 0; // 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
bool preload = false;
int maxPreloadImages = 0; // If set we only preload if there are less images to be loade.
bool useSampleOutput = false;

int mode = 0;

using namespace dso;

dmvio::IMUCalibration imuCalibration;
dmvio::IMUSettings imuSettings;

void my_exit_handler(int s)
{
    printf("Caught signal %d\n", s);
    exit(1);
}

void exitThread()
{
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = my_exit_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    while (true)
        pause();
}

void settingsDefault(int preset)
{
    printf("\n=============== PRESET Settings: ===============\n");
    if (preset == 0 || preset == 1) {
        printf("DEFAULT settings:\n"
               "- %s real-time enforcing\n"
               "- 2000 active points\n"
               "- 5-7 active frames\n"
               "- 1-6 LM iteration each KF\n"
               "- original image resolution\n",
            preset == 0 ? "no " : "1x");

        playbackSpeed = (preset == 0 ? 0 : 1.0);
        preload = preset == 1;
        setting_desiredImmatureDensity = 1500;
        // setting_desiredPointDensity = 2000;
        setting_desiredPointDensity = 1000;
        setting_minFrames = 5;
        setting_maxFrames = 7;
        setting_maxOptIterations = 6;
        setting_minOptIterations = 1;

        setting_logStuff = false;
    }

    if (preset == 2 || preset == 3) {
        printf("FAST settings:\n"
               "- %s real-time enforcing\n"
               "- 800 active points\n"
               "- 4-6 active frames\n"
               "- 1-4 LM iteration each KF\n"
               "- 424 x 320 image resolution\n",
            preset == 0 ? "no " : "5x");

        playbackSpeed = (preset == 2 ? 0 : 5);
        preload = preset == 3;
        setting_desiredImmatureDensity = 600;
        setting_desiredPointDensity = 800;
        setting_minFrames = 4;
        setting_maxFrames = 6;
        setting_maxOptIterations = 4;
        setting_minOptIterations = 1;

        benchmarkSetting_width = 424;
        benchmarkSetting_height = 320;

        setting_logStuff = false;
    }

    printf("==============================================\n");
}

void parseArgument(char* arg, dmvio::SettingsUtil& settingsUtil)
{
    int option;
    float foption;
    char buf[1000];

    // --------------------------------------------------
    // These are mostly the original DSO commandline arguments which also work for DM-VIO.
    // The DM-VIO specific settings can also be set with commandline arguments (and also with the yaml settings file)
    // and have been registered in the main function. See IMUSettings and its members for details.
    // --------------------------------------------------

    if (1 == sscanf(arg, "sampleoutput=%d", &option)) {
        if (option == 1) {
            useSampleOutput = true;
            printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "quiet=%d", &option)) {
        if (option == 1) {
            setting_debugout_runquiet = true;
            printf("QUIET MODE, I'll shut up!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "preset=%d", &option)) {
        settingsDefault(option);
        return;
    }

    if (1 == sscanf(arg, "nolog=%d", &option)) {
        if (option == 1) {
            setting_logStuff = false;
            printf("DISABLE LOGGING!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "reverse=%d", &option)) {
        if (option == 1) {
            reverse = true;
            printf("REVERSE!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "nogui=%d", &option)) {
        if (option == 1) {
            disableAllDisplay = true;
            printf("NO GUI!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "nomt=%d", &option)) {
        if (option == 1) {
            multiThreading = false;
            printf("NO MultiThreading!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "start=%d", &option)) {
        start = option;
        printf("START AT %d!\n", start);
        return;
    }
    if (1 == sscanf(arg, "end=%d", &option)) {
        end = option;
        printf("END AT %d!\n", end);
        return;
    }

    if (1 == sscanf(arg, "files=%s", buf)) {
        source = buf;
        printf("loading data from %s!\n", source.c_str());
        return;
    }

    if (1 == sscanf(arg, "calib=%s", buf)) {
        calib = buf;
        printf("loading calibration from %s!\n", calib.c_str());
        return;
    }

    if (1 == sscanf(arg, "vignette=%s", buf)) {
        vignette = buf;
        printf("loading vignette from %s!\n", vignette.c_str());
        return;
    }

    if (1 == sscanf(arg, "gtFile=%s", buf)) {
        gtFile = buf;
        printf("loading gtFile from %s!\n", gtFile.c_str());
        return;
    }

    if (1 == sscanf(arg, "imuCalib=%s", buf)) {
        imuCalibFile = buf;
        printf("Loading imu parameters from %s!\n", imuCalibFile.c_str());
        imuCalibration.loadFromFile(imuCalibFile);
        return;
    }

    if (1 == sscanf(arg, "imuFile=%s", buf)) {
        imuFile = buf;
        printf("IMU file is locatd at: %s!\n", imuFile.c_str());
        return;
    }

    if (1 == sscanf(arg, "useimu=%d", &option)) {
        if (option == 0) {
            printf("Disabling IMU integration!\n");
            setting_useIMU = false;
        } else if (option == 1) {
            printf("Enabling IMU integration!\n");
            setting_useIMU = true;
        }
        return;
    }

    if (1 == sscanf(arg, "gamma=%s", buf)) {
        gammaCalib = buf;
        printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
        return;
    }

    if (1 == sscanf(arg, "speed=%f", &foption)) {
        playbackSpeed = foption;
        printf("PLAYBACK SPEED %f!\n", playbackSpeed);
        return;
    }

    if (1 == sscanf(arg, "save=%d", &option)) {
        if (option == 1) {
            debugSaveImages = true;
            if (42 == system("rm -rf images_out"))
                printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            if (42 == system("mkdir images_out"))
                printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            if (42 == system("rm -rf images_out"))
                printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            if (42 == system("mkdir images_out"))
                printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            printf("SAVE IMAGES!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "mode=%d", &option)) {

        mode = option;
        if (option == 0) {
            printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
        }
        if (option == 1) {
            printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
            setting_photometricCalibration = 0;
            setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
        }
        if (option == 2) {
            printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
            setting_photometricCalibration = 0;
            setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_minGradHistAdd = 3;
        }
        return;
    }

    if (1 == sscanf(arg, "settingsFile=%s", buf)) {
        YAML::Node settings = YAML::LoadFile(buf);
        settingsUtil.tryReadFromYaml(settings);
        printf("Loading settings from yaml file: %s!\n", imuCalibFile.c_str());
        return;
    }

    if (settingsUtil.tryReadFromCommandLine(arg)) {
        return;
    }

    printf("could not parse argument \"%s\"!!!!\n", arg);
    assert(0);
}

void run(ImageFolderReader* reader, IOWrap::PangolinDSOViewer* viewer)
{

    if (setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0) {
        printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
        exit(1);
    }

    int lstart = start;
    int lend = end;
    int linc = 1; // 每隔linc个读一张图片
    if (reverse) {
        assert(!setting_useIMU); // Reverse is not supported with IMU data at the moment!
        printf("REVERSE!!!!");
        lstart = end - 1;
        if (lstart >= reader->getNumImages())
            lstart = reader->getNumImages() - 1;
        lend = start;
        linc = -1;
    }

    bool linearizeOperation = (playbackSpeed == 0);

    if (linearizeOperation && setting_minFramesBetweenKeyframes < 0) {
        setting_minFramesBetweenKeyframes = -setting_minFramesBetweenKeyframes;
        std::cout << "Using setting_minFramesBetweenKeyframes=" << setting_minFramesBetweenKeyframes
                  << " because of non-realtime mode." << std::endl;
    }

    FullSystem* fullSystem = new FullSystem(linearizeOperation, imuCalibration, imuSettings);
    fullSystem->setGammaFunction(reader->getPhotometricGamma());

    if (viewer != nullptr) {
        fullSystem->outputWrapper.push_back(viewer);
    }

    std::unique_ptr<IOWrap::SampleOutputWrapper> sampleOutPutWrapper;
    if (useSampleOutput) {
        sampleOutPutWrapper.reset(new IOWrap::SampleOutputWrapper());
        fullSystem->outputWrapper.push_back(sampleOutPutWrapper.get());
    }

    std::vector<int> idsToPlay; // 储存需要用到的图片ids
    std::vector<double> timesToPlayAt; // 储存每个图片什么时候要被播放(根据playbackSpeed调节播放速度)
    for (int i = lstart; i >= 0 && i < reader->getNumImages() && linc * i < linc * lend; i += linc) {
        idsToPlay.push_back(i);
        if (timesToPlayAt.empty()) {
            timesToPlayAt.push_back((double)0);
        } else {
            double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size() - 1]);
            double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size() - 2]);
            timesToPlayAt.push_back(timesToPlayAt.back() + fabs(tsThis - tsPrev) / playbackSpeed);
        }
    }

    if (preload && maxPreloadImages > 0) {
        if (reader->getNumImages() > maxPreloadImages) {
            printf("maxPreloadImages EXCEEDED! NOT PRELOADING!\n");
            preload = false;
        }
    }

    std::vector<ImageAndExposure*> preloadedImages;
    if (preload) {
        printf("LOADING ALL IMAGES!\n");
        for (int i : idsToPlay) {
            preloadedImages.push_back(reader->getImage(i));
        }
    }

    struct timeval tv_start;
    gettimeofday(&tv_start, nullptr); // 获取日历时间
    clock_t started = clock();
    double sInitializerOffset = 0;

    bool gtDataThere = reader->loadGTData(gtFile);

    bool imuDataSkipped = false;
    dmvio::IMUData skippedIMUData;
    for (int ii = 0; ii < (int)idsToPlay.size(); ii++) {
        if (!fullSystem->initialized) // if not initialized: reset start time.
        {
            gettimeofday(&tv_start, nullptr);
            started = clock();
            sInitializerOffset = timesToPlayAt[ii];
        }

        int i = idsToPlay[ii];

        ImageAndExposure* img;
        ImageAndExposure* featureImg;
        if (preload)
            img = preloadedImages[ii];
        else {
            img = reader->getImage(i);
            featureImg = reader->getFeatureImage(i);
        }

        bool skipFrame = false;
        if (playbackSpeed != 0) {
            struct timeval tv_now;
            gettimeofday(&tv_now, NULL);
            double sSinceStart = sInitializerOffset + ((tv_now.tv_sec - tv_start.tv_sec) + (tv_now.tv_usec - tv_start.tv_usec) / (1000.0f * 1000.0f));

            if (sSinceStart < timesToPlayAt[ii])
                usleep((int)((timesToPlayAt[ii] - sSinceStart) * 1000 * 1000));
            else if (sSinceStart > timesToPlayAt[ii] + 0.5 + 0.1 * (ii % 2)) {
                printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii, timesToPlayAt[ii], sSinceStart);
                skipFrame = true;
            }
        }

        dmvio::GTData data;
        bool found = false;
        if (gtDataThere) {
            data = reader->getGTData(i, found);
        }

        std::unique_ptr<dmvio::IMUData> imuData; // unique_ptr: 指向一个给定的对象，不支持拷贝和赋值
        if (setting_useIMU) {
            // returning IMU data between frame i-1 and frame i!
            imuData = std::make_unique<dmvio::IMUData>(reader->getIMUData(i));
        }
        if (!skipFrame) {
            // 添加未处理的imu数据（即被skipFrame时的IMU信息）
            if (imuDataSkipped && imuData) {
                imuData->insert(imuData->begin(), skippedIMUData.begin(), skippedIMUData.end());
                skippedIMUData.clear();
                imuDataSkipped = false;
            }
            // VIO入口：输入image,imu,gt
            // fullSystem->addActiveFrame(img, i, imuData.get(), (gtDataThere && found) ? &data : nullptr);
            fullSystem->addActiveFrame(img, featureImg, i, imuData.get(), (gtDataThere && found) ? &data : nullptr);
            if (gtDataThere && found && !disableAllDisplay) {
                viewer->addGTCamPose(data.pose);
            }
        } else if (imuData) {
            imuDataSkipped = true;
            skippedIMUData.insert(skippedIMUData.end(), imuData->begin(), imuData->end());
        }

        delete img;

        if (fullSystem->initFailed || setting_fullResetRequested) {
            // TODO: 这里与图片序列有关联,实时需要去掉
            if (ii < 250 || setting_fullResetRequested) {
                printf("RESETTING!\n");
                std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
                delete fullSystem;
                for (IOWrap::Output3DWrapper* ow : wraps)
                    ow->reset();

                fullSystem = new FullSystem(linearizeOperation, imuCalibration, imuSettings);
                fullSystem->setGammaFunction(reader->getPhotometricGamma());
                fullSystem->outputWrapper = wraps;

                setting_fullResetRequested = false;
            }
        }

        if (viewer != nullptr && viewer->shouldQuit()) {
            std::cout << "User closed window -> Quit!" << std::endl;
            break;
        }

        if (fullSystem->isLost) {
            printf("LOST!!\n");
            break;
        }
    }
    fullSystem->blockUntilMappingIsFinished();
    clock_t ended = clock();
    struct timeval tv_end;
    gettimeofday(&tv_end, NULL);

    fullSystem->printResult(imuSettings.resultsPrefix + "result.txt", false, false, true);
    fullSystem->printResult(imuSettings.resultsPrefix + "resultKFs.txt", true, false, false);
    fullSystem->printResult(imuSettings.resultsPrefix + "resultScaled.txt", false, true, true);

    dmvio::TimeMeasurement::saveResults(imuSettings.resultsPrefix + "timings.txt");

    int numFramesProcessed = abs(idsToPlay[0] - idsToPlay.back());
    double numSecondsProcessed = fabs(reader->getTimestamp(idsToPlay[0]) - reader->getTimestamp(idsToPlay.back()));
    double MilliSecondsTakenSingle = 1000.0f * (ended - started) / (float)(CLOCKS_PER_SEC);
    double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);
    printf("\n======================"
           "\n%d Frames (%.1f fps)"
           "\n%.2fms per frame (single core); "
           "\n%.2fms per frame (multi core); "
           "\n%.3fx (single core); "
           "\n%.3fx (multi core); "
           "\n======================\n\n",
        numFramesProcessed, numFramesProcessed / numSecondsProcessed,
        MilliSecondsTakenSingle / numFramesProcessed,
        MilliSecondsTakenMT / (float)numFramesProcessed,
        1000 / (MilliSecondsTakenSingle / numSecondsProcessed),
        1000 / (MilliSecondsTakenMT / numSecondsProcessed));
    fullSystem->printFrameLifetimes();
    if (setting_logStuff) {
        std::ofstream tmlog;
        tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
        tmlog << 1000.0f * (ended - started) / (float)(CLOCKS_PER_SEC * reader->getNumImages()) << " "
              << ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f) / (float)reader->getNumImages() << "\n";
        tmlog.flush();
        tmlog.close();
    }

    for (IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper) {
        ow->join();
    }

    printf("DELETE FULLSYSTEM!\n");
    delete fullSystem;

    printf("DELETE READER!\n");
    delete reader;

    printf("EXIT NOW!\n");
}

int main(int argc, char** argv)
{
    setlocale(LC_ALL, "C");

#ifdef DEBUG
    std::cout << "DEBUG MODE!" << std::endl;
#endif

    bool use16Bit = false;

    auto settingsUtil = std::make_shared<dmvio::SettingsUtil>();

    // Create Settings files.
    imuSettings.registerArgs(*settingsUtil);
    imuCalibration.registerArgs(*settingsUtil);
    settingsUtil->registerArg("setting_minOptIterations", setting_minOptIterations);
    settingsUtil->registerArg("setting_maxOptIterations", setting_maxOptIterations);
    settingsUtil->registerArg("setting_minIdepth", setting_minIdepth);
    settingsUtil->registerArg("setting_solverMode", setting_solverMode);
    settingsUtil->registerArg("use16Bit", use16Bit);
    settingsUtil->registerArg("setting_weightZeroPriorDSOInitY", setting_weightZeroPriorDSOInitY);
    settingsUtil->registerArg("setting_weightZeroPriorDSOInitX", setting_weightZeroPriorDSOInitX);
    settingsUtil->registerArg("setting_forceNoKFTranslationThresh", setting_forceNoKFTranslationThresh);
    settingsUtil->registerArg("setting_minFramesBetweenKeyframes", setting_minFramesBetweenKeyframes);
    settingsUtil->registerArg("preload", preload);
    settingsUtil->registerArg("maxPreloadImages", maxPreloadImages);

    for (int i = 1; i < argc; i++)
        parseArgument(argv[i], *settingsUtil);

    // Print settings to commandline and file.
    std::cout << "Settings:\n";
    settingsUtil->printAllSettings(std::cout);
    {
        std::ofstream settingsStream;
        settingsStream.open(imuSettings.resultsPrefix + "usedSettingsdso.txt");
        settingsUtil->printAllSettings(settingsStream);
    }

    // hook crtl+C.
    boost::thread exThread = boost::thread(exitThread);

    std::cout << "calib:" << calib << std::endl;
    ImageFolderReader* reader = new ImageFolderReader(source, calib, gammaCalib, vignette, use16Bit);
    reader->loadIMUData(imuFile);
    reader->setGlobalCalibration();

    if (!disableAllDisplay) {
        IOWrap::PangolinDSOViewer* viewer = new IOWrap::PangolinDSOViewer(wG[0], hG[0], false, settingsUtil);

        boost::thread runThread = boost::thread(boost::bind(run, reader, viewer));

        viewer->run();

        delete viewer;

        // Make sure that the destructor of FullSystem, etc. finishes, so all log files are properly flushed.
        runThread.join();
    } else {
        run(reader, 0);
    }
    std::cout << "Finished!" << std::endl;
    cv::waitKey();

    return 0;
}
