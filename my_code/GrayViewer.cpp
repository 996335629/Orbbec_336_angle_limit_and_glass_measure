#include <iostream>
#include <opencv2/opencv.hpp>  // ← 新增

#include "libobsensor/hpp/Error.hpp"
#include "libobsensor/hpp/Pipeline.hpp"
#include "window.hpp"  // ← 保留！它间接 include 了 FrameSet、StreamProfile 等

const char *metaDataTypes[] = {"TIMESTAMP",
                               "SENSOR_TIMESTAMP",
                               "FRAME_NUMBER",
                               "AUTO_EXPOSURE",
                               "EXPOSURE",
                               "GAIN",
                               "AUTO_WHITE_BALANCE",
                               "WHITE_BALANCE",
                               "BRIGHTNESS",
                               "CONTRAST",
                               "SATURATION",
                               "SHARPNESS",
                               "BACKLIGHT_COMPENSATION",
                               "HUE",
                               "GAMMA",
                               "POWER_LINE_FREQUENCY",
                               "LOW_LIGHT_COMPENSATION",
                               "MANUAL_WHITE_BALANCE",
                               "ACTUAL_FRAME_RATE",
                               "FRAME_RATE",
                               "AE_ROI_LEFT",
                               "AE_ROI_TOP",
                               "AE_ROI_RIGHT",
                               "AE_ROI_BOTTOM",
                               "EXPOSURE_PRIORITY",
                               "HDR_SEQUENCE_NAME",
                               "HDR_SEQUENCE_SIZE",
                               "HDR_SEQUENCE_INDEX",
                               "LASER_POWER",
                               "LASER_POWER_LEVEL",
                               "LASER_STATUS",
                               "GPIO_INPUT_DATA"};

int main(int argc, char **argv) try {
  ob::Pipeline pipe;

  std::shared_ptr<ob::Config> config = std::make_shared<ob::Config>();
  config->enableVideoStream(OB_STREAM_COLOR, 1280, 720, 30, OB_FORMAT_Y16);

  pipe.start(config);
  auto currentProfile = pipe.getEnabledStreamProfileList()
                            ->getProfile(0)
                            ->as<ob::VideoStreamProfile>();

  // 不再创建 Window，改用 OpenCV
  // Window app("ColorViewer", currentProfile->width(),
  // currentProfile->height());

  while (true) {
    auto frameSet = pipe.waitForFrames(100);
    if (frameSet == nullptr) {
      continue;
    }

    auto colorFrame = frameSet->colorFrame();
    if (colorFrame == nullptr) {
      continue;
    }

    // print metadata every 30 frames
    auto index = colorFrame->index();
    if (index % 30 == 0) {
      std::cout << "*************************** Color Frame #" << index
                << " Metadata List ********************************"
                << std::endl;
      for (int metaDataType = 0; metaDataType < OB_FRAME_METADATA_TYPE_COUNT;
           metaDataType++) {
        if (colorFrame->hasMetadata((OBFrameMetadataType)metaDataType)) {
          std::cout << metaDataTypes[metaDataType] << ": "
                    << colorFrame->getMetadataValue(
                           (OBFrameMetadataType)metaDataType)
                    << std::endl;
        } else {
          std::cout << metaDataTypes[metaDataType] << ": "
                    << "unsupported" << std::endl;
        }
      }
      std::cout << "***********************************************************"
                   "*********************"
                << std::endl
                << std::endl;
    }

    // ====== 用 OpenCV 显示 Y16 灰度图 ======
    int w = colorFrame->width();
    int h = colorFrame->height();

    // Y16 = 16-bit 单通道，每像素 2 字节
    cv::Mat y16(h, w, CV_16UC1, (void *)colorFrame->data());

    // 16-bit → 8-bit：取高 8 位（除以 256）
    cv::Mat y8;
    y16.convertTo(y8, CV_8UC1, 1.0 / 256.0);

    cv::imshow("Grayscale Y16", y8);

    // 按 ESC 退出
    if (cv::waitKey(1) == 27) break;
    // ==========================================
  }

  pipe.stop();
  cv::destroyAllWindows();
  return 0;
} catch (ob::Error &e) {
  std::cerr << "function:" << e.getName() << "\nargs:" << e.getArgs()
            << "\nmessage:" << e.getMessage()
            << "\ntype:" << e.getExceptionType() << std::endl;
  exit(EXIT_FAILURE);
}
