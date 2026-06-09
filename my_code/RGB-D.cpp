/**
 * 灰度/RGB-深度对齐示例 - OpenCV 版本
 * 支持命令行参数：0=灰度模式，1=彩色模式
 * 默认彩色模式
 */

#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>

#include "libobsensor/hpp/Error.hpp"
#include "libobsensor/hpp/Pipeline.hpp"
#include "window.hpp"  // 可选，保留原有包含

// 将 OBFrame 转换为 cv::Mat（原始格式）
cv::Mat frameToMat(std::shared_ptr<ob::VideoFrame> frame) {
  if (frame == nullptr) return cv::Mat();

  int w = frame->width();
  int h = frame->height();

  switch (frame->format()) {
    case OB_FORMAT_YUYV:
      return cv::Mat(h, w, CV_8UC2, frame->data());  // YUYV 2字节/像素
    case OB_FORMAT_MJPG:
      return cv::imdecode(cv::Mat(1, frame->dataSize(), CV_8UC1, frame->data()),
                          cv::IMREAD_COLOR);
    case OB_FORMAT_RGB:
      return cv::Mat(h, w, CV_8UC3, frame->data());
    case OB_FORMAT_BGR:
      return cv::Mat(h, w, CV_8UC3, frame->data());
    case OB_FORMAT_Y8:
    case OB_FORMAT_BA81:
      return cv::Mat(h, w, CV_8UC1, frame->data());
    case OB_FORMAT_Y16:
    case OB_FORMAT_Z16:
      return cv::Mat(h, w, CV_16UC1, frame->data());
    case OB_FORMAT_NV12:
      return cv::Mat(h * 3 / 2, w, CV_8UC1, frame->data());
    case OB_FORMAT_NV21:
      return cv::Mat(h * 3 / 2, w, CV_8UC1, frame->data());
    default:
      std::cerr << "Unsupported format: " << frame->format() << std::endl;
      return cv::Mat();
  }
}

// 将彩色 OBFrame 转换为可显示的 BGR 图像
cv::Mat frameToBGR(std::shared_ptr<ob::VideoFrame> frame) {
  cv::Mat raw = frameToMat(frame);
  if (raw.empty()) return raw;

  switch (frame->format()) {
    case OB_FORMAT_YUYV: {
      cv::Mat bgr;
      cv::cvtColor(raw, bgr, cv::COLOR_YUV2BGR_YUYV);
      return bgr;
    }
    case OB_FORMAT_NV12: {
      cv::Mat bgr;
      cv::cvtColor(raw, bgr, cv::COLOR_YUV2BGR_NV12);
      return bgr;
    }
    case OB_FORMAT_NV21: {
      cv::Mat bgr;
      cv::cvtColor(raw, bgr, cv::COLOR_YUV2BGR_NV21);
      return bgr;
    }
    case OB_FORMAT_RGB: {
      cv::Mat bgr;
      cv::cvtColor(raw, bgr, cv::COLOR_RGB2BGR);
      return bgr;
    }
    case OB_FORMAT_BGR:
      return raw.clone();
    case OB_FORMAT_MJPG:
      return raw;  // imdecode 已返回 BGR
    case OB_FORMAT_Y8:
    case OB_FORMAT_BA81: {
      cv::Mat bgr;
      cv::cvtColor(raw, bgr, cv::COLOR_GRAY2BGR);
      return bgr;
    }
    default:
      return raw;
  }
}

// 将深度帧转为彩色热力图（伪彩色）
cv::Mat depthToColorMap(std::shared_ptr<ob::VideoFrame> depthFrame) {
  cv::Mat raw = frameToMat(depthFrame);
  if (raw.empty()) return raw;

  if (raw.type() != CV_16UC1) {
    return cv::Mat();
  }

  cv::Mat depth8;
  cv::normalize(raw, depth8, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::Mat colorMap;
  cv::applyColorMap(depth8, colorMap, cv::COLORMAP_JET);
  return colorMap;
}

// Alpha 混合：彩色图（BGR 或灰度转 BGR） + 深度伪彩色图
cv::Mat blendOverlay(cv::Mat foreground, cv::Mat depthColorMap, float alpha) {
  if (foreground.empty() || depthColorMap.empty()) return cv::Mat();

  // 确保前景图是3通道 BGR
  cv::Mat fgBGR;
  if (foreground.channels() == 1) {
    cv::cvtColor(foreground, fgBGR, cv::COLOR_GRAY2BGR);
  } else {
    fgBGR = foreground.clone();
  }

  cv::Mat resized;
  if (fgBGR.size() != depthColorMap.size()) {
    cv::resize(depthColorMap, resized, fgBGR.size());
  } else {
    resized = depthColorMap;
  }

  cv::Mat blended;
  cv::addWeighted(fgBGR, alpha, resized, 1.0f - alpha, 0, blended);
  return blended;
}

static bool syncEnabled = false;
static float alpha = 0.5f;

void handleKeyEvents(int key, ob::Pipeline &pipe) {
  if (key == '+' || key == '=') {
    alpha = std::min(alpha + 0.1f, 1.0f);
    std::cout << "Alpha: " << alpha << std::endl;
  } else if (key == '-' || key == '_') {
    alpha = std::max(alpha - 0.1f, 0.0f);
    std::cout << "Alpha: " << alpha << std::endl;
  } else if (key == 'f' || key == 'F') {
    syncEnabled = !syncEnabled;
    try {
      if (syncEnabled) {
        pipe.enableFrameSync();
        std::cout << "Frame sync: ON" << std::endl;
      } else {
        pipe.disableFrameSync();
        std::cout << "Frame sync: OFF" << std::endl;
      }
    } catch (...) {
      std::cerr << "Sync not supported" << std::endl;
    }
  }
}

void printUsage(const char *progName) {
  std::cout << "Usage: " << progName << " [mode]\n"
            << "  mode 0 : Grayscale mode (color image converted to gray)\n"
            << "  mode 1 : Color mode (default)\n";
}

int main(int argc, char **argv) try {
  // 解析命令行参数
  int mode = 1;  // 默认彩色模式
  if (argc > 1) {
    mode = std::atoi(argv[1]);
    if (mode != 0 && mode != 1) {
      std::cerr << "Invalid mode! Using default (1).\n";
      mode = 1;
    }
  }
  printUsage(argv[0]);
  std::cout << "Selected mode: " << (mode == 0 ? "Grayscale" : "Color")
            << std::endl;

  ob::Pipeline pipe;

  std::shared_ptr<ob::Config> config = std::make_shared<ob::Config>();

  // 配置彩色流（使用默认配置）
  auto colorProfiles = pipe.getStreamProfileList(OB_SENSOR_COLOR);
  //auto colorProfile = colorProfiles->getProfile(OB_PROFILE_DEFAULT);
  auto colorProfile =
      colorProfiles->getVideoStreamProfile(640, 480, OB_FORMAT_ANY, 30);
  config->enableStream(colorProfile);

  // 配置深度流（使用默认配置）
  auto depthProfiles = pipe.getStreamProfileList(OB_SENSOR_DEPTH);
  auto depthProfile =
      depthProfiles->getVideoStreamProfile(640, 480, OB_FORMAT_ANY, 30);
  config->enableStream(depthProfile);

  // 对齐：深度对齐到彩色
  OBStreamType alignTo = OB_STREAM_COLOR;
  ob::Align align(alignTo);

  pipe.start(config);

  auto colorVideoProfile = colorProfile->as<ob::VideoStreamProfile>();
  std::cout << "Color stream resolution: " << colorVideoProfile->width() << "x"
            << colorVideoProfile->height() << std::endl;

  while (true) {
    int key = cv::waitKey(1);
    if (key == 27) break;  // ESC 退出
    handleKeyEvents(key, pipe);

    auto frameSet = pipe.waitForFrames(100);
    if (frameSet == nullptr) continue;

    auto colorFrame = frameSet->colorFrame();
    auto depthFrame = frameSet->depthFrame();
    if (colorFrame == nullptr || depthFrame == nullptr) continue;

    // 对齐
    auto alignedFrame = align.process(frameSet);
    auto alignedFrameSet = alignedFrame->as<ob::FrameSet>();
    auto alignedColor = alignedFrameSet->colorFrame();
    auto alignedDepth = alignedFrameSet->depthFrame();

    // 转为 OpenCV 格式
    cv::Mat colorBGR = frameToBGR(alignedColor);  // 得到 BGR 彩色图
    cv::Mat displayImage;
    if (mode == 0) {
      // 灰度模式：转换为灰度图
      cv::cvtColor(colorBGR, displayImage, cv::COLOR_BGR2GRAY);
    } else {
      // 彩色模式：直接使用彩色图
      displayImage = colorBGR;
    }
    cv::Mat depthColor = depthToColorMap(alignedDepth);

    if (displayImage.empty() || depthColor.empty()) continue;

    // 显示
    if (mode == 0) {
      cv::imshow("Grayscale (from Color)", displayImage);
    } else {
      cv::imshow("Color (RGB)", displayImage);
    }
    cv::imshow("Depth (Pseudo Color)", depthColor);

    cv::Mat blended = blendOverlay(displayImage, depthColor, alpha);
    if (!blended.empty()) {
      std::string overlayTitle =
          mode == 0 ? "Overlay (Gray + Depth) | +/- alpha, F sync"
                    : "Overlay (RGB + Depth) | +/- alpha, F sync";
      cv::imshow(overlayTitle, blended);
    }
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