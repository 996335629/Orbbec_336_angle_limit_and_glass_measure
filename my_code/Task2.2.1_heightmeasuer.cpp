/**
 * 玻璃面板高度检测程序（基于 Orbbec Gemini 336）
 * 
 * 功能：
 *   - 实时检测幕墙水平框架线
 *   - 使用视觉里程计累积相机高度\\这个后面理论上可以从点云获得高度
 *   - 计算框架线绝对高度，去重合并，闭环校正
 *   - 飞行时自动计算玻璃面板高度并归类
 * 
 */
#include "window.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <thread>
#include <opencv2/opencv.hpp>
#include "libobsensor/hpp/Error.hpp"
#include "libobsensor/hpp/Pipeline.hpp"

// ===================== 数据结构 =====================
// 记录一条框架线（横梁/胶缝）
struct FrameRecord {
    double abs_height_mm;   // 绝对高度（毫米），向上为正
    double weight;          // 检测次数（用于加权平均）
    double last_seen_mm;    // 最后一次检测时的绝对高度（未使用，可保留）
};

// 全局框架线记录列表
std::vector<FrameRecord> g_frameRecords;

// ===================== 辅助函数（来自原示例） =====================
cv::Mat frameToMat(std::shared_ptr<ob::VideoFrame> frame) {
    if (frame == nullptr) return cv::Mat();
    int w = frame->width();
    int h = frame->height();
    switch (frame->format()) {
        case OB_FORMAT_YUYV:
            return cv::Mat(h, w, CV_8UC2, frame->data());
        case OB_FORMAT_MJPG:
            return cv::imdecode(cv::Mat(1, frame->dataSize(), CV_8UC1, frame->data()), cv::IMREAD_COLOR);
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
            return raw;
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

// ===================== 核心算法 =====================

/**
 * 水平框架线检测（灰度投影法）
 * @param gray 灰度图 (CV_8UC1)
 * @return 检测到的行坐标列表（像素）
 */
std::vector<int> detectHorizontalLines(const cv::Mat& gray) {
    std::vector<int> lines;
    int h = gray.rows;
    // 每行平均灰度
    std::vector<double> rowMean(h, 0.0);
    for (int y = 0; y < h; ++y) {
        const uchar* row = gray.ptr<uchar>(y);
        double sum = 0.0;
        for (int x = 0; x < gray.cols; ++x) {
            sum += row[x];
        }
        rowMean[y] = sum / gray.cols;
    }

    // 去趋势：减去滑动窗口平滑值（窗口大小51）
    int winSize = 51;
    std::vector<double> smooth(h, 0.0);
    for (int y = 0; y < h; ++y) {
        int start = std::max(0, y - winSize/2);
        int end = std::min(h-1, y + winSize/2);
        double sum = 0.0;
        for (int i = start; i <= end; ++i) sum += rowMean[i];
        smooth[y] = sum / (end - start + 1);
    }
    for (int y = 0; y < h; ++y) rowMean[y] -= smooth[y];

    // 找局部极小值（暗谷）
    int searchRad = 5;
    double threshold = 10.0;   // 谷深阈值（可调）
    for (int y = searchRad; y < h - searchRad; ++y) {
        double val = rowMean[y];
        bool isMin = true;
        for (int d = -searchRad; d <= searchRad; ++d) {
            if (d == 0) continue;
            if (rowMean[y+d] <= val) {
                isMin = false;
                break;
            }
        }
        if (isMin) {
            double leftMax = *std::max_element(rowMean.begin() + y - searchRad, rowMean.begin() + y);
            double rightMax = *std::max_element(rowMean.begin() + y + 1, rowMean.begin() + y + searchRad + 1);
            double depth = std::max(leftMax, rightMax) - val;
            if (depth > threshold) {
                lines.push_back(y);
            }
        }
    }

    // 去重：相邻10像素内只保留一个
    std::vector<int> filtered;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i > 0 && lines[i] - lines[i-1] < 10) continue;
        filtered.push_back(lines[i]);
    }
    return filtered;
}

/**
 * 从深度图更新相机累积高度（视觉里程计）
 * @param depth       深度图 (CV_16UC1)
 * @param fy          相机焦距y（像素）
 * @param cy          相机光心y（像素）
 * @param prevMedianY 上一帧的中值Y坐标（输入输出）
 * @param currHeight  当前相机累积高度（mm，向上为正，输出）
 * @return 是否成功更新
 */
bool updateCameraHeight(const cv::Mat& depth, double fy, double cy,
                        double& prevMedianY, double& currHeight) {
    int h = depth.rows, w = depth.cols;
    int roiX = w * 0.1, roiY = h * 0.1;
    int roiW = w * 0.8, roiH = h * 0.8;
    cv::Rect roi(roiX, roiY, roiW, roiH);
    cv::Mat roiDepth = depth(roi);

    std::vector<double> yCoords;
    const uint16_t* data = roiDepth.ptr<uint16_t>();
    for (int r = 0; r < roiH; ++r) {
        for (int c = 0; c < roiW; ++c) {
            uint16_t d_mm = data[r * roiW + c];
            if (d_mm > 0 && d_mm < 3000) { // 有效深度 0.1~3m
                int globalY = roiY + r;
                double y_coord = (globalY - cy) * d_mm / fy;
                yCoords.push_back(y_coord);
            }
        }
    }
    if (yCoords.empty()) return false;

    size_t n = yCoords.size();
    std::nth_element(yCoords.begin(), yCoords.begin() + n/2, yCoords.end());
    double medianY = yCoords[n/2];

    if (prevMedianY != 0.0) {
        double delta = medianY - prevMedianY;
        if (std::abs(delta) < 150.0) {   // 单帧位移限制150mm
            currHeight += delta;          // 相机向上移动时 medianY 减小，delta为负，高度增加
        }
    }
    prevMedianY = medianY;
    return true;
}

/**
 * 计算某一行（框架线）的真实高度偏移 real_y
 * @param y_pixel  行坐标（像素）
 * @param depth    深度图 (CV_16UC1)
 * @param fy       相机焦距y（像素）
 * @param cy       相机光心y（像素）
 * @return real_y (mm)，向下为正，若无效返回0
 */
double computeRealY(int y_pixel, const cv::Mat& depth, double fy, double cy) {
    int w = depth.cols;
    int startCol = w * 0.1;
    int endCol = w * 0.9;
    std::vector<uint16_t> validDepths;
    const uint16_t* rowData = depth.ptr<uint16_t>(y_pixel);
    for (int x = startCol; x < endCol; ++x) {
        uint16_t d = rowData[x];
        if (d > 0 && d < 3000) validDepths.push_back(d);
    }
    if (validDepths.empty()) return 0.0;
    size_t n = validDepths.size();
    std::nth_element(validDepths.begin(), validDepths.begin() + n/2, validDepths.end());
    double d_mm = validDepths[n/2];
    double real_y = (y_pixel - cy) * d_mm / fy;
    return real_y;
}

/**
 * 添加或更新框架线记录，并返回相机高度校正量
 * @param h_mm           检测到的绝对高度（mm，向上为正）
 * @param currHeight_mm  当前相机累积高度（mm）
 * @param real_y_mm      当前检测的 real_y（mm，向下为正）
 * @param outCorrection  输出的相机高度校正量（mm）
 * @return 是否触发了闭环校正（即重复检测到已有框架线）
 */
bool addOrUpdateFrame(double h_mm, double currHeight_mm, double real_y_mm, double& outCorrection) {
    const double mergeThresh = 40.0;   // 4cm内认为同一条线
    int bestIdx = -1;
    double bestDist = mergeThresh;
    for (size_t i = 0; i < g_frameRecords.size(); ++i) {
        double dist = std::abs(g_frameRecords[i].abs_height_mm - h_mm);
        if (dist < bestDist) {
            bestDist = dist;
            bestIdx = i;
        }
    }

    outCorrection = 0.0;
    if (bestIdx >= 0) {
        // 已存在：加权平均更新
        FrameRecord& rec = g_frameRecords[bestIdx];
        double newHeight = (rec.abs_height_mm * rec.weight + h_mm) / (rec.weight + 1.0);
        rec.abs_height_mm = newHeight;
        rec.weight += 1.0;

        // 闭环校正：利用已知绝对高度修正当前相机高度
        double correctCamHeight = rec.abs_height_mm + real_y_mm;
        double delta = correctCamHeight - currHeight_mm;
        double correction = delta * 0.3;          // 学习率0.3
        if (std::abs(correction) > 10.0)          // 单次最大校正10mm
            correction = (correction > 0 ? 10.0 : -10.0);
        outCorrection = correction;
        return true;
    } else {
        // 新框架线
        FrameRecord rec;
        rec.abs_height_mm = h_mm;
        rec.weight = 1.0;
        g_frameRecords.push_back(rec);
        return false;
    }
}

// ===================== 主函数 =====================
int main(int argc, char **argv) try {
    // 解析模式
    int mode = 1;  // 默认彩色模式
    if (argc > 1) {
        mode = std::atoi(argv[1]);
        if (mode != 0 && mode != 1) {
            std::cerr << "Invalid mode! Using default (1).\n";
            mode = 1;
        }
    }
    std::cout << "Selected mode: " << (mode == 0 ? "Grayscale" : "Color") << std::endl;

    ob::Pipeline pipe;
    std::shared_ptr<ob::Config> config = std::make_shared<ob::Config>();

    // 配置彩色流
    auto colorProfiles = pipe.getStreamProfileList(OB_SENSOR_COLOR);
    auto colorProfile =
        colorProfiles->getVideoStreamProfile(640, 480, OB_FORMAT_ANY, 10);
    config->enableStream(colorProfile);

    // 配置深度流
    auto depthProfiles = pipe.getStreamProfileList(OB_SENSOR_DEPTH);
    auto depthProfile =
        depthProfiles->getVideoStreamProfile(640, 480, OB_FORMAT_ANY, 10);
    config->enableStream(depthProfile);

    // 对齐：深度对齐到彩色
    OBStreamType alignTo = OB_STREAM_COLOR;
    ob::Align align(alignTo);

    pipe.start(config);

    // 获取相机内参（必须在 start 之后）
    auto camParam = pipe.getCameraParam();
    double fx = camParam.rgbIntrinsic.fx;
    double cx = camParam.rgbIntrinsic.cx;
    double fy = camParam.rgbIntrinsic.fy;
    double cy = camParam.rgbIntrinsic.cy;
    std::cout << "Camera fy: " << fy << ", cy: " << cy << std::endl;

    // 状态变量
    double currCameraHeight_mm = 0.0;   // 相机累积高度（向上为正）
    double prevMedianY = 0.0;           // 上一帧的中值Y坐标
    bool firstFrame = true;

    while (true) {
        int key = cv::waitKey(1);
        if (key == 27) break;            // ESC退出
        if (key == 'c') {                // 按c清空记录（可选）
            g_frameRecords.clear();
            std::cout << "Frame records cleared." << std::endl;
        }

        auto frameSet = pipe.waitForFrames(3000);
        if (frameSet == nullptr) continue;

        auto colorFrame = frameSet->colorFrame();
        auto depthFrame = frameSet->depthFrame();
        if (colorFrame == nullptr || depthFrame == nullptr) continue;

        // 对齐
        auto alignedFrame = align.process(frameSet);
        auto alignedFrameSet = alignedFrame->as<ob::FrameSet>();
        auto alignedColor = alignedFrameSet->colorFrame();
        auto alignedDepth = alignedFrameSet->depthFrame();

        // 转 OpenCV
        cv::Mat colorBGR = frameToBGR(alignedColor);
        cv::Mat depth16 = frameToMat(alignedDepth);  // CV_16UC1
        if (colorBGR.empty() || depth16.empty()) continue;

        // 模糊检测（拉普拉斯方差阈值100）//不要太高，调高了直接不显示
        cv::Mat gray;
        cv::cvtColor(colorBGR, gray, cv::COLOR_BGR2GRAY);
        cv::Mat lap;
        cv::Laplacian(gray, lap, CV_64F);
        cv::Scalar mean, stddev;
        cv::meanStdDev(lap, mean, stddev);
        double blurVal = stddev[0] * stddev[0];
        if (blurVal < 100.0) {
          continue;  // 模糊帧跳过
        }

        // 1. 更新相机高度（视觉里程计）
        if (!firstFrame) {
            updateCameraHeight(depth16, fy, cy, prevMedianY, currCameraHeight_mm);
        } else {
            // 第一帧只初始化 prevMedianY
            double dummy = 0.0;
            updateCameraHeight(depth16, fy, cy, prevMedianY, dummy);
            firstFrame = false;
        }

        // 2. 检测水平框架线
        std::vector<int> lines = detectHorizontalLines(gray);
        double totalCorrection = 0.0;
        for (int y_pixel : lines) {
            double real_y = computeRealY(y_pixel, depth16, fy, cy);
            if (real_y == 0.0) continue;   // 深度无效
            double frameAbsHeight = currCameraHeight_mm - real_y;   // 绝对高度（向上为正）
            double correction = 0.0;
            bool isRepeated = addOrUpdateFrame(frameAbsHeight, currCameraHeight_mm, real_y, correction);
            if (isRepeated) {
                totalCorrection += correction;   // 累积校正量，稍后统一应用
            }
        }
        // 应用累积的相机高度校正（所有重复检测的框架线共同校正）
        if (totalCorrection != 0.0) {
            currCameraHeight_mm += totalCorrection;
        }

        // 可视化：在彩色图上绘制检测到的线条
        cv::Mat display = colorBGR.clone();
        for (int y : lines) {
            cv::line(display, cv::Point(0, y), cv::Point(display.cols, y), cv::Scalar(0, 0, 255), 2);
        }
        // 可选：显示当前相机高度
        std::string heightText = "Cam height: " + std::to_string(currCameraHeight_mm / 1000.0) + " m";
        cv::putText(display, heightText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
        cv::imshow("Glass Height Detection", display);
    }

    // 飞行结束，计算玻璃面板高度并归类
    std::cout << "\n===== Processing completed. Total frames recorded: " << g_frameRecords.size() << " =====" << std::endl;
    if (g_frameRecords.size() >= 2) {
        // 按绝对高度排序
        std::sort(g_frameRecords.begin(), g_frameRecords.end(),
                  [](const FrameRecord& a, const FrameRecord& b) {
                      return a.abs_height_mm < b.abs_height_mm;
                  });

        // 计算相邻框架线的高度差（即玻璃高度）
        std::vector<double> glassHeights;
        for (size_t i = 1; i < g_frameRecords.size(); ++i) {
            double diff = g_frameRecords[i].abs_height_mm - g_frameRecords[i-1].abs_height_mm;
            if (diff > 100.0 && diff < 10000.0) {  // 合理范围 0.1~10m
                glassHeights.push_back(diff);
            }
        }

        if (glassHeights.empty()) {
            std::cout << "No valid glass heights found." << std::endl;
        } else {
            // 一维聚类（差值<60mm归为一类）
            std::sort(glassHeights.begin(), glassHeights.end());
            std::vector<std::pair<double, int>> classes; // (平均高度mm, 计数)
            for (double gh : glassHeights) {
                if (classes.empty() || gh - classes.back().first > 60.0) {
                    classes.push_back({gh, 1});
                } else {
                    double newMean = (classes.back().first * classes.back().second + gh) / (classes.back().second + 1);
                    classes.back().first = newMean;
                    classes.back().second++;
                }
            }

            std::cout << "\n===== Glass Panel Heights (mm) =====" << std::endl;
            for (const auto& cls : classes) {
                std::string label;
                if (cls.first > 3000.0) label = "lobby";
                else if (cls.first < 800.0) label = "mechanical";
                else if (cls.second >= 3) label = "standard";
                else label = "non-standard";
                std::cout << "Height: " << cls.first / 1000.0 << " m, count: " << cls.second
                          << " -> " << label << std::endl;
            }
        }
    } else {
        std::cout << "Insufficient frame lines detected. Need at least 2 lines to compute glass heights." << std::endl;
    }

    pipe.stop();
    cv::destroyAllWindows();
    return 0;
} catch (ob::Error &e) {
    std::cerr << "Orbbec error: " << e.getName() << " - " << e.getMessage() << std::endl;
    return EXIT_FAILURE;
}