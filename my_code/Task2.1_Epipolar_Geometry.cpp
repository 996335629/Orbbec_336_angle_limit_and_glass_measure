// wall_angle_estimator.cpp
// 基于对极几何的墙面角度估计 —— 利用视觉里程计累积旋转求偏航角
//
// 编译 (OpenCV 4.x):
//   g++ -std=c++17 -O2 wall_angle_estimator.cpp -o wall_angle \
//       $(pkg-config --cflags --libs opencv4)
//
// 运行:
//   ./wall_angle [摄像头ID，默认0]
//   按 R 重置  |  按 ESC 退出
//
// 坐标系约定 (OpenCV):  X→右, Y→下, Z→前方(光轴)
// 初始状态: 相机光轴垂直于墙面, 定义夹角 = 90°
// 有效范围: [45°, 135°]  (±45°)

#include <algorithm>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>

// ==================================================================
//  配置参数 —— 根据实际工况和标定结果修改
// ==================================================================
struct Config {
  // ---------- 相机内参（必须标定！） ----------
  // 默认值为 640×480 VGA 分辨率下的典型值，仅供调试
  double fx = 411.0462341308594;
  double fy = 411.0462341308594;
  double cx = 421.0187683105469;
  double cy = 240;

  // ---------- ORB 特征检测 ----------
  int orb_features = 2000;  // 最大特征点数
  float orb_scale = 1.2f;   // 金字塔缩放因子
  int orb_levels = 8;       // 金字塔层数

  // ---------- 特征匹配 ----------
  float ratio_thresh = 0.75f;  // Lowe's ratio test 阈值
  int min_matches = 30;        // 最少匹配对数

  // ---------- 对极几何 RANSAC ----------
  double ransac_thresh = 1.5;  // RANSAC 内点阈值（像素）
  double ransac_conf = 0.999;  // RANSAC 置信度
  int min_inliers = 15;        // 最少内点数
  double max_rot_per_frame = 30.0;  // 单帧最大合理旋转角(度), 超过则丢弃

  // ---------- 角度约束 ----------
  double initial_angle = 90.0;  // 初始角度（光轴垂直墙面）
  double angle_limit = 45.0;    // ±偏转限制

  // ---------- 时序平滑 ----------
  int smooth_window = 5;   // 滑动窗口大小
  double max_delta = 3.0;  // 单帧最大角度跳变（度）
};

// ==================================================================
//  帧间位姿求解结果
// ==================================================================
struct PoseResult {
  cv::Mat R;               // 3×3 相对旋转矩阵
  cv::Mat t;               // 3×1 相对平移向量
  double rot_angle = 0.0;  // 旋转角度（度）
  int inliers = 0;
  bool ok = false;
};

// ==================================================================
//  核心类：墙面角度估计器
// ==================================================================
class WallAngleEstimator {
 public:
  explicit WallAngleEstimator(const Config& cfg = Config{})
      : cfg_(cfg),
        R_accum_(cv::Mat::eye(3, 3, CV_64F)),
        last_angle_(cfg.initial_angle),
        frame_count_(0) {
    // 构造相机矩阵 K
    K_ = (cv::Mat_<double>(3, 3) << cfg_.fx, 0.0, cfg_.cx, 0.0, cfg_.fy,
          cfg_.cy, 0.0, 0.0, 1.0);

    // 初始化 ORB 检测器（可复用，避免每帧重建）
    orb_ = cv::ORB::create(cfg_.orb_features, cfg_.orb_scale, cfg_.orb_levels,
                           31,  // edgeThreshold
                           0,   // firstLevel
                           2,   // WTA_K
                           cv::ORB::HARRIS_SCORE,
                           31,  // patchSize
                           20   // fastThreshold
    );

    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);
  }

  // =============================================================
  //  主接口：输入一帧 BGR/灰度图像，返回墙面夹角（度）
  // =============================================================
  double process(const cv::Mat& frame) {
    frame_count_++;

    // 转灰度
    cv::Mat gray;
    if (frame.channels() == 3)
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else
      gray = frame;

    // 第一帧只做初始化
    if (prev_gray_.empty()) {
      prev_gray_ = gray.clone();
      std::cout << "[INFO] 首帧初始化, 角度 = " << cfg_.initial_angle << "°"
                << std::endl;
      return cfg_.initial_angle;
    }

    // ---------- Step 1: 对极几何求解帧间 R_rel ----------
    PoseResult pose = estimatePose(prev_gray_, gray);
    prev_gray_ = gray.clone();

    if (!pose.ok) {
      // 求解失败，保持上一帧角度（不更新累积旋转）
      if (frame_count_ % 30 == 0)
        std::cout << "[WARN] 帧 " << frame_count_
                  << " 位姿估计失败, 保持上一角度" << std::endl;
      return last_angle_;
    }

    // ---------- Step 2: 累积旋转矩阵 ----------
    // R_total ← R_rel × R_total
    // （从初始帧到当前帧的全局旋转）
    R_accum_ = pose.R * R_accum_;

    // SVD 重正交化，防止数值漂移
    enforceOrthogonality(R_accum_);

    // ---------- Step 3: 提取偏航角 ----------
    // 偏航角 = 绕 Y 轴的旋转角
    double yaw_deg = extractYaw(R_accum_);

    // ---------- Step 4: 换算为墙面角度 ----------
    // 初始90°，右偏为正yaw→角度减小，左偏为负yaw→角度增大
    double raw_angle = cfg_.initial_angle - yaw_deg;

    // 限幅
    double clamped = clampAngle(raw_angle);

    // 滑动窗口均值平滑
    double smoothed = smooth(clamped);

    // 防突变（限速）
    last_angle_ = rateLimit(smoothed, last_angle_);

    // 定期打印诊断信息
    if (frame_count_ % 30 == 0) {
      std::cout << std::fixed << std::setprecision(2) << "[Frame "
                << frame_count_ << "] "
                << "yaw=" << yaw_deg << "° "
                << "angle=" << last_angle_ << "° "
                << "deviation=" << (last_angle_ - cfg_.initial_angle) << "° "
                << "inliers=" << pose.inliers << " "
                << "rot/frame=" << pose.rot_angle << "°" << std::endl;
    }

    return last_angle_;
  }

  // 重置到初始状态（切换工况 / 累积误差过大时使用）
  void reset() {
    R_accum_ = cv::Mat::eye(3, 3, CV_64F);
    prev_gray_ = cv::Mat();
    last_angle_ = cfg_.initial_angle;
    angle_buf_.clear();
    frame_count_ = 0;
    std::cout << "[INFO] 估计器已重置" << std::endl;
  }

 private:
  Config cfg_;
  cv::Mat K_;          // 3×3 相机矩阵
  cv::Mat R_accum_;    // 累积旋转矩阵
  cv::Mat prev_gray_;  // 上一帧灰度图
  cv::Ptr<cv::ORB> orb_;
  cv::Ptr<cv::BFMatcher> matcher_;
  double last_angle_;
  std::deque<double> angle_buf_;
  int frame_count_;

  // ----------------------------------------------------------
  //  对极几何核心：从两帧图像求解相对旋转和平移
  // ----------------------------------------------------------
  //  流程:
  //    ORB检测 → KNN匹配 → Ratio Test →
  //    findEssentialMat(RANSAC) → recoverPose(R, t)
  // ----------------------------------------------------------
  PoseResult estimatePose(const cv::Mat& g1, const cv::Mat& g2) {
    PoseResult res;

    // ---- Step A: ORB 特征检测与描述 ----
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    orb_->detectAndCompute(g1, cv::noArray(), kp1, desc1);
    orb_->detectAndCompute(g2, cv::noArray(), kp2, desc2);

    if (desc1.empty() || desc2.empty()) return res;

    // ---- Step B: KNN 匹配 + Lowe's Ratio Test ----
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(desc1, desc2, knn_matches, 2);

    std::vector<cv::Point2f> pts1, pts2;
    for (auto& pair : knn_matches) {
      if (pair.size() == 2 &&
          pair[0].distance < cfg_.ratio_thresh * pair[1].distance) {
        pts1.push_back(kp1[pair[0].queryIdx].pt);
        pts2.push_back(kp2[pair[0].trainIdx].pt);
      }
    }

    if ((int)pts1.size() < cfg_.min_matches) return res;

    // ---- Step C: 求解本质矩阵 E（内置 RANSAC 去除外点） ----
    //   对极约束:  x₂ᵀ · E · x₁ = 0
    //   其中 E = [t]× · R
    cv::Mat inlier_mask;
    cv::Mat E =
        cv::findEssentialMat(pts1, pts2, K_, cv::RANSAC, cfg_.ransac_conf,
                             cfg_.ransac_thresh, inlier_mask);

    if (E.empty()) return res;

    // ---- Step D: 从 E 分解出 R 和 t（SVD 分解） ----
    //   E = U · Σ · Vᵀ
    //   R = U · W · Vᵀ  或  U · Wᵀ · Vᵀ (两组解，取合理者)
    //   t = U的最后一列
    cv::Mat R, t;
    int num_inliers = cv::recoverPose(E, pts1, pts2, K_, R, t, inlier_mask);

    if (num_inliers < cfg_.min_inliers) return res;

    // ---- Step E: 合理性检查 ----
    // 计算旋转角度（轴角表示）
    double trace = R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2);
    double cos_angle = std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0));
    double rot_angle_deg = std::acos(cos_angle) * 180.0 / CV_PI;

    if (rot_angle_deg > cfg_.max_rot_per_frame) {
      // 单帧旋转过大，大概率是误匹配
      if (frame_count_ % 30 == 0)
        std::cout << "[WARN] 单帧旋转过大: " << rot_angle_deg << "° > "
                  << cfg_.max_rot_per_frame << "°, 丢弃" << std::endl;
      return res;
    }

    // 通过所有检查
    res.R = R;
    res.t = t;
    res.rot_angle = rot_angle_deg;
    res.inliers = num_inliers;
    res.ok = true;
    return res;
  }

  // ----------------------------------------------------------
  //  从旋转矩阵提取偏航角（yaw, 绕 Y 轴旋转）
  // ----------------------------------------------------------
  //  坐标系: X→右, Y→下, Z→前(光轴)
  //
  //  绕 Y 轴旋转 θ 的矩阵:
  //    [cosθ   0   sinθ]
  //    [  0    1     0 ]
  //    [-sinθ  0   cosθ]
  //
  //  因此: yaw = atan2(R[0][2], R[2][2])
  // ----------------------------------------------------------
  static double extractYaw(const cv::Mat& R) {
    double yaw_rad = std::atan2(R.at<double>(0, 2), R.at<double>(2, 2));
    return yaw_rad * 180.0 / CV_PI;
  }

  // ----------------------------------------------------------
  //  SVD 重正交化：强制 R 为合法旋转矩阵
  // ----------------------------------------------------------
  //  累积乘法会导致 R 逐渐偏离 SO(3)，需定期修正
  //  R_ortho = U · Vᵀ, 保证 det(R) = +1
  // ----------------------------------------------------------
  static void enforceOrthogonality(cv::Mat& R) {
    cv::Mat U, W, Vt;
    cv::SVDecomp(R, W, U, Vt);
    R = U * Vt;
    // 确保是旋转矩阵（det=+1）而非反射（det=-1）
    if (cv::determinant(R) < 0) {
      R = -R;
    }
  }

  // 角度限幅到 [initial-angle_limit, initial+angle_limit]
  double clampAngle(double a) const {
    double lo = cfg_.initial_angle - cfg_.angle_limit;
    double hi = cfg_.initial_angle + cfg_.angle_limit;
    return std::max(lo, std::min(hi, a));
  }

  // 滑动窗口均值平滑
  double smooth(double a) {
    angle_buf_.push_back(a);
    if ((int)angle_buf_.size() > cfg_.smooth_window) angle_buf_.pop_front();
    return std::accumulate(angle_buf_.begin(), angle_buf_.end(), 0.0) /
           (double)angle_buf_.size();
  }

  // 限速器：限制单帧角度变化量，防止突跳
  double rateLimit(double current, double prev) const {
    double delta = current - prev;
    if (std::abs(delta) > cfg_.max_delta) {
      return prev + (delta > 0 ? cfg_.max_delta : -cfg_.max_delta);
    }
    return current;
  }
};

// ==================================================================
//  可视化：半圆仪表盘 + 偏差指示
// ==================================================================
void drawDashboard(cv::Mat& img, double angle, int inliers, const Config& cfg) {
  int w = img.cols;

  // ---------- 半圆仪表盘 ----------
  cv::Point center(w / 2, img.rows - 50);
  int R = 120;

  // 圆弧范围：45°→135° 映射到屏幕半圆
  double lo = cfg.initial_angle - cfg.angle_limit;  // 45
  double hi = cfg.initial_angle + cfg.angle_limit;  // 135

  // 背景弧（灰色）
  cv::ellipse(img, center, cv::Size(R, R), 0, 180 - hi, 180 - lo,
              cv::Scalar(80, 80, 80), 4, cv::LINE_AA);

  // 90°基准线（红色虚线效果）
  cv::Point ref_tip(center.x, center.y - R);
  cv::line(img, center, ref_tip, cv::Scalar(0, 0, 200), 1, cv::LINE_AA);

  // 当前角度指针（绿色）
  double screen_angle = (180.0 - angle) * CV_PI / 180.0;
  cv::Point tip(center.x + (int)(R * std::cos(screen_angle)),
                center.y - (int)(R * std::sin(screen_angle)));
  cv::line(img, center, tip, cv::Scalar(0, 230, 0), 3, cv::LINE_AA);
  cv::circle(img, tip, 5, cv::Scalar(0, 230, 0), -1);

  // 刻度标注
  auto putLabel = [&](double deg, const std::string& label) {
    double rad = (180.0 - deg) * CV_PI / 180.0;
    cv::Point p(center.x + (int)((R + 20) * std::cos(rad)),
                center.y - (int)((R + 20) * std::sin(rad)));
    cv::putText(img, label, cv::Point(p.x - 12, p.y + 5),
                cv::FONT_HERSHEY_PLAIN, 0.9, cv::Scalar(180, 180, 180), 1);
  };
  putLabel(45, "45");
  putLabel(90, "90");
  putLabel(135, "135");

  // ---------- 文字信息 ----------
  // 角度值
  std::ostringstream oss_angle;
  oss_angle << std::fixed << std::setprecision(1) << angle << " deg";
  cv::putText(img, oss_angle.str(), cv::Point(center.x - 50, center.y + 35),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

  // 偏差
  double deviation = angle - cfg.initial_angle;
  std::ostringstream oss_dev;
  oss_dev << std::showpos << std::fixed << std::setprecision(1) << deviation
          << " deg";
  cv::Scalar dev_color = (std::abs(deviation) > 30.0)
                             ? cv::Scalar(0, 0, 255)  // 大偏差红色警告
                             : cv::Scalar(0, 200, 200);
  cv::putText(img, "Deviation: " + oss_dev.str(), cv::Point(20, img.rows - 15),
              cv::FONT_HERSHEY_SIMPLEX, 0.55, dev_color, 1);

  // 状态信息（左上角）
  cv::putText(img, "Wall Angle Estimator", cv::Point(15, 25),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);

  std::ostringstream oss_info;
  oss_info << "Inliers: " << inliers;
  cv::putText(img, oss_info.str(), cv::Point(15, 50), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(150, 150, 150), 1);

  cv::putText(img, "[R] Reset  [ESC] Quit", cv::Point(15, 75),
              cv::FONT_HERSHEY_PLAIN, 0.9, cv::Scalar(120, 120, 120), 1);
}

// ==================================================================
//  main
// ==================================================================
int main(int argc, char** argv) {
  int cam_id = 0;
  if (argc > 1) cam_id = std::atoi(argv[1]);

  // ---------- 配置 ----------
  Config cfg;

  // !!! 必须根据实际标定结果修改以下内参 !!!
  // cfg.fx = ...;  cfg.fy = ...;
  // cfg.cx = ...;  cfg.cy = ...;
  //
  // 如果使用 Orbbec 深度相机，可通过 SDK 获取内参：
  //   auto param = pipe.getCameraParam();
  //   cfg.fx = param.depthIntrinsic.fx;
  //   cfg.fy = param.depthIntrinsic.fy;
  //   cfg.cx = param.depthIntrinsic.cx;
  //   cfg.cy = param.depthIntrinsic.cy;

  // ---------- 初始化 ----------
  WallAngleEstimator estimator(cfg);

  cv::VideoCapture cap(cam_id);
  if (!cap.isOpened()) {
    std::cerr << "[ERROR] 无法打开摄像头 " << cam_id << std::endl;
    return -1;
  }
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

  std::cout << "============================================\n"
            << "  墙面角度估计器 (对极几何 + 视觉里程计)\n"
            << "============================================\n"
            << "  初始角度: 90° (相机光轴垂直于墙面)\n"
            << "  有效范围: [45°, 135°]  ±45°\n"
            << "  按 R 重置  |  按 ESC 退出\n"
            << "============================================\n"
            << std::endl;

  // ---------- 主循环 ----------
  cv::Mat frame;
  while (true) {
    cap >> frame;
    if (frame.empty()) break;

    // 估计墙面角度
    double angle = estimator.process(frame);

    // 在画面上叠加仪表盘和信息
    drawDashboard(frame, angle, 0, cfg);

    cv::imshow("Wall Angle Estimator", frame);

    int key = cv::waitKey(1);
    if (key == 27) break;            // ESC 退出
    if (key == 'r' || key == 'R') {  // R 重置
      estimator.reset();
    }
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
