#include <iostream>
#include "gms_matcher.h"
#include <chrono>
#include <opencv2/cudafeatures2d.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <ceres/ceres.h>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include "rotation.h"

#define USE_GPU
using cuda::GpuMat;
using namespace std;
using namespace cv;

Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type);
struct Keypoints_GMS_Strcut{
  vector<cv::KeyPoint> kp1;
  vector<cv::KeyPoint> kp2;
  vector<DMatch> gms_matches;
};
Keypoints_GMS_Strcut GmsMatch(Mat &img1, Mat &img2);
struct Pose_Strcut{
  Mat R;
  Mat t;
};
Pose_Strcut Pose_Estimation_2d2d(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2,
								vector<DMatch> matches);
const Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
inline Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d
	  (
		  (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
		  (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
	  );
}
void Verify_Epipolar_Constraints(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2,
								 vector<DMatch> matches, Mat &t_x, Mat &R);
vector<Point3f> Points_trans_2d3d(cv::Mat &img, cv::Mat &depth, vector<DMatch> &matches,
									vector<KeyPoint> kp1, vector<KeyPoint> kp2);
string Filename_Read(int n, string str);

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

Sophus::SE3d BAadjustment(const VecVector3d &points_3d, const VecVector2d &points_2d, Sophus::SE3d pose);

struct PnPCeres
{
  PnPCeres ( Point2f uv,Point3f xyz ) : _uv(uv),_xyz(xyz) {} //构造函数
  // 残差的计算
  template <typename T>
  bool operator() (
	  const T* const camera,     // 位姿参数，有6维
	  T* residual ) const     // 残差
  {
	T p[3];
	T point[3];
	point[0] = T(_xyz.x);
	point[1] = T(_xyz.y);
	point[2] = T(_xyz.z);
	AngleAxisRotatePoint(camera, point, p);//计算RP
	p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];
	T xp = p[0] / p[2];
	T yp = p[1] / p[2];//xp,yp是归一化坐标，深度为p[2]
	T u_= xp * K.at<double>(0,0) + K.at<double>(0,2);
	T v_= yp * K.at<double>(1,1) + K.at<double>(1,2);
	residual[0] = T(_uv.x) - u_;
	residual[1] = T(_uv.y) - v_;
	return true;
  }
  static ceres::CostFunction* Create(const Point2f uv,const Point3f xyz) {
	return (new ceres::AutoDiffCostFunction<PnPCeres, 2, 6>(
		new PnPCeres(uv,xyz)));
  }
  const Point2f _uv;
  const Point3f _xyz;
};

const int image_number = 160;
const string rgb_path = "/home/xz/CLionProjects/gms_slam/desk_rgb/";
const string depth_path = "/home/xz/CLionProjects/gms_slam/desk_depth/";
const string path = "/home/xz/CLionProjects/gms_slam/rgbd-scenes/meeting_small/meeting_small_1/";

int main()
{
#ifdef USE_GPU
  int flag = cuda::getCudaEnabledDeviceCount();
	if (flag != 0) { cuda::setDevice(0); }
#endif // USE_GPU

//  Mat img1 = imread("../desk_rgb/desk_1_4.png");
//  Mat img2 = imread("../desk_rgb/desk_1_5.png");
//  Mat depth_1 = imread("../desk_depth/desk_1_1_depth.png", CV_LOAD_IMAGE_UNCHANGED);
  for (int i = 1; i != image_number ; ++i) {

    Mat img1 = imread(path + Filename_Read(i, ".png"));
    Mat depth_1 = imread(path + Filename_Read(i, "_depth.png"));
    int j = i + 1;
    Mat img2 = imread(path + Filename_Read(j, ".png"));

	chrono::steady_clock::time_point t11 = chrono::steady_clock::now();
	Keypoints_GMS_Strcut gms_matches_frames = GmsMatch(img1, img2);
	chrono::steady_clock::time_point t12 = chrono::steady_clock::now();
	chrono::duration<double> time_gms = chrono::duration_cast<chrono::duration<double>>(t12 - t11);
	cout << "gms_match cost time: " << time_gms.count() << " seconds." << endl;
	//Pose_Strcut pose_2d2d = Pose_Estimation_2d2d(gms_matches_frames.kp1, gms_matches_frames.kp2,
	//	                                           gms_matches_frames.gms_matches);
	//Verify_Epipolar_Constraints(gms_matches_frames.kp1, gms_matches_frames.kp2,
	//	                          gms_matches_frames.gms_matches, pose_2d2d.t, pose_2d2d.R);
	Mat show =
		DrawInlier(img1, img2, gms_matches_frames.kp1, gms_matches_frames.kp2, gms_matches_frames.gms_matches, 2);
	imshow("show", show);
	waitKey(30);

	vector<Point3f> pts_3d;
	vector<Point2f> pts_2d;
	int number_d0 = 0;
	for (DMatch m: gms_matches_frames.gms_matches) {
	  ushort d =
		  depth_1.ptr<unsigned short>(int(gms_matches_frames.kp1[m.queryIdx].pt.y))[int(gms_matches_frames.kp1[m.queryIdx].pt.x)];
	  if (d == 0)   // bad depth
	  {
		number_d0++;
		continue;
	  }
	  float dd = d / 5000.0;
	  Point2d p1 = pixel2cam(gms_matches_frames.kp1[m.queryIdx].pt, K);
	  pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
	  pts_2d.push_back(gms_matches_frames.kp2[m.trainIdx].pt);
	}
	cout << "3d-2d pairs: " << pts_3d.size() << endl;
	cout << "bad_depth_points: " << number_d0 << std::endl;

	/******OpenCV求解 pnp******/
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	Mat r, t;
	solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
	Mat R;
	cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
	cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;
	cout << "R=" << endl << R << endl;
	cout << "t=" << endl << t << endl;

	/************手写pnp******/
//	VecVector3d pts_3d_eigen;
//	VecVector2d pts_2d_eigen;
//	for (size_t i = 0; i < pts_3d.size(); ++i) {
//	  pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
//	  pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
//	}
//	Sophus::SE3d pose_gn;
//	Sophus::SE3d pose_opt = BAadjustment(pts_3d_eigen, pts_2d_eigen, pose_gn);

/**************Ceres优化pnp***********/
//  double camera[6]={0,1,2,0,0,0};
//  ceres::Problem problem;
//  for (int i = 0; i < pts_2d.size(); ++i)
//  {
//	ceres::CostFunction* cost_function =
//		PnPCeres::Create(pts_2d[i],pts_3d[i]);
//	problem.AddResidualBlock(cost_function,
//							 NULL /* squared loss */,
//							 camera);
//  }
//  ceres::Solver::Options options;
//  options.linear_solver_type = ceres::DENSE_SCHUR;
//  options.minimizer_progress_to_stdout = true;
//  ceres::Solver::Summary summary;
//  ceres::Solve(options, &problem, &summary);
//  std::cout << summary.FullReport() << "\n";
//  Mat R_vec = (Mat_<double>(3,1) << camera[0],camera[1],camera[2]);//数组转cv向量
//  Mat R_cvest;
//  cv::Rodrigues(R_vec,R_cvest);//罗德里格斯公式，旋转向量转旋转矩阵
//  cout<<"R_cvest="<<R_cvest<<endl;
//  Eigen::Matrix3d R_est;
//  cv2eigen(R_cvest,R_est);//cv矩阵转eigen矩阵
//  cout<<"R_est="<<R_est<<endl;
//  Eigen::Vector3d t_est(camera[3],camera[4],camera[5]);
//  cout<<"t_est="<<t_est<<endl;
//  Eigen::Isometry3d T(R_est);//构造变换矩阵与输出
//  T.pretranslate(t_est);
//  cout<<T.matrix()<<endl;


  }

  return 0;
}

Keypoints_GMS_Strcut GmsMatch(Mat &img1, Mat &img2) {
  vector<KeyPoint> kp1, kp2;
  Mat d1, d2;
  vector<DMatch> matches_all, matches_gms;

  Ptr<ORB> orb = ORB::create(10000);
  orb->setFastThreshold(0);

  orb->detectAndCompute(img1, Mat(), kp1, d1);
  orb->detectAndCompute(img2, Mat(), kp2, d2);

#ifdef USE_GPU
  GpuMat gd1(d1), gd2(d2);
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	matcher->match(gd1, gd2, matches_all);
#else
  BFMatcher matcher(NORM_HAMMING);
  matcher.match(d1, d2, matches_all);
#endif

  // GMS filter
  std::vector<bool> vbInliers;
  gms_matcher gms(kp1, img1.size(), kp2, img2.size(), matches_all);
  int num_inliers = gms.GetInlierMask(vbInliers, false, false);
  cout << "Get total " << num_inliers << " matches." << endl;

  // collect matches
  for (size_t i = 0; i < vbInliers.size(); ++i)
  {
	if (vbInliers[i] == true)
	{
	  matches_gms.push_back(matches_all[i]);
	}
  }
  Keypoints_GMS_Strcut tmp_struct;
  tmp_struct.kp1 = kp1;
  tmp_struct.kp2 = kp2;
  tmp_struct.gms_matches = matches_gms;
  return tmp_struct;
}

Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type) {
  const int height = max(src1.rows, src2.rows);
  const int width = src1.cols + src2.cols;
  Mat output(height, width, CV_8UC3, Scalar(0, 0, 0));
  src1.copyTo(output(Rect(0, 0, src1.cols, src1.rows)));
  src2.copyTo(output(Rect(src1.cols, 0, src2.cols, src2.rows)));

  if (type == 1)
  {
	for (size_t i = 0; i < inlier.size(); i++)
	{
	  Point2f left = kpt1[inlier[i].queryIdx].pt;
	  Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
	  line(output, left, right, Scalar(0, 255, 255));
	}
  }
  else if (type == 2)
  {
	for (size_t i = 0; i < inlier.size(); i++)
	{
	  Point2f left = kpt1[inlier[i].queryIdx].pt;
	  Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
	  line(output, left, right, Scalar(255, 0, 0));
	}

	for (size_t i = 0; i < inlier.size(); i++)
	{
	  Point2f left = kpt1[inlier[i].queryIdx].pt;
	  Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
	  circle(output, left, 1, Scalar(0, 255, 255), 2);
	  circle(output, right, 1, Scalar(0, 255, 0), 2);
	}
  }

  return output;
}

Pose_Strcut Pose_Estimation_2d2d(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2,
								 vector<DMatch> matches){
  vector<Point2f> points1;
  vector<Point2f> points2;

  for (int i = 0; i < (int)matches.size(); ++i) {
	points1.push_back(keypoints_1[matches[i].queryIdx].pt);
	points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }

  //-- 计算基础矩阵
  Mat fundamental_matrix;
  fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
  cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;
  //-- 计算本质矩阵
  Point2d principal_point(325.1, 249.7);  //相机光心, TUM dataset标定值
  double focal_length = 521;      //相机焦距, TUM dataset标定值
  Mat essential_matrix;
  essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
  cout << "essential_matrix is " << endl << essential_matrix << endl;

  Mat R_tmp, t_tmp;
  Pose_Strcut pose_strcut_tmp;
  recoverPose(essential_matrix, points1, points2, R_tmp, t_tmp, focal_length, principal_point);
  pose_strcut_tmp.R = R_tmp;
  pose_strcut_tmp.t = t_tmp;

  return pose_strcut_tmp;
}

void Verify_Epipolar_Constraints(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2,
								 vector<DMatch> matches, Mat &t, Mat &R){
  Mat t_x =
	  (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
		  t.at<double>(2, 0), 0, -t.at<double>(0, 0),
		  -t.at<double>(1, 0), t.at<double>(0, 0), 0);
  for (DMatch m: matches){
    Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
	Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
	Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
	Mat d = y2.t() * t_x * R * y1;
	cout << "epipolar constraint = " << d << endl;
  }
}
string Filename_Read(int n, string str){
  string file_tmp;
  string file_full = "meeting_small_1_";
  std::stringstream stm;
  stm << n;
  stm >> file_tmp;
  file_tmp += str;
  file_full += file_tmp;
  return file_full;
}

Sophus::SE3d BAadjustment(const VecVector3d &points_3d, const VecVector2d &points_2d, Sophus::SE3d pose){

  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  const int iterations = 10;
  double cost = 0, lastcost = 0;
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 0);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  for (int iter = 0; iter < iterations; ++iter) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Vector6d b = Vector6d::Zero();

	cost = 0;
	// compute cost
	for (int i = 0; i < points_3d.size(); i++) {
	  Eigen::Vector3d pc = pose * points_3d[i];
	  double inv_z = 1.0 / pc[2];
	  double inv_z2 = inv_z * inv_z;
	  Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);

	  Eigen::Vector2d e = points_2d[i] - proj;

	  cost += e.squaredNorm();
	  Eigen::Matrix<double, 2, 6> J;
	  J << -fx * inv_z,
		  0,
		  fx * pc[0] * inv_z2,
		  fx * pc[0] * pc[1] * inv_z2,
		  -fx - fx * pc[0] * pc[0] * inv_z2,
		  fx * pc[1] * inv_z,
		  0,
		  -fy * inv_z,
		  fy * pc[1] * inv_z2,
		  fy + fy * pc[1] * pc[1] * inv_z2,
		  -fy * pc[0] * pc[1] * inv_z2,
		  -fy * pc[0] * inv_z;

	  H += J.transpose() * J;
	  b += -J.transpose() * e;
	}

	Vector6d dx;
	dx = H.ldlt().solve(b);

	if (isnan(dx[0])) {
	  cout << "result is nan!" << endl;
	  break;
	}

	if (iter > 0 && cost >= lastcost) {
	  // cost increase, update is not good
	  cout << "cost: " << cost << ", last cost: " << lastcost << endl;
	  break;
	}
	// update your estimation
	pose = Sophus::SE3d::exp(dx) * pose;
	lastcost = cost;

	cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl;
	if (dx.norm() < 1e-6) {
	  // converge
	  break;
	}
  }
  return pose;
}

