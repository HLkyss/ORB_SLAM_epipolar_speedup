#ifndef MATCH_EPIPOLAR_CUH
#define MATCH_EPIPOLAR_CUH

#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

__device__ int hammingDistance(const uchar* desc1, const uchar* desc2, int length);
// CUDA核函数声明
__global__ void matchEpipolarConstraint(
        const float* left_kps_x, const float* left_kps_y, int num_left_kps,
        const float* right_kps_x, const float* right_kps_y, int num_right_kps,
        const uchar* left_descs, int left_descs_step,
        const uchar* right_descs, int right_descs_step,
        const float* F, int desc_len,
        int* match_indices, float* match_dists);

// 启动CUDA核函数的声明
extern "C" void launchMatchEpipolarConstraint(
        const std::vector<float>& left_kps_x, const std::vector<float>& left_kps_y,
        const std::vector<float>& right_kps_x, const std::vector<float>& right_kps_y,
        const cv::Mat& left_descs, const cv::Mat& right_descs,
        const Eigen::Matrix3f& F,
        std::vector<int>& match_indices, std::vector<float>& match_dists);

#endif // MATCH_EPIPOLAR_CUH
