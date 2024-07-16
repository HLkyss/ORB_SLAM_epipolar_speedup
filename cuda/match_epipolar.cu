#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include "match_epipolar.cuh"

__device__ int hammingDistance(const uchar* desc1, const uchar* desc2, int length) {//定义一个用于计算 Hamming 距离的 CUDA 函数
    int dist = 0;
    for (int i = 0; i < length; i++) {
        dist += __popc(desc1[i] ^ desc2[i]);
    }
    return dist;
}

// CUDA核函数声明
__global__ void matchEpipolarConstraint(
        const float* left_kps_x, const float* left_kps_y, int num_left_kps,
        const float* right_kps_x, const float* right_kps_y, int num_right_kps,
        const uchar* left_descs, int left_descs_step,
        const uchar* right_descs, int right_descs_step,
        const float* F, int desc_len,
        int* match_indices, float* match_dists)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_left_kps) return;

    float best_dist = 1e9;
    int best_j = -1;

    float p1_x = left_kps_x[i];
    float p1_y = left_kps_y[i];

    for (int j = 0; j < num_right_kps; ++j) {
        float p2_x = right_kps_x[j];
        float p2_y = right_kps_y[j];

        float3 p1 = make_float3(p1_x, p1_y, 1.0f);
        float3 p2 = make_float3(p2_x, p2_y, 1.0f);

        float error = p2.x * (F[0] * p1.x + F[1] * p1.y + F[2]) +
                      p2.y * (F[3] * p1.x + F[4] * p1.y + F[5]) +
                      (F[6] * p1.x + F[7] * p1.y + F[8]);
        if (fabs(error) < 0.5f) {
//            float dist = 0.0f;
//            for (int k = 0; k < desc_len; ++k) {
//                dist += abs(left_descs[i * left_descs_step + k] - right_descs[j * right_descs_step + k]);
//            }
            int dist = hammingDistance(&left_descs[i * left_descs_step], &right_descs[j * right_descs_step], desc_len);
            if (dist < best_dist) {
                best_dist = dist;
                best_j = j;
            }
//            printf("CUDA Match: left %d, right %d, error %f, dist %f\n", i, best_j, error, best_dist);
        }
    }

    match_indices[i] = best_j;
    match_dists[i] = best_dist;
}

extern "C" void launchMatchEpipolarConstraint(
        const std::vector<float>& left_kps_x, const std::vector<float>& left_kps_y,
        const std::vector<float>& right_kps_x, const std::vector<float>& right_kps_y,
        const cv::Mat& left_descs, const cv::Mat& right_descs,
        const Eigen::Matrix3f& F,
        std::vector<int>& match_indices, std::vector<float>& match_dists)
{
    int num_left_kps = left_kps_x.size();
    int num_right_kps = right_kps_x.size();
    int desc_len = left_descs.cols;

    float *d_left_kps_x, *d_left_kps_y;
    float *d_right_kps_x, *d_right_kps_y;
    uchar *d_left_descs, *d_right_descs;
    float *d_F;
    int *d_match_indices;
    float *d_match_dists;

    cudaMalloc(&d_left_kps_x, num_left_kps * sizeof(float));
    cudaMalloc(&d_left_kps_y, num_left_kps * sizeof(float));
    cudaMalloc(&d_right_kps_x, num_right_kps * sizeof(float));
    cudaMalloc(&d_right_kps_y, num_right_kps * sizeof(float));
    cudaMalloc(&d_left_descs, left_descs.rows * left_descs.cols * sizeof(uchar));
    cudaMalloc(&d_right_descs, right_descs.rows * right_descs.cols * sizeof(uchar));
    cudaMalloc(&d_F, 9 * sizeof(float));
    cudaMalloc(&d_match_indices, num_left_kps * sizeof(int));
    cudaMalloc(&d_match_dists, num_left_kps * sizeof(float));

    cudaMemcpy(d_left_kps_x, left_kps_x.data(), num_left_kps * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_left_kps_y, left_kps_y.data(), num_left_kps * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right_kps_x, right_kps_x.data(), num_right_kps * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right_kps_y, right_kps_y.data(), num_right_kps * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_left_descs, left_descs.ptr<uchar>(), left_descs.rows * left_descs.cols * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right_descs, right_descs.ptr<uchar>(), right_descs.rows * right_descs.cols * sizeof(uchar), cudaMemcpyHostToDevice);

    float F_array[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            F_array[i * 3 + j] = F(i, j);
    cudaMemcpy(d_F, F_array, 9 * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (num_left_kps + block_size - 1) / block_size;
    matchEpipolarConstraint<<<grid_size, block_size>>>(
            d_left_kps_x, d_left_kps_y, num_left_kps,
            d_right_kps_x, d_right_kps_y, num_right_kps,
            d_left_descs, left_descs.step1(),
            d_right_descs, right_descs.step1(),
            d_F, desc_len,
            d_match_indices, d_match_dists);

    cudaMemcpy(match_indices.data(), d_match_indices, num_left_kps * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(match_dists.data(), d_match_dists, num_left_kps * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_left_kps_x);
    cudaFree(d_left_kps_y);
    cudaFree(d_right_kps_x);
    cudaFree(d_right_kps_y);
    cudaFree(d_left_descs);
    cudaFree(d_right_descs);
    cudaFree(d_F);
    cudaFree(d_match_indices);
    cudaFree(d_match_dists);
}
