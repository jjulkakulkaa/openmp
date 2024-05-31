#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>


#include <random>
#include <ctime>
#include <cstdlib>

#include <curand.h>
#include <curand_kernel.h>

#include <fstream>




__global__ void encodeSegmentCUDA(const uchar* inputImage, uchar* outputImage1, uchar* outputImage2, int rows, int cols) {
    cudaError_t cudaStatus;

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols) {
        int baseRow = i * 2;
        int baseCol = j * 2;
        int inputIdx = i * cols + j;

        // Random number generation
        curandState state;
        curand_init((unsigned long long)clock() + i + j, 0, 0, &state);
        float random = curand_uniform(&state);

        uchar inputPixel = inputImage[inputIdx];

        // Access pixel positions
        int pos1_1 = baseRow * cols * 2 + baseCol;
        int pos1_2 = baseRow * cols * 2 + baseCol + 1;
        int pos1_3 = (baseRow + 1) * cols * 2 + baseCol;
        int pos1_4 = (baseRow + 1) * cols * 2 + baseCol + 1;

        int pos2_1 = baseRow * cols * 2 + baseCol;
        int pos2_2 = baseRow * cols * 2 + baseCol + 1;
        int pos2_3 = (baseRow + 1) * cols * 2 + baseCol;
        int pos2_4 = (baseRow + 1) * cols * 2 + baseCol + 1;

        if (inputPixel == 0) {
            if (random < 0.333f) {
                random = curand_uniform(&state);
                if (random < 0.5f) {
                    outputImage1[pos1_1] = 0;
                    outputImage1[pos1_2] = 0;
                    outputImage1[pos1_3] = 255;
                    outputImage1[pos1_4] = 255;

                    outputImage2[pos2_1] = 255;
                    outputImage2[pos2_2] = 255;
                    outputImage2[pos2_3] = 0;
                    outputImage2[pos2_4] = 0;
                }
                else {
                    outputImage2[pos2_1] = 0;
                    outputImage2[pos2_2] = 0;
                    outputImage2[pos2_3] = 255;
                    outputImage2[pos2_4] = 255;

                    outputImage1[pos1_1] = 255;
                    outputImage1[pos1_2] = 255;
                    outputImage1[pos1_3] = 0;
                    outputImage1[pos1_4] = 0;
                }
            }
            else if (random < 0.666f) {
                random = curand_uniform(&state);
                if (random < 0.5f) {
                    outputImage1[pos1_1] = 0;
                    outputImage1[pos1_2] = 255;
                    outputImage1[pos1_3] = 0;
                    outputImage1[pos1_4] = 255;

                    outputImage2[pos2_1] = 255;
                    outputImage2[pos2_2] = 0;
                    outputImage2[pos2_3] = 255;
                    outputImage2[pos2_4] = 0;
                }
                else {
                    outputImage2[pos2_1] = 0;
                    outputImage2[pos2_2] = 255;
                    outputImage2[pos2_3] = 0;
                    outputImage2[pos2_4] = 255;

                    outputImage1[pos1_1] = 255;
                    outputImage1[pos1_2] = 0;
                    outputImage1[pos1_3] = 255;
                    outputImage1[pos1_4] = 0;
                }
            }
            else {
                random = curand_uniform(&state);
                if (random < 0.5f) {
                    outputImage1[pos1_1] = 0;
                    outputImage1[pos1_2] = 255;
                    outputImage1[pos1_3] = 255;
                    outputImage1[pos1_4] = 0;

                    outputImage2[pos2_1] = 255;
                    outputImage2[pos2_2] = 0;
                    outputImage2[pos2_3] = 0;
                    outputImage2[pos2_4] = 255;
                }
                else {
                    outputImage2[pos2_1] = 0;
                    outputImage2[pos2_2] = 255;
                    outputImage2[pos2_3] = 255;
                    outputImage2[pos2_4] = 0;

                    outputImage1[pos1_1] = 255;
                    outputImage1[pos1_2] = 0;
                    outputImage1[pos1_3] = 0;
                    outputImage1[pos1_4] = 255;
                }
            }
        }
        else {
            if (random < 0.166f) {
                outputImage1[pos1_1] = 0;
                outputImage1[pos1_2] = 0;
                outputImage1[pos1_3] = 255;
                outputImage1[pos1_4] = 255;

                outputImage2[pos2_1] = 0;
                outputImage2[pos2_2] = 0;
                outputImage2[pos2_3] = 255;
                outputImage2[pos2_4] = 255;
            }
            else if (random < 0.333f) {
                outputImage1[pos1_1] = 255;
                outputImage1[pos1_2] = 255;
                outputImage1[pos1_3] = 0;
                outputImage1[pos1_4] = 0;

                outputImage2[pos2_1] = 255;
                outputImage2[pos2_2] = 255;
                outputImage2[pos2_3] = 0;
                outputImage2[pos2_4] = 0;
            }
            else if (random < 0.499f) {
                outputImage1[pos1_1] = 0;
                outputImage1[pos1_2] = 255;
                outputImage1[pos1_3] = 0;
                outputImage1[pos1_4] = 255;

                outputImage2[pos2_1] = 0;
                outputImage2[pos2_2] = 255;
                outputImage2[pos2_3] = 0;
                outputImage2[pos2_4] = 255;
            }
            else if (random < 0.666f) {
                outputImage1[pos1_1] = 255;
                outputImage1[pos1_2] = 0;
                outputImage1[pos1_3] = 255;
                outputImage1[pos1_4] = 0;

                outputImage2[pos2_1] = 255;
                outputImage2[pos2_2] = 0;
                outputImage2[pos2_3] = 255;
                outputImage2[pos2_4] = 0;
            }
            else if (random < 0.833f) {
                outputImage1[pos1_1] = 255;
                outputImage1[pos1_2] = 0;
                outputImage1[pos1_3] = 0;
                outputImage1[pos1_4] = 255;

                outputImage2[pos2_1] = 255;
                outputImage2[pos2_2] = 0;
                outputImage2[pos2_3] = 0;
                outputImage2[pos2_4] = 255;
            }
            else {
                outputImage1[pos1_1] = 0;
                outputImage1[pos1_2] = 255;
                outputImage1[pos1_3] = 255;
                outputImage1[pos1_4] = 0;

                outputImage2[pos2_1] = 0;
                outputImage2[pos2_2] = 255;
                outputImage2[pos2_3] = 255;
                outputImage2[pos2_4] = 0;
            }
        }
    }
}


void encodeImage(const cv::Mat& inputImage) {
    int rows = inputImage.rows;
    int cols = inputImage.cols;

    // Allocate memory for images on the GPU
    uchar* d_inputImage, * d_outputImage1, * d_outputImage2;
    cudaMalloc(&d_inputImage, rows * cols * sizeof(uchar));
    cudaMalloc(&d_outputImage1, rows * cols * 4 * sizeof(uchar)); // Output images are 2x larger
    cudaMalloc(&d_outputImage2, rows * cols * 4 * sizeof(uchar));

    // Copy input image to GPU memory
    cudaMemcpy(d_inputImage, inputImage.data, rows * cols * sizeof(uchar), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    // Launch CUDA kernel
    encodeSegmentCUDA << <gridDim, blockDim >> > (d_inputImage, d_outputImage1, d_outputImage2, rows, cols);
    cudaDeviceSynchronize(); 

    // Copy output images from GPU to CPU
    cv::Mat outputImage1(rows * 2, cols * 2, CV_8UC1);
    cv::Mat outputImage2(rows * 2, cols * 2, CV_8UC1);
    cudaMemcpy(outputImage1.data, d_outputImage1, rows * cols * 4 * sizeof(uchar), cudaMemcpyDeviceToHost);
    cudaMemcpy(outputImage2.data, d_outputImage2, rows * cols * 4 * sizeof(uchar), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage1);
    cudaFree(d_outputImage2);

    // Process or save the output images as needed

}

double count_time( const cv::Mat& inputImage, cv::Mat& outputImage1, cv::Mat& outputImage2) {
    auto start = std::chrono::high_resolution_clock::now();
    encodeImage(inputImage);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("%f\n", duration);
    return duration.count();
}



cv::Mat generate_random_image(int rows, int cols) {
    cv::Mat img(rows, cols, CV_8UC1);
    cv::randu(img, cv::Scalar(0), cv::Scalar(256));
    cv::threshold(img, img, 128, 255, cv::THRESH_BINARY);
    return img;
}

void time_test(const std::vector<cv::Size>& image_sizes) {

    for (const auto& size : image_sizes) {
        cv::Mat inputImage = generate_random_image(size.height, size.width);
        cv::Mat outputImage1, outputImage2;

        std::vector<double> parallel_times;


        double mes_time = 0;
        for (int i = 0; i < 10; i++) {
            mes_time += count_time(inputImage, outputImage1, outputImage2);

        }
        double parallel_time = mes_time / 10;

        parallel_times.push_back(parallel_time);
        std::cout << "Parallel time with " << size << ": " << parallel_time << " seconds\n"; 
        std::string filename = "results_" + std::to_string(size.width) + "x" + std::to_string(size.height) + ".csv";
     
    }
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);
    
    //cv::Mat inputImage = cv::imread("data/input_image.png", cv::IMREAD_GRAYSCALE);
    //if (inputImage.empty()) {
    //    std::cerr << "Cannot read img!" << std::endl;
    //    return -1;
    //}

   // std::vector<cv::Size> image_sizes = {
   //cv::Size(256, 256),
   //cv::Size(512, 512),
   //cv::Size(1024, 1024),
   ////cv::Size(2048, 2048),
   ////cv::Size(4096, 4096),
   ////cv::Size(8192, 8192)
   // };
   // printf("sizes created");

   // //encodeImage(inputImage, 1);

   // time_test(image_sizes);

    cv::Mat inputImage = generate_random_image(100, 100);
    cv::Mat outputImage1, outputImage2;
    encodeImage(inputImage);
    return 0;
}
