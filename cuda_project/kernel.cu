#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <fstream>


// random binary image (0 or 255)
void generateRandomImage(std::vector<unsigned char>& image, unsigned int width, unsigned int height) {
    image.resize(width * height);
    srand(time(NULL));
    for (unsigned int i = 0; i < width * height; ++i) {
        image[i] = rand() % 2 * 255; // Either 0 or 255
    }
}

__device__ float rand_float(int* seed)
{
    *seed = *seed * 1103515245 + 12345;
    unsigned int rand_int = ((*seed) / 65536) % 32768;
    return (float)rand_int / 32767.0f;
}

__global__ void encodeImage(unsigned char* output1, unsigned char* output2, const unsigned char* input, unsigned int width, unsigned int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        int idx = y * width + x;

        // read pixel
        unsigned char pixel = input[idx];

        int seed = idx;

        float random_number = rand_float(&seed);

        // out pixels
        unsigned char out1_pixel1, out1_pixel2, out1_pixel3, out1_pixel4;
        unsigned char out2_pixel1, out2_pixel2, out2_pixel3, out2_pixel4;

        if (pixel == 255) { // White
            if (random_number < 0.1666f) {
                out1_pixel1 = 0;
                out1_pixel2 = 0;
                out1_pixel3 = 255;
                out1_pixel4 = 255;

                out2_pixel1 = 0;
                out2_pixel2 = 0;
                out2_pixel3 = 255;
                out2_pixel4 = 255;
            }
            else if (random_number < 0.3333f) {
                out1_pixel1 = 255;
                out1_pixel2 = 255;
                out1_pixel3 = 0;
                out1_pixel4 = 0;

                out2_pixel1 = 255;
                out2_pixel2 = 255;
                out2_pixel3 = 0;
                out2_pixel4 = 0;
            }
            else if (random_number < 0.5f) {
                out1_pixel1 = 0;
                out1_pixel2 = 255;
                out1_pixel3 = 0;
                out1_pixel4 = 255;

                out2_pixel1 = 0;
                out2_pixel2 = 255;
                out2_pixel3 = 0;
                out2_pixel4 = 255;
            }
            else if (random_number < 0.6666f) {
                out1_pixel1 = 255;
                out1_pixel2 = 0;
                out1_pixel3 = 255;
                out1_pixel4 = 0;

                out2_pixel1 = 255;
                out2_pixel2 = 0;
                out2_pixel3 = 255;
                out2_pixel4 = 0;
            }
            else if (random_number < 0.83333) {
                out1_pixel1 = 255;
                out1_pixel2 = 0;
                out1_pixel3 = 0;
                out1_pixel4 = 255;

                out2_pixel1 = 255;
                out2_pixel2 = 0;
                out2_pixel3 = 0;
                out2_pixel4 = 255;
            }
            else {
                out1_pixel1 = 0;
                out1_pixel2 = 255;
                out1_pixel3 = 255;
                out1_pixel4 = 0;

                out2_pixel1 = 0;
                out2_pixel2 = 255;
                out2_pixel3 = 255;
                out2_pixel4 = 0;
            }
        }
        else { // Black
            float sec_random_number = rand_float(&seed);
            if (random_number < 0.333f) {
                if (sec_random_number < 0.5f) {
                    out1_pixel1 = 0;
                    out1_pixel2 = 0;
                    out1_pixel3 = 255;
                    out1_pixel4 = 255;

                    out2_pixel1 = 255;
                    out2_pixel2 = 255;
                    out2_pixel3 = 0;
                    out2_pixel4 = 0;
                }
                else {
                    out1_pixel1 = 255;
                    out1_pixel2 = 255;
                    out1_pixel3 = 0;
                    out1_pixel4 = 0;

                    out2_pixel1 = 0;
                    out2_pixel2 = 0;
                    out2_pixel3 = 255;
                    out2_pixel4 = 255;
                }
            }
            else if (random_number < 0.666f) {
                if (sec_random_number < 0.5f) {
                    out1_pixel1 = 0;
                    out1_pixel2 = 255;
                    out1_pixel3 = 0;
                    out1_pixel4 = 255;

                    out2_pixel1 = 255;
                    out2_pixel2 = 0;
                    out2_pixel3 = 255;
                    out2_pixel4 = 0;
                }
                else {
                    out1_pixel1 = 255;
                    out1_pixel2 = 0;
                    out1_pixel3 = 255;
                    out1_pixel4 = 0;

                    out2_pixel1 = 0;
                    out2_pixel2 = 255;
                    out2_pixel3 = 0;
                    out2_pixel4 = 255;
                }
            }
            else {
                if (sec_random_number < 0.5f) {
                    out1_pixel1 = 255;
                    out1_pixel2 = 0;
                    out1_pixel3 = 0;
                    out1_pixel4 = 255;

                    out2_pixel1 = 0;
                    out2_pixel2 = 255;
                    out2_pixel3 = 255;
                    out2_pixel4 = 0;
                }
                else {
                    out1_pixel1 = 0;
                    out1_pixel2 = 255;
                    out1_pixel3 = 255;
                    out1_pixel4 = 0;

                    out2_pixel1 = 255;
                    out2_pixel2 = 0;
                    out2_pixel3 = 0;
                    out2_pixel4 = 255;
                }
            }
        }

        // output 1
        int out_x = 2 * x;
        int out_y = 2 * y;
        output1[out_y * (2 * width) + out_x] = out1_pixel1;
        output1[out_y * (2 * width) + out_x + 1] = out1_pixel2;
        output1[(out_y + 1) * (2 * width) + out_x] = out1_pixel3;
        output1[(out_y + 1) * (2 * width) + out_x + 1] = out1_pixel4;

        // Write to output2
        output2[out_y * (2 * width) + out_x] = out2_pixel1;
        output2[out_y * (2 * width) + out_x + 1] = out2_pixel2;
        output2[(out_y + 1) * (2 * width) + out_x] = out2_pixel3;
        output2[(out_y + 1) * (2 * width) + out_x + 1] = out2_pixel4;
    }
}

int main()
{
    // Test different image sizes
    std::vector<unsigned int> imageSizes = { 256, 512, 1024, 2048, 4096, 8192 }; // Adjust as needed
    std::vector<dim3> threadConfigs = { dim3(8, 8), dim3(16, 16), dim3(32, 32) }; // Adjust as needed

    // Open CSV file for writing
    std::ofstream outFile("results.csv");
    outFile << "ImageSize,ThreadsPerBlockX,ThreadsPerBlockY,AverageTime(ms)\n";

    for (auto widthHeight : imageSizes) {
        for (auto threadsPerBlock : threadConfigs) {
            std::cout << "Testing image size: " << widthHeight << "x" << widthHeight
                << " with thread configuration: (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")" << std::endl;

            float totalMilliseconds = 0.0f;

            for (int test = 0; test < 10; ++test) {
                std::vector<unsigned char> inputImage, output1, output2;
                generateRandomImage(inputImage, widthHeight, widthHeight);
                output1.resize(4 * widthHeight * widthHeight);
                output2.resize(4 * widthHeight * widthHeight);

                // Allocate memory on device
                unsigned char* dev_input, * dev_output1, * dev_output2;
                cudaMalloc((void**)&dev_input, widthHeight * widthHeight * sizeof(unsigned char));
                cudaMalloc((void**)&dev_output1, 4 * widthHeight * widthHeight * sizeof(unsigned char));
                cudaMalloc((void**)&dev_output2, 4 * widthHeight * widthHeight * sizeof(unsigned char));

                // Copy input data from host to device
                cudaMemcpy(dev_input, inputImage.data(), widthHeight * widthHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);

                // Define kernel launch configuration
                dim3 blocksPerGrid(
                    (widthHeight + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (widthHeight + threadsPerBlock.y - 1) / threadsPerBlock.y
                );

                // Timing CUDA events
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                // Record start event
                cudaEventRecord(start);

                // Launch kernel
                encodeImage << <blocksPerGrid, threadsPerBlock >> > (dev_output1, dev_output2, dev_input, widthHeight, widthHeight);
                cudaError err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("Kernel launch error: ");
                    std::cout << cudaGetErrorString(err) << std::endl;
                }

                // Record stop event
                cudaEventRecord(stop);

                // Synchronize to wait for the stop event
                cudaEventSynchronize(stop);

                // Calculate elapsed time
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);

                // Accumulate time
                totalMilliseconds += milliseconds;

                // Copy results back from device to host
                cudaMemcpy(output1.data(), dev_output1, 4 * widthHeight * widthHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);
                cudaMemcpy(output2.data(), dev_output2, 4 * widthHeight * widthHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);

                // Cleanup
                cudaFree(dev_input);
                cudaFree(dev_output1);
                cudaFree(dev_output2);
            }

            // Calculate average time
            float averageMilliseconds = totalMilliseconds / 10.0f;

            // Save results to CSV
            outFile << widthHeight << "," << threadsPerBlock.x << "," << threadsPerBlock.y << "," << averageMilliseconds << "\n";

            // Output average timing results
            std::cout << "Average time taken: " << averageMilliseconds << " ms" << std::endl;
        }
    }

    outFile.close();
    return 0;
}
