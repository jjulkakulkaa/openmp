#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <random>
#include <fstream>


// ENCODING

void encodeSegment(const cv::Mat& inputImage, cv::Mat& outputImage1, cv::Mat& outputImage2, int startRow, int endRow, int num_threads) {
    int cols = inputImage.cols;

    std::random_device rd;
    std::mt19937 generator(rd() ^ omp_get_thread_num());
    std::uniform_real_distribution<float> distribution(0.0, 1.0);


    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < cols; ++j) {
            int baseRow = i * 2;
            int baseCol = j * 2;

            float random = distribution(generator);

            if (inputImage.at<uchar>(i, j) == 0) {
                if (random < 0.333f) {
                    random = distribution(generator);
                    if (random < 0.5f) {
                        outputImage1.at<uchar>(baseRow, baseCol) = 0;
                        outputImage1.at<uchar>(baseRow, baseCol + 1) = 0;
                        outputImage1.at<uchar>(baseRow + 1, baseCol) = 255;
                        outputImage1.at<uchar>(baseRow + 1, baseCol + 1) = 255;

                        outputImage2.at<uchar>(baseRow, baseCol) = 255;
                        outputImage2.at<uchar>(baseRow, baseCol + 1) = 255;
                        outputImage2.at<uchar>(baseRow + 1, baseCol) = 0;
                        outputImage2.at<uchar>(baseRow + 1, baseCol + 1) = 0;
                    }
                    else {
                        outputImage2.at<uchar>(baseRow, baseCol) = 0;
                        outputImage2.at<uchar>(baseRow, baseCol + 1) = 0;
                        outputImage2.at<uchar>(baseRow + 1, baseCol) = 255;
                        outputImage2.at<uchar>(baseRow + 1, baseCol + 1) = 255;

                        outputImage1.at<uchar>(baseRow, baseCol) = 255;
                        outputImage1.at<uchar>(baseRow, baseCol + 1) = 255;
                        outputImage1.at<uchar>(baseRow + 1, baseCol) = 0;
                        outputImage1.at<uchar>(baseRow + 1, baseCol + 1) = 0;
                    }
                }
                else if (random < 0.666f) {

                    random = distribution(generator);
                    if (random < 0.5f) {
                        outputImage1.at<uchar>(baseRow, baseCol) = 0;
                        outputImage1.at<uchar>(baseRow, baseCol + 1) = 255;
                        outputImage1.at<uchar>(baseRow + 1, baseCol) = 0;
                        outputImage1.at<uchar>(baseRow + 1, baseCol + 1) = 255;

                        outputImage2.at<uchar>(baseRow, baseCol) = 255;
                        outputImage2.at<uchar>(baseRow, baseCol + 1) = 0;
                        outputImage2.at<uchar>(baseRow + 1, baseCol) = 255;
                        outputImage2.at<uchar>(baseRow + 1, baseCol + 1) = 0;
                    }
                    else {
                        outputImage2.at<uchar>(baseRow, baseCol) = 0;
                        outputImage2.at<uchar>(baseRow, baseCol + 1) = 255;
                        outputImage2.at<uchar>(baseRow + 1, baseCol) = 0;
                        outputImage2.at<uchar>(baseRow + 1, baseCol + 1) = 255;

                        outputImage1.at<uchar>(baseRow, baseCol) = 255;
                        outputImage1.at<uchar>(baseRow, baseCol + 1) = 0;
                        outputImage1.at<uchar>(baseRow + 1, baseCol) = 255;
                        outputImage1.at<uchar>(baseRow + 1, baseCol + 1) = 0;
                    }

                }
                else {
                    random = distribution(generator);
                    if (random < 0.5f) {
                        outputImage1.at<uchar>(baseRow, baseCol) = 0;
                        outputImage1.at<uchar>(baseRow, baseCol + 1) = 255;
                        outputImage1.at<uchar>(baseRow + 1, baseCol) = 255;
                        outputImage1.at<uchar>(baseRow + 1, baseCol + 1) = 0;

                        outputImage2.at<uchar>(baseRow, baseCol) = 255;
                        outputImage2.at<uchar>(baseRow, baseCol + 1) = 0;
                        outputImage2.at<uchar>(baseRow + 1, baseCol) = 0;
                        outputImage2.at<uchar>(baseRow + 1, baseCol + 1) = 255;
                    }
                    else {
                        outputImage2.at<uchar>(baseRow, baseCol) = 0;
                        outputImage2.at<uchar>(baseRow, baseCol + 1) = 255;
                        outputImage2.at<uchar>(baseRow + 1, baseCol) = 255;
                        outputImage2.at<uchar>(baseRow + 1, baseCol + 1) = 0;

                        outputImage1.at<uchar>(baseRow, baseCol) = 255;
                        outputImage1.at<uchar>(baseRow, baseCol + 1) = 0;
                        outputImage1.at<uchar>(baseRow + 1, baseCol) = 0;
                        outputImage1.at<uchar>(baseRow + 1, baseCol + 1) = 255;
                    }
                }
            }
            // white
            else {
                if (random < 0.166f) {
                    outputImage1.at<uchar>(baseRow, baseCol) = 0;
                    outputImage1.at<uchar>(baseRow, baseCol + 1) = 0;
                    outputImage1.at<uchar>(baseRow + 1, baseCol) = 255;
                    outputImage1.at<uchar>(baseRow + 1, baseCol + 1) = 255;

                    outputImage2.at<uchar>(baseRow, baseCol) = 0;
                    outputImage2.at<uchar>(baseRow, baseCol + 1) = 0;
                    outputImage2.at<uchar>(baseRow + 1, baseCol) = 255;
                    outputImage2.at<uchar>(baseRow + 1, baseCol + 1) = 255;
                }
                if (random < 0.333f) {
                    outputImage1.at<uchar>(baseRow, baseCol) = 255;
                    outputImage1.at<uchar>(baseRow, baseCol + 1) = 255;
                    outputImage1.at<uchar>(baseRow + 1, baseCol) = 0;
                    outputImage1.at<uchar>(baseRow + 1, baseCol + 1) = 0;

                    outputImage2.at<uchar>(baseRow, baseCol) = 255;
                    outputImage2.at<uchar>(baseRow, baseCol + 1) = 255;
                    outputImage2.at<uchar>(baseRow + 1, baseCol) = 0;
                    outputImage2.at<uchar>(baseRow + 1, baseCol + 1) = 0;
                }
                else if (random < 0.499f) {
                    outputImage1.at<uchar>(baseRow, baseCol) = 0;
                    outputImage1.at<uchar>(baseRow, baseCol + 1) = 255;
                    outputImage1.at<uchar>(baseRow + 1, baseCol) = 0;
                    outputImage1.at<uchar>(baseRow + 1, baseCol + 1) = 255;

                    outputImage2.at<uchar>(baseRow, baseCol) = 0;
                    outputImage2.at<uchar>(baseRow, baseCol + 1) = 255;
                    outputImage2.at<uchar>(baseRow + 1, baseCol) = 0;
                    outputImage2.at<uchar>(baseRow + 1, baseCol + 1) = 255;
                }
                else if (random < 0.666f) {
                    outputImage1.at<uchar>(baseRow, baseCol) = 255;
                    outputImage1.at<uchar>(baseRow, baseCol + 1) = 0;
                    outputImage1.at<uchar>(baseRow + 1, baseCol) = 255;
                    outputImage1.at<uchar>(baseRow + 1, baseCol + 1) = 0;

                    outputImage2.at<uchar>(baseRow, baseCol) = 255;
                    outputImage2.at<uchar>(baseRow, baseCol + 1) = 0;
                    outputImage2.at<uchar>(baseRow + 1, baseCol) = 255;
                    outputImage2.at<uchar>(baseRow + 1, baseCol + 1) = 0;
                }
                else if (random < 0.833f) {
                    outputImage1.at<uchar>(baseRow, baseCol) = 255;
                    outputImage1.at<uchar>(baseRow, baseCol + 1) = 0;
                    outputImage1.at<uchar>(baseRow + 1, baseCol) = 0;
                    outputImage1.at<uchar>(baseRow + 1, baseCol + 1) = 255;

                    outputImage2.at<uchar>(baseRow, baseCol) = 255;
                    outputImage2.at<uchar>(baseRow, baseCol + 1) = 0;
                    outputImage2.at<uchar>(baseRow + 1, baseCol) = 0;
                    outputImage2.at<uchar>(baseRow + 1, baseCol + 1) = 255;
                }

                else {
                    outputImage1.at<uchar>(baseRow, baseCol) = 0;
                    outputImage1.at<uchar>(baseRow, baseCol + 1) = 255;
                    outputImage1.at<uchar>(baseRow + 1, baseCol) = 255;
                    outputImage1.at<uchar>(baseRow + 1, baseCol + 1) = 0;

                    outputImage2.at<uchar>(baseRow, baseCol) = 0;
                    outputImage2.at<uchar>(baseRow, baseCol + 1) = 255;
                    outputImage2.at<uchar>(baseRow + 1, baseCol) = 255;
                    outputImage2.at<uchar>(baseRow + 1, baseCol + 1) = 0;
                }
            }
        }
    }
}

void encodeImage(const cv::Mat& inputImage, int num_threads) {
    int rows = inputImage.rows;
    int cols = inputImage.cols;

    // outputs
    cv::Mat outputImage1 = cv::Mat(rows * 2, cols * 2, CV_8UC1, cv::Scalar(255));
    cv::Mat outputImage2 = cv::Mat(rows * 2, cols * 2, CV_8UC1, cv::Scalar(255));

    // parallelize 
#pragma omp parallel num_threads(num_threads)
    {

        int tid = omp_get_thread_num();
        int numThreads = omp_get_num_threads();
        int chunkSize = rows / numThreads;
        int startRow = tid * chunkSize;
        int endRow = (tid == numThreads - 1) ? rows : startRow + chunkSize;

        encodeSegment(inputImage, outputImage1, outputImage2, startRow, endRow, num_threads);
    }
}


// DECODING
void decodeImage(const cv::Mat& encodedImage1, const cv::Mat& encodedImage2, cv::Mat& decodedImage) {
    int rows = encodedImage1.rows / 2;
    int cols = encodedImage1.cols / 2;

    decodedImage = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(255));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int baseRow = i * 2;
            int baseCol = j * 2;

            // Check the patterns in both images and reconstruct the original pixel
            if ((encodedImage1.at<uchar>(baseRow, baseCol) == 0 && encodedImage2.at<uchar>(baseRow, baseCol) == 255) ||
                (encodedImage1.at<uchar>(baseRow, baseCol + 1) == 0 && encodedImage2.at<uchar>(baseRow, baseCol + 1) == 255) ||
                (encodedImage1.at<uchar>(baseRow + 1, baseCol) == 0 && encodedImage2.at<uchar>(baseRow + 1, baseCol) == 255) ||
                (encodedImage1.at<uchar>(baseRow + 1, baseCol + 1) == 0 && encodedImage2.at<uchar>(baseRow + 1, baseCol + 1) == 255)) {
                decodedImage.at<uchar>(i, j) = 0; // Black pixel
            }
            else {
                decodedImage.at<uchar>(i, j) = 255; // White pixel
            }
        }
    }
}

void decodeImageWithNoise(const cv::Mat& encodedImage1, const cv::Mat& encodedImage2, cv::Mat& decodedImage) {
    int rows = encodedImage1.rows;
    int cols = encodedImage1.cols;

    // Initialize the decoded image to be the same size as the encoded images
    decodedImage = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(255));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Combine the pixel values from both encoded images
            int pixelValue1 = encodedImage1.at<uchar>(i, j);
            int pixelValue2 = encodedImage2.at<uchar>(i, j);

            // Calculate the combined pixel value
            int combinedPixelValue = (pixelValue1 / 2) + (pixelValue2 / 2);

            // Set the pixel value in the decoded image
            decodedImage.at<uchar>(i, j) = combinedPixelValue;
        }
    }
}



// Function to generate a random binary image of given size
cv::Mat generate_random_image(int rows, int cols) {
    cv::Mat img(rows, cols, CV_8UC1);
    cv::randu(img, cv::Scalar(0), cv::Scalar(256));
    cv::threshold(img, img, 128, 255, cv::THRESH_BINARY);
    return img;
}

// TESTS

// time counting
double count_time(int threads, const cv::Mat& inputImage, cv::Mat& outputImage1, cv::Mat& outputImage2) {
    auto start = std::chrono::high_resolution_clock::now();
    encodeImage(inputImage, threads);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count();
}

// saving test results
void save_results(const std::vector<int>& threads, const std::vector<double>& parallel_times, double sequential_time, const std::string& filename) {
    std::ofstream file(filename);
    file << "Threads,ParallelTime,SpeedupAmdahl,SpeedupGustafson\n";
    for (size_t i = 0; i < threads.size(); ++i) {
        double speedup_amdahl = sequential_time / parallel_times[i];
        double f = 0.95; // parallelizable part
        double speedup_gustafson = threads[i] - (threads[i] - 1) * (1 - f);
        file << threads[i] << "," << parallel_times[i] << "," << speedup_amdahl << "," << speedup_gustafson << "\n";
    }
}

void time_test(const std::vector<cv::Size>& image_sizes) {
    std::vector<int> threads = { 1, 2, 4, 6, 8, 12 };

    for (const auto& size : image_sizes) {
        cv::Mat inputImage = generate_random_image(size.height, size.width);
        cv::Mat outputImage1, outputImage2;

        std::vector<double> parallel_times;


        for (int t : threads) {
            double mes_time = 0;
            for (int i = 0; i < 10; i++) {
                mes_time += count_time(t, inputImage, outputImage1, outputImage2);

            }
            double parallel_time = mes_time / 10;

            parallel_times.push_back(parallel_time);
            std::cout << "Parallel time with " << t << " threads for " << size << ": " << parallel_time << " seconds\n";
        }

        std::string filename = "results_" + std::to_string(size.width) + "x" + std::to_string(size.height) + ".csv";
        save_results(threads, parallel_times, parallel_times[0], filename);
    }
}

int main() {
    cv::Mat inputImage = cv::imread("input_image.png", cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Cannot read img!" << std::endl;
        return -1;
    }

    std::vector<cv::Size> image_sizes = {
    cv::Size(256, 256),
    cv::Size(512, 512),
    cv::Size(1024, 1024),
    cv::Size(2048, 2048),
    cv::Size(4096, 4096),
    cv::Size(8192, 8192)
    };
    //encodeImage(inputImage, 1);
    time_test(image_sizes);

    /*
       
    cv::Mat encodedImage1 = cv::imread("encoded_image1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat encodedImage2 = cv::imread("encoded_image2.png", cv::IMREAD_GRAYSCALE);

    if (encodedImage1.empty() || encodedImage2.empty()) {
        std::cerr << "Cannot read encoded images!" << std::endl;
        return -1;
    }

    cv::Mat decodedImage;


    decodeImageWithNoise(encodedImage1, encodedImage2, decodedImage);


    cv::imwrite("decoded_image.png", decodedImage);

    std::cout << "Reconstructed the original image" << std::endl;*/
    return 0;
}