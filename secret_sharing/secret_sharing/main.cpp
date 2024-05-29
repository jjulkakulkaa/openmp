#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <cstdlib>
#include <ctime>

// Encode a segment of the image using the pixel division method
void encodeSegment(const cv::Mat& inputImage, cv::Mat& outputImage1, cv::Mat& outputImage2, int startRow, int endRow, int num_threads) {
    int cols = inputImage.cols;

    srand(time(NULL) ^ omp_get_thread_num());

    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < cols; ++j) {
            int baseRow = i * 2;
            int baseCol = j * 2;

            float random = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

            if (inputImage.at<uchar>(i, j) == 0) {
                if (random < 0.333f) {
                    outputImage1.at<uchar>(baseRow, baseCol) = 0;
                    outputImage1.at<uchar>(baseRow, baseCol + 1) = 0;
                    outputImage1.at<uchar>(baseRow + 1, baseCol) = 255;
                    outputImage1.at<uchar>(baseRow + 1, baseCol + 1) = 255;

                    outputImage2.at<uchar>(baseRow, baseCol) = 255;
                    outputImage2.at<uchar>(baseRow, baseCol + 1) = 255;
                    outputImage2.at<uchar>(baseRow + 1, baseCol) = 0;
                    outputImage2.at<uchar>(baseRow + 1, baseCol + 1) = 0;
                }
                else if (random < 0.666f) {
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
                    outputImage1.at<uchar>(baseRow, baseCol) = 0;
                    outputImage1.at<uchar>(baseRow, baseCol + 1) = 255;
                    outputImage1.at<uchar>(baseRow + 1, baseCol) = 255;
                    outputImage1.at<uchar>(baseRow + 1, baseCol + 1) = 0;

                    outputImage2.at<uchar>(baseRow, baseCol) = 255;
                    outputImage2.at<uchar>(baseRow, baseCol + 1) = 0;
                    outputImage2.at<uchar>(baseRow + 1, baseCol) = 0;
                    outputImage2.at<uchar>(baseRow + 1, baseCol + 1) = 255;
                }
            }
            else {
                if (random < 0.33f) {
                    outputImage1.at<uchar>(baseRow, baseCol) = 0;
                    outputImage1.at<uchar>(baseRow, baseCol + 1) = 0;
                    outputImage1.at<uchar>(baseRow + 1, baseCol) = 255;
                    outputImage1.at<uchar>(baseRow + 1, baseCol + 1) = 255;

                    outputImage2.at<uchar>(baseRow, baseCol) = 0;
                    outputImage2.at<uchar>(baseRow, baseCol + 1) = 0;
                    outputImage2.at<uchar>(baseRow + 1, baseCol) = 255;
                    outputImage2.at<uchar>(baseRow + 1, baseCol + 1) = 255;
                }
                else if (random < 0.66f) {
                    outputImage1.at<uchar>(baseRow, baseCol) = 0;
                    outputImage1.at<uchar>(baseRow, baseCol + 1) = 255;
                    outputImage1.at<uchar>(baseRow + 1, baseCol) = 0;
                    outputImage1.at<uchar>(baseRow + 1, baseCol + 1) = 255;

                    outputImage2.at<uchar>(baseRow, baseCol) = 0;
                    outputImage2.at<uchar>(baseRow, baseCol + 1) = 255;
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

void encodeImage(const cv::Mat& inputImage, cv::Mat& outputImage1, cv::Mat& outputImage2, int num_threads) {
    int rows = inputImage.rows;
    int cols = inputImage.cols;

    // Initialize output images
    outputImage1 = cv::Mat(rows * 2, cols * 2, CV_8UC1, cv::Scalar(255));
    outputImage2 = cv::Mat(rows * 2, cols * 2, CV_8UC1, cv::Scalar(255));

    // Parallelize the processing
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

void convertToBlackAndWhite(const std::string& inputImagePath, const std::string& outputImagePath, int thresholdValue = 128) {
    cv::Mat grayscaleImage = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);
    if (grayscaleImage.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return;
    }

    cv::Mat binaryImage;
    cv::threshold(grayscaleImage, binaryImage, thresholdValue, 255, cv::THRESH_BINARY);
    cv::imwrite(outputImagePath, binaryImage);
}

int main() {
    cv::Mat inputImage = cv::imread("input_image.png", cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Nie uda³o siê wczytaæ obrazu!" << std::endl;
        return -1;
    }

    cv::Mat outputImage1;
    cv::Mat outputImage2;

    int num_threads = 4; // Adjust the number of threads as needed

    encodeImage(inputImage, outputImage1, outputImage2, num_threads);

    cv::imwrite("encoded_image1.png", outputImage1);
    cv::imwrite("encoded_image2.png", outputImage2);

    std::cout << "Obraz zosta³ zakodowany i zapisany jako encoded_image.png" << std::endl;

    return 0;
}
