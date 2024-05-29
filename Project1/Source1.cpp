#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h> 

// Funkcja do zakodowania obrazu metod¹ podzia³u piksela na 4 podpiksele
void encodeImage(const cv::Mat& inputImage, cv::Mat& outputImage1, cv::Mat& outputImage2) {
    int rows = inputImage.rows;
    int cols = inputImage.cols;

    // Tworzymy nowy obraz, który bêdzie mia³ 4 razy wiêcej pikseli (2x w ka¿d¹ stronê)
    outputImage1 = cv::Mat(rows * 2, cols * 2, CV_8UC1, cv::Scalar(255));
    outputImage2 = cv::Mat(rows * 2, cols * 2, CV_8UC1, cv::Scalar(255));

    srand(time(NULL));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Podziel piksel na 4 podpiksele
            int baseRow = i * 2;
            int baseCol = j * 2;


            float random = ((double)rand() / (RAND_MAX));
            printf("%f\n", random);

            if (inputImage.at<uchar>(i, j) == 0) {
                // Jeœli piksel jest czarny
                if (random < 0.333) {
                    outputImage1.at<uchar>(baseRow, baseCol) = 0;
                    outputImage1.at<uchar>(baseRow, baseCol + 1) = 0;
                    outputImage1.at<uchar>(baseRow + 1, baseCol) = 255;
                    outputImage1.at<uchar>(baseRow + 1, baseCol + 1) = 255;

                    outputImage2.at<uchar>(baseRow, baseCol) = 255;
                    outputImage2.at<uchar>(baseRow, baseCol + 1) = 255;
                    outputImage2.at<uchar>(baseRow + 1, baseCol) = 0;
                    outputImage2.at<uchar>(baseRow + 1, baseCol + 1) = 0;
                }
                else if (random < 0.666) {
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
                // Jeœli piksel jest bia³y
                if (random < 0.33) {
                    outputImage1.at<uchar>(baseRow, baseCol) = 0;
                    outputImage1.at<uchar>(baseRow, baseCol + 1) = 0;
                    outputImage1.at<uchar>(baseRow + 1, baseCol) = 255;
                    outputImage1.at<uchar>(baseRow + 1, baseCol + 1) = 255;

                    outputImage2.at<uchar>(baseRow, baseCol) = 0;
                    outputImage2.at<uchar>(baseRow, baseCol + 1) = 0;
                    outputImage2.at<uchar>(baseRow + 1, baseCol) = 255;
                    outputImage2.at<uchar>(baseRow + 1, baseCol + 1) = 255;
                }
                else if (random < 0.66) {
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

void convertToBlackAndWhite(const std::string& inputImagePath, const std::string& outputImagePath, int thresholdValue = 128) {
    // Read the image in grayscale mode



    cv::Mat grayscaleImage = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);
    if (grayscaleImage.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return;
    }

    // Create a binary image using the specified threshold value
    cv::Mat binaryImage;
    cv::threshold(grayscaleImage, binaryImage, thresholdValue, 255, cv::THRESH_BINARY);

    // Save the result
    cv::imwrite(outputImagePath, binaryImage);
}


int main() {

    // Wczytaj czarno-bia³y obraz (zak³adamy, ¿e obraz jest w odcieniach szaroœci)
    cv::Mat inputImage = cv::imread("input_image.png", cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Nie uda³o siê wczytaæ obrazu!" << std::endl;
        return -1;
    }

    cv::Mat outputImage1;
    cv::Mat outputImage2;

    // Zakodowanie obrazu metod¹ podzia³u piksela na 4 podpiksele
    encodeImage(inputImage, outputImage1, outputImage2);

    // Zapisanie zakodowanego obrazu
    cv::imwrite("encoded_image1.png", outputImage1);
    cv::imwrite("encoded_image2.png", outputImage2);

    std::cout << "Obraz zosta³ zakodowany i zapisany jako encoded_image.png" << std::endl;

    return 0;
}
