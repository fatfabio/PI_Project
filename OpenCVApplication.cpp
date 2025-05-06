#include "stdafx.h"
#include "common.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <algorithm>

wchar_t* projectPath;

Mat manualNearestNeighbor(Mat& src, double scale) {
    int newWidth = (int)(src.cols * scale);
    int newHeight = (int)(src.rows * scale);
    Mat dst(newHeight, newWidth, src.type());

    for (int i = 0; i < newHeight; i++) {
        for (int j = 0; j < newWidth; j++) {
            int a = min((int)(j / scale), src.cols - 1);
            int b = min((int)(i / scale), src.rows - 1);
            dst.at<Vec3b>(i, j) = src.at<Vec3b>(b, a);
        }
    }
    return dst;
}

Mat manualBilinear(const cv::Mat& src, double scale) {
    int newWidth = static_cast<int>(src.cols * scale);
    int newHeight = static_cast<int>(src.rows * scale);
    Mat dst(newHeight, newWidth, src.type());

    for (int i = 0; i < newHeight; i++) {
        for (int j = 0; j < newWidth; j++) {
            float a = j / scale;
            float b = i / scale;

            // Gasim cei 4 pixeli
            int x0 = (int)a;
            int y0 = (int)b;
            int x1 = min(x0 + 1, src.cols - 1);
            int y1 = min(y0 + 1, src.rows - 1);

            float dx = a - x0;
            float dy = b - y0;

            // Extragem valorile
            Vec3b p00 = src.at<Vec3b>(y0, x0);
            Vec3b p01 = src.at<Vec3b>(y0, x1);
            Vec3b p10 = src.at<Vec3b>(y1, x0);
            Vec3b p11 = src.at<Vec3b>(y1, x1);

            Vec3b result;
            for (int i = 0; i < 3; i++) {
                float val = (1 - dx) * (1 - dy) * p00[i] +
                    dx * (1 - dy) * p01[i] +
                    (1 - dx) * dy * p10[i] +
                    dx * dy * p11[i];
                result[i] = (uchar)val;
            }
            dst.at<Vec3b>(i, j) = result;
        }
    }
    return dst;
}

void zoomShrinkImage() {
    char fname[MAX_PATH];
    while (openFileDlg(fname)) {
        Mat src = imread(fname);
        if (src.empty()) {
            std::cout << "Eroare :(" << std::endl;
            continue;
        }

        cv::imshow("Imagine Originala", src);

        double scale = 1.0;
        const double scaleStep = 0.2;

        while (true) {

            int key = cv::waitKey(0);

            key == 'q' ? exit(0) : (key == 'a' ? scale += scaleStep : scale -= scaleStep);

            Mat dst1 = manualNearestNeighbor(src, scale);
            imshow("Nearest Neighbor", dst1);

            Mat dst2 = manualBilinear(src, scale);
            imshow("Bilinear Interpolation", dst2);
        }
    }
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

    zoomShrinkImage();

    return 0;
}