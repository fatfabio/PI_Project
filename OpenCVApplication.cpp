#include "stdafx.h"
#include "common.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <cmath>
#include <chrono>
#include <Windows.h>

using namespace cv;
using namespace std;

wchar_t* projectPath;

//clamp manual
auto clampVal = [](int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
    };

Mat manualNearestNeighbor(Mat& src, double scale) {
    int newWidth = int(src.cols * scale);
    int newHeight = int(src.rows * scale);
    Mat dst(newHeight, newWidth, src.type());

    for (int i = 0; i < newHeight; i++) {
        for (int j = 0; j < newWidth; j++) {
            int a = min(int(j / scale), src.cols - 1);
            int b = min(int(i / scale), src.rows - 1);
            dst.at<Vec3b>(i, j) = src.at<Vec3b>(b, a);
        }
    }
    return dst;
}

Mat manualBilinear(Mat& src, double scale) {
    int newWidth = int(src.cols * scale);
    int newHeight = int(src.rows * scale);
    Mat dst(newHeight, newWidth, src.type());

    for (int i = 0; i < newHeight; i++) {
        for (int j = 0; j < newWidth; j++) {
            float a = j / float(scale);
            float b = i / float(scale);

            int x0 = int(a), y0 = int(b);
            int x1 = min(x0 + 1, src.cols - 1);
            int y1 = min(y0 + 1, src.rows - 1);

            float dx = a - x0, dy = b - y0;

            Vec3b p00 = src.at<Vec3b>(y0, x0);
            Vec3b p01 = src.at<Vec3b>(y0, x1);
            Vec3b p10 = src.at<Vec3b>(y1, x0);
            Vec3b p11 = src.at<Vec3b>(y1, x1);

            Vec3b result;
            for (int c = 0; c < 3; c++) {
                float val =
                    (1 - dx) * (1 - dy) * p00[c] +
                    dx * (1 - dy) * p01[c] +
                    (1 - dx) * dy * p10[c] +
                    dx * dy * p11[c];
                result[c] = uchar(val);
            }
            dst.at<Vec3b>(i, j) = result;
        }
    }
    return dst;
}

static inline float cubicWeight(float x) {
    const float a = -0.5f;
    x = fabsf(x);
    if (x <= 1.0f)
        return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;
    else if (x < 2.0f)
        return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;
    return 0.0f;
}

Mat manualBicubic(Mat& src, double scale) {
    int newW = int(src.cols * scale);
    int newH = int(src.rows * scale);
    Mat dst(newH, newW, src.type());

    for (int dstY = 0; dstY < newH; dstY++) {
        for (int dstX = 0; dstX < newW; dstX++) {
            float gx = dstX / float(scale);
            float gy = dstY / float(scale);
            int xInt = int(floorf(gx));
            int yInt = int(floorf(gy));
            float dx = gx - xInt;
            float dy = gy - yInt;

            Vec3f accum(0, 0, 0);
            float wsum = 0.0f;

            for (int m = -1; m <= 2; m++) {
                for (int n = -1; n <= 2; n++) {
                    int xm = clampVal(xInt + n, 0, src.cols - 1);
                    int ym = clampVal(yInt + m, 0, src.rows - 1);

                    float w = cubicWeight(n - dx) * cubicWeight(m - dy);
                    Vec3b p = src.at<Vec3b>(ym, xm);
                    accum[0] += w * p[0];
                    accum[1] += w * p[1];
                    accum[2] += w * p[2];
                    wsum += w;
                }
            }

            Vec3b result;
            for (int c = 0; c < 3; c++) {
                result[c] = uchar(accum[c] / wsum);
            }
            dst.at<Vec3b>(dstY, dstX) = result;
        }
    }
    return dst;
}

// 4) Downsampling with Filtering
Mat manualDownsampleWithFiltering(Mat& src, double scale) {
    // Aplicam un blur Gaussian pentru a reduce aliasing-ul
    // marimea kernel-ului se bazeaza pe factorul de downsampling
    int k = static_cast<int>(floor(1.0 / scale));
    if (k % 2 == 0) k++;             // kernel impar
    if (k < 3) k = 3;                // cel putin 3x3
    Mat blurred;
    GaussianBlur(src, blurred, Size(k, k), 0);
    // Apoi reducem cu nearest-neighbor
    return manualNearestNeighbor(blurred, scale);
}



void createOutputFolder(const std::string& path) {
    CreateDirectoryA(path.c_str(), NULL);
}


double getPSNR(const Mat& I1, const Mat& I2) {
    Mat s1;
    // calculeaza diferenta absoluta intre cele doua imagini
    absdiff(I1, I2, s1);
    // converteste imaginea la tip float pentru calcule precise
    s1.convertTo(s1, CV_32F);
    // ridica la patrat fiecare diferenta
    s1 = s1.mul(s1);

    // calculeaza suma valorilor pe fiecare canal
    Scalar s = sum(s1);
    double sse = s[0] + s[1] + s[2];

    // daca suma este foarte mica, imaginile sunt identice
    if (sse <= 1e-10) return INFINITY;

    // calculeaza eroarea patratica medie
    double mse = sse / (double)(I1.channels() * I1.total());
    // calculeaza psnr pe baza mse
    return 10.0 * log10((255 * 255) / mse);
}

double getSSIM(const Mat& img1, const Mat& img2) {
    Mat I1, I2;
    //grayscale
    cvtColor(img1, I1, COLOR_BGR2GRAY);
    cvtColor(img2, I2, COLOR_BGR2GRAY);

    Mat I1f, I2f;
    //float
    I1.convertTo(I1f, CV_32F);
    I2.convertTo(I2f, CV_32F);

    Mat mu1, mu2;
    // aplica blur gaussian pentru a obtine medii locale
    GaussianBlur(I1f, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2f, mu2, Size(11, 11), 1.5);

    // calculeaza patratul mediilor si produsul mediilor
    Mat mu1_sq = mu1.mul(mu1);
    Mat mu2_sq = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);

    // calculeaza variantele si covarianta
    Mat sigma1_sq, sigma2_sq, sigma12;
    GaussianBlur(I1f.mul(I1f), sigma1_sq, Size(11, 11), 1.5);
    sigma1_sq -= mu1_sq;

    GaussianBlur(I2f.mul(I2f), sigma2_sq, Size(11, 11), 1.5);
    sigma2_sq -= mu2_sq;

    GaussianBlur(I1f.mul(I2f), sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    // constante pentru stabilizarea fractiei
    const float C1 = 6.5025f, C2 = 58.5225f;

    // calculeaza ssim map pe fiecare pixel
    Mat num = (2 * mu1_mu2 + C1).mul(2 * sigma12 + C2);
    Mat den = (mu1_sq + mu2_sq + C1).mul(sigma1_sq + sigma2_sq + C2);
    Mat ssim_map;
    divide(num, den, ssim_map);

    // media pe intreaga imagine
    Scalar mssim = mean(ssim_map);
    return mssim[0];
}



void testInterpolationAlgorithms(Mat& src, double scale, const std::string& outDir) {
    createOutputFolder(outDir); 

    
    auto runAndSave = [&](const std::string& name, Mat(*func)(Mat&, double), int interp) {
        auto start = chrono::high_resolution_clock::now(); // inceput cronometrare functie manuala
        Mat result = func(src, scale); // executa functia manuala
        auto end = chrono::high_resolution_clock::now(); // final cronometrare
        double duration = chrono::duration<double, milli>(end - start).count(); // calculeaza durata
      //  imwrite(outDir + "\\" + name + "_manual.jpg", result); // salvare img
        cout << name << " (manual): " << duration << " ms\n"; //afiseaza durata

        start = chrono::high_resolution_clock::now(); // cronometrare resize OpenCV
        Mat ref;
        resize(src, ref, Size(), scale, scale, interp); // resize cu OpenCV
        end = chrono::high_resolution_clock::now();
        duration = chrono::duration<double, milli>(end - start).count(); // durata OpenCV
        imwrite(outDir + "\\" + name + "_opencv.jpg", ref); // salveaza imaginea OpenCV
        cout << name << " (OpenCV): " << duration << " ms\n"; // afiseaza durata

        if (result.size() != ref.size()) {
            cout << "  [!] Ajustare dimensiune automata pentru PSNR/SSIM\n";
            resize(result, result, ref.size()); //dimensiunea
        }

        if (result.type() != ref.type()) {
            cout << "  [!] Ajustare tip automata pentru PSNR/SSIM\n";
            result.convertTo(result, ref.type()); //tipul
        }

        double psnrVal = getPSNR(result, ref); //PSNR intre imaginile manuala si OpenCV
        double ssimVal = getSSIM(result, ref); //SSIM intre imaginile manuala si OpenCV
        cout << "  \u2794 PSNR: " << psnrVal << " dB\n";
        cout << "  \u2794 SSIM: " << ssimVal << "\n\n"; // afiseaza valorile de calitate
        };

    runAndSave("nearest", manualNearestNeighbor, INTER_NEAREST); // testeaza metoda nearest
    runAndSave("bilinear", manualBilinear, INTER_LINEAR); // testeaza metoda biliniara
    runAndSave("bicubic", manualBicubic, INTER_CUBIC); // testeaza metoda bicubica

    if (scale < 1.0) { // daca se face micsorare
        auto start = chrono::high_resolution_clock::now();
        Mat down = manualDownsampleWithFiltering(src, scale); // executa metoda cu filtrare
        auto end = chrono::high_resolution_clock::now();
        double duration = chrono::duration<double, milli>(end - start).count();
      //  imwrite(outDir + "\\downsample_manual.jpg", down); // salveaza imaginea man
        cout << "downsample+filter (manual): " << duration << " ms\n";

        start = chrono::high_resolution_clock::now();
        Mat ref;
        resize(src, ref, Size(), scale, scale, INTER_LINEAR); // resize OpenCV pentru downsample
        end = chrono::high_resolution_clock::now();
        duration = chrono::duration<double, milli>(end - start).count();
      //  imwrite(outDir + "\\downsample_opencv.jpg", ref); // salveaza imaginea OpenCV downsample
        cout << "downsample+filter (OpenCV): " << duration << " ms\n";

        if (down.size() != ref.size())
            resize(down, down, ref.size()); // ajusteaza dimensiunea daca este nevoie
        if (down.type() != ref.type())
            down.convertTo(down, ref.type()); // ajusteaza tipul daca este nevoie

        double psnrVal = getPSNR(down, ref); 
        double ssimVal = getSSIM(down, ref); 
        cout << "  \u2794 PSNR: " << psnrVal << " dB\n";
        cout << "  \u2794 SSIM: " << ssimVal << "\n";
    }
}

int main() {
    // dezactiveaza log-urile opencv
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);

    // salveaza calea curenta
    projectPath = _wgetcwd(nullptr, 0);

    // deschide fereastra de selectie a fisierului
    char fname[MAX_PATH];
    if (openFileDlg(fname)) {
        // incarca imaginea
        Mat src = imread(fname);
        if (src.empty()) {
            cout << "eroare la incarcarea imaginii.\n";
            return -1;
        }

        // seteaza factorul de scalare
        double scale = 0.5;

        // apeleaza functia de testare
        testInterpolationAlgorithms(src, scale, "results");
    }

    return 0;
}

/*
    #include "stdafx.h"
#include "common.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <cmath>

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

            // Gasim cei 4 pixeli0
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

// Kernel cubic Catmull–Rom (a = -0.5)
static inline float cubicWeight(float x) {
    const float a = -0.5f;
    x = std::fabs(x);
    if (x <= 1.0f) {
        return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;
    }
    else if (x < 2.0f) {
        return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;
    }
    return 0.0f;
}

Mat manualBicubic(Mat& src, double scale) {
    int newW = (int)(src.cols * scale);
    int newH = (int)(src.rows * scale);
    Mat dst(newH, newW, src.type());

    for (int dstY = 0; dstY < newH; dstY++) {
        for (int dstX = 0; dstX < newW; dstX++) {
            float gx = dstX / scale;
            float gy = dstY / scale;
            int xInt = (int)(floor(gx));
            int yInt = (int)(floor(gy));
            float dx = gx - xInt;
            float dy = gy - yInt;

            Vec3f accum(0, 0, 0);
            float wsum = 0.0f;

            for (int m = -1; m <= 2; m++) {
                for (int n = -1; n <= 2; n++) {
                    int xm = xInt + n;
                    if (xm < 0) xm = 0;
                    else if (xm >= src.cols) xm = src.cols - 1;

                    int ym = yInt + m;
                    if (ym < 0) ym = 0;
                    else if (ym >= src.rows) ym = src.rows - 1;

                    float w = cubicWeight(n - dx) * cubicWeight(m - dy);
                    Vec3b p = src.at<Vec3b>(ym, xm);
                    accum[0] += w * p[0];
                    accum[1] += w * p[1];
                    accum[2] += w * p[2];
                    wsum += w;
                }
            }

            Vec3b result;
            for (int c = 0; c < 3; c++) {
                result[c] = (uchar)(accum[c] / wsum);
            }
            dst.at<Vec3b>(dstY, dstX) = result;
        }
    }
    return dst;
}

// 4) Downsampling with Filtering
Mat manualDownsampleWithFiltering(Mat& src, double scale) {
    // Aplicam un blur Gaussian pentru a reduce aliasing-ul
    // marimea kernel-ului se bazeaza pe factorul de downsampling
    int k = static_cast<int>(floor(1.0 / scale));
    if (k % 2 == 0) k++;             // kernel impar
    if (k < 3) k = 3;                // cel putin 3x3
    Mat blurred;
    GaussianBlur(src, blurred, Size(k, k), 0);
    // Apoi reducem cu nearest-neighbor
    return manualNearestNeighbor(blurred, scale);
}

void zoomShrinkImage() {
    char fname[MAX_PATH];
    while (openFileDlg(fname)) {
        Mat src = imread(fname);
        if (src.empty()) {
            std::cout << "Eroare la încărcare :(\n";
            continue;
        }


        imshow("Imagine Originala", src);
        double scale = 1.0;
        const double step = 0.2;

        while (true) {
            int key = cv::waitKey(0);
            if (key == 'q') exit(0);
            if (key == 'a') scale += step;
            if (key == 's') scale = max(0.1, scale - step);

            imshow("Nearest Neighbor", manualNearestNeighbor(src, scale));
            imshow("Bilinear", manualBilinear(src, scale));
            imshow("Bicubic", manualBicubic(src, scale));
            imshow("Downsample+Filter",
                manualDownsampleWithFiltering(src, scale));
        }
    }
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(nullptr, 0);
    zoomShrinkImage();
    return 0;
}
*/