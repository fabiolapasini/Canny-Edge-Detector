// OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// std:
#include <Math.h>
#include <fstream>
#include <iostream>
#include <string>

#include "include/utils.h"

// using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void maxPooling(const cv::Mat& image, int size, int stride, cv::Mat& out) {
  int newSizeCols = std::floor(((image.cols + 2 * 0 - size) / stride) + 1);
  int newSizeRows = std::floor(((image.cols + 2 * 0 - size) / stride) + 1);
  std::cout << std::endl << "New Col:" << newSizeCols << std::endl;
  std::cout << "New Row:" << newSizeRows << std::endl;

  int max = 0;
  out = cv::Mat(cv::Size(newSizeCols, newSizeRows), CV_8UC1, cv::Scalar(0));

  for (int v = 0; v <= image.rows - size; v += stride) {
    for (int u = 0; u <= image.cols - size; u += stride) {
      max = 0;

      for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
          if (image.data[(u + j + (v + i) * image.cols)] > max) {
            max = image.data[(u + j + (v + i) * image.cols)];
          }
        }
      }
      out.data[((u / stride) + (v / stride) * out.cols)] = max;
    }
  }
}

void averagePooling(const cv::Mat& image, int size, int stride, cv::Mat& out) {
  int newSizeCols = std::floor(((image.cols + 2 * 0 - size) / stride) + 1);
  int newSizeRows = std::floor(((image.cols + 2 * 0 - size) / stride) + 1);

  out = cv::Mat(cv::Size(newSizeCols, newSizeRows), CV_8UC1, cv::Scalar(0));
  float average = 0.0f;

  for (int v = 0; v <= image.rows - size; v += stride) {
    for (int u = 0; u <= image.cols - size; u += stride) {
      average = 0.0f;

      for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
          average += image.data[(u + j + (v + i) * image.cols)];
        }
      }
      average /= pow(size, 2);
      out.data[((u / stride) + (v / stride) * out.cols)] = average;
    }
  }
}

void convFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out,
               int stride) {
  int newSizeCols =
      std::floor(((image.cols + 2 * 0 - kernel.cols) / stride) + 1);
  int newSizeRows =
      std::floor(((image.cols + 2 * 0 - kernel.rows) / stride) + 1);

  out = cv::Mat(cv::Size(newSizeCols, newSizeRows), CV_32FC1, cv::Scalar(0));

  int offsetR = std::floor(kernel.rows) / 2;
  int offsetC = std::floor(kernel.cols) / 2;
  int pixelIm;
  float pixelElem;
  float sum = 0.0f;
  float* fker = (float*)kernel.data;
  float* fim = (float*)out.data;

  for (int v = 0; v < out.rows; v++) {
    for (int u = 0; u < out.cols; u++) {
      sum = 0.0f;
      for (int i = -offsetR; i <= offsetR; i++) {
        for (int j = -offsetC; j <= offsetC; j++) {
          pixelIm =
              image.data[((u * stride) + j + ((v * stride) + i) * image.cols)];
          pixelElem = fker[(offsetC + j + (offsetR + i) * kernel.cols)];
          sum += (pixelElem * pixelIm);
        }
      }
      fim[u + v * out.cols] = sum / 255.0f;
    }
  }
}

void convInt(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out,
             int stride) {
  cv::Mat support;
  convFloat(image, kernel, support, stride);
  out = cv::Mat(support.rows, support.cols, CV_8U);

  float* fim = (float*)support.data;
  float floatVal;
  int intVal;

  for (int v = 0; v < out.rows; v++) {
    for (int u = 0; u < out.cols; u++) {
      floatVal = fim[u + v * out.cols];
      intVal = floatVal * 255;

      if (intVal > 255) intVal = 255;
      if (intVal < 0) intVal = 0;

      out.data[u + v * out.cols] = intVal;
    }
  }
}

void gaussianKernel(float sigma, int radius, cv::Mat& kernel) {
  kernel = cv::Mat(1, 2 * radius + 1, CV_32F);
  float* fker = (float*)kernel.data;
  float g;
  float sum = 0;

  for (int c = 0; c < kernel.cols; c++) {
    g = (1 / (2 * M_PI * powf(sigma, 2))) *
        exp(-powf(c, 2) / (2 * powf(sigma, 2)));
    fker[c] = g;
    sum += g;
  }

  for (int c = 0; c < kernel.cols; c++) {
    fker[c] /= sum;
  }
}

void sobelFilter(const cv::Mat& image, cv::Mat& magnitude, cv::Mat& orientation) {
  cv::Mat Gx, Gy;
  magnitude = cv::Mat(cv::Size(image.cols, image.rows), CV_32F);
  orientation = cv::Mat(cv::Size(image.cols, image.rows), CV_32F);

  float vet1[9] = {1.0f, 0.0f, -1.0f, 2.0f, 0.0f, -2.0f, 1.0f, 0.0f, -1.0f};
  cv::Mat Y = cv::Mat(3, 3, CV_32F, vet1);

  // perform convolution btw img and X (Y), then save the result in Gx (Gy)
  convFloat(image, Y.t(), Gx, 1);
  convFloat(image, Y, Gy, 1);

  // iterate throught elem per elem, get the absolute value and save it in the magnitude
  float* elemX = (float*)Gx.data;
  float* elemY = (float*)Gy.data;

  float max = 0;
  float min = 0;
  float max1 = 0;
  float min1 = 0;

  float mag = 0.0f;
  float ori = 0.0f;
  float pixelX, pixelY;

  for (int v = 0; v < Gx.rows; v++) {
    mag = 0.0f;
    ori = 0.0f;

    for (int u = 0; u < Gx.cols; u++) {
      pixelX = elemX[u + v * Gx.cols];
      pixelY = elemY[u + v * Gy.cols];

      mag = sqrtf(powf(pixelX, 2) + powf(pixelY, 2));
      if (mag >= max1) max1 = mag;
      if (mag <= min1) min1 = mag;
      magnitude.at<float>(v, u) = mag;

      ori = atan2f(pixelY, pixelX);
      if (ori >= max) max = ori;
      if (ori <= min) min = ori;
      orientation.at<float>(v, u) = ori;
    }
  }

  for (int v = 0; v < orientation.rows; v++) {
    for (int u = 0; u < orientation.cols; u++) {
      orientation.at<float>(v, u) =
          2 * M_PI * (orientation.at<float>(v, u) - min) / (max - min);
    }
  }

  for (int v = 0; v < magnitude.rows; v++) {
    for (int u = 0; u < magnitude.cols; u++) {
      magnitude.at<float>(v, u) =
          1 * (magnitude.at<float>(v, u) - min1) / (max1 - min1);
    }
  }
}

float bilinearInterpolation(const cv::Mat& image, float r, float c) {
  float bilinearInterpolation = 0.0f;
  int cF = (int)floor(c);
  int rF = (int)floor(r);
  float s = r - rF;
  float t = c - cF;

  if (cF < 0 || rF < 0 || cF + 1 >= image.cols || rF + 1 >= image.rows) {
    return 0.0f;
  }
  if (image.type() == CV_8UC1) {
    float f00 = image.data[cF + rF * image.cols];
    float f10 = image.data[cF + (rF + 1) * image.cols];
    float f01 = image.data[(cF + 1) + rF * image.cols];
    float f11 = image.data[(cF + 1) + (rF + 1) * image.cols];
    bilinearInterpolation = ((1 - t) * (1 - s) * f00 + s * (1 - t) * f10 +
                             (1 - s) * t * f01 + s * t * f11);
  } else if (image.type() == CV_32F) {
    float f00 = image.at<float>(rF, cF);
    float f10 = image.at<float>(rF, cF + 1);
    float f01 = image.at<float>(rF + 1, cF);
    float f11 = image.at<float>(rF + 1, cF + 1);
    bilinearInterpolation = (1 - t) * (1 - s) * f00 + s * (1 - t) * f10 +
                            (1 - s) * t * f01 + s * t * f11;
  }
  return bilinearInterpolation;
}

int findPeaks(const cv::Mat& magnitude, const cv::Mat& orientation,
              cv::Mat& out, float th0) {
  out = cv::Mat(cv::Size(magnitude.cols, magnitude.rows), CV_32F);
  float angle, pixel, e1x, e1y, e2x, e2y, e1, e2;

  for (int r = 0; r < magnitude.rows; r++) {
    for (int c = 0; c < magnitude.cols; c++) {
      pixel = magnitude.at<float>(r, c);
      angle = orientation.at<float>(r, c);

      e1x = c + 1 * cosf(angle);
      e1y = r + 1 * sinf(angle);
      e2x = c - 1 * cosf(angle);
      e2y = r - 1 * sinf(angle);

      e1 = bilinearInterpolation(magnitude, e1y, e1x);
      e2 = bilinearInterpolation(magnitude, e2y, e2x);

      if (pixel >= e1 && pixel >= e2 && pixel >= th0) {
        out.at<float>(r, c) = pixel;
      } else {
        out.at<float>(r, c) = 0;
      }
    }
  }
  return 0;
}

int doubleTh(const cv::Mat& magnitude, cv::Mat& out, float th1, float th2) {
  float val;
  out = cv::Mat(cv::Size(magnitude.cols, magnitude.rows), CV_8UC1);
  for (int v = 0; v < magnitude.rows; v++) {
    for (int u = 0; u < magnitude.cols; u++) {
      val = magnitude.at<float>(v, u);

      if (val > th1) {
        out.data[u + v * magnitude.cols] = 255.0f;
      } else if (val <= th1 && val > th2) {
        out.data[u + v * magnitude.cols] = 128.0f;
      } else if (val <= th2) {
        out.data[u + v * magnitude.cols] = 0.0f;
      }
    }
  }
  return 0;
}

int cannyEdgeDetector(const cv::Mat& image, cv::Mat& out, float th0, float th1, float th2) {
  cv::Mat magnitude, orientation, support;
  sobelFilter(image, magnitude, orientation);
  findPeaks(magnitude, orientation, support, th0);
  doubleTh(support, out, th1, th2);
  return 0;
}

int runCannyEdgeDetector() {
  int frame_number = 0;
  char frame_name[256];
  bool exit_loop = false;

  cv::Mat image = cv::imread("Lenna.png", CV_8UC1);
  if (image.empty()) {
    std::cout << "Unable to open " << frame_name << std::endl;
    return 1;
  }

  cv::namedWindow("Lenna", cv::WINDOW_NORMAL);
  cv::imshow("Lenna", image);
  cv::waitKey(5000);
  cv::destroyWindow("Lenna");

  int size = 3;
  int stride = 2;
  int sigma = 5;
  int radius = 10;

  float th0 = 0.15f;
  float th1 = 0.3f;
  float th2 = 0.2f;

  float vet[9] = {0.0f, -1.0f, 0.0f, -1.0f, 5.0f, -1.0f, 0.0f, -1.0f, 0.0f};
  cv::Mat kernel = cv::Mat(3, 3, CV_32F, vet);

  float vet3[3] = {0.0f, 1.0f, 0.0f};
  cv::Mat kernel1D = cv::Mat(1, 3, CV_32F, vet3);

  // 1) MAX POOLLING
  cv::Mat first(cv::Size(image.cols, image.rows), CV_8UC1, cv::Scalar(0));
  maxPooling(image, size, stride, first);
  cv::namedWindow("Max Polling");
  cv::imshow("Max Polling", first);
  cv::waitKey(5000);
  cv::destroyWindow("Max Polling");

  // 2) AVERAGE POOLLING
  cv::Mat second(cv::Size(image.cols, image.rows), CV_8UC1, cv::Scalar(0));
  averagePooling(image, size, stride, second);
  cv::namedWindow("Everage Polling");
  cv::imshow("Everage Polling", second);
  cv::waitKey(5000);
  cv::destroyWindow("Everage Polling");

  // 3) FLOAT CONVOLUTION
  cv::Mat third;
  convFloat(image, kernel, third);
  cv::namedWindow("Float convolution");
  cv::imshow("Float convolution", third);
  cv::waitKey(5000);
  cv::destroyWindow("Float convolution");

  // 4) INT CONVOLUTION
  cv::Mat fourth;
  convInt(image, kernel, fourth);
  cv::namedWindow("Int convolution");
  cv::imshow("Int convolution", fourth);
  cv::waitKey(5000);
  cv::destroyWindow("Int convolution");

  // 5) GAUSSIAN KERNEL
  cv::Mat fifth;
  gaussianKernel(sigma, radius, kernel1D);
  convInt(image, kernel1D, fifth, 1);
  cv::namedWindow("Orizontal gaussian Kernel");
  cv::imshow("Orizontal gaussian Kernel", fifth);
  cv::waitKey(5000);
  cv::destroyWindow("Orizontal gaussian Kernel");

  // 6) VARIUS FILTERS (need number 5))
  cv::Mat sixtB;
  convInt(image, kernel1D.t(), sixtB, 1);
  cv::namedWindow("Vertical gaussian Kernel");
  cv::imshow("Vertical gaussian Kernel", sixtB);
  cv::waitKey(5000);
  cv::destroyWindow("Vertical gaussian Kernel");

  cv::Mat sixt;
  convInt(fifth, kernel1D.t(), sixt, 1);
  cv::namedWindow("Bidimensional gaussian Kernel");
  cv::imshow("Bidimensional gaussian Kernel", sixt);
  cv::waitKey(5000);
  cv::destroyWindow("Bidimensional gaussian Kernel");

  // 7) MAGNITUDE & ORIENTATION
  cv::Mat magnitude, orientation, Gx, Gy;
  sobelFilter(image, magnitude, orientation);
  cv::namedWindow("Magnitude");
  cv::imshow("Magnitude", magnitude);
  cv::waitKey(5000);
  cv::destroyWindow("Magnitude");

  cv::Mat adjMap;
  convertScaleAbs(orientation, adjMap, 255 / (2 * M_PI));
  cv::Mat falseColorsMap;
  applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_AUTUMN);
  cv::namedWindow("Orientation");
  cv::imshow("Orientation", falseColorsMap);
  cv::waitKey(5000);
  cv::destroyWindow("Orientation");

  // 8) BILINEAR INTERPOLATION
  float r = 27.8f;
  float c = 11.4f;
  float eight = bilinearInterpolation(image, r, c);
  std::cout << "Bilinear interpolation between: " << r << " and " << c << " = "
       << eight << std::endl;

  // 9) FIND PEAKS
  cv::Mat ninth(cv::Size(image.cols, image.rows), CV_32F);
  findPeaks(magnitude, orientation, ninth, th0);
  cv::namedWindow("Non maxima suppression");
  cv::imshow("Non maxima suppression", ninth);
  cv::waitKey(5000);
  cv::destroyWindow("Non maxima suppression");

  // 10) HISTERESIS THRESHOLD
  cv::Mat tenth(cv::Size(image.cols, image.rows), CV_8U);
  doubleTh(magnitude, tenth, th1, th2);
  cv::namedWindow("Histeresis");
  cv::imshow("Histeresis", tenth);
  cv::waitKey(5000);
  cv::destroyWindow("Histeresis");

  // 11) CANNY
  cv::Mat eleventh(cv::Size(image.cols, image.rows), CV_8UC1);
  cannyEdgeDetector(image, eleventh, th0, th1, th2);
  cv::namedWindow("Canny");
  cv::imshow("Canny", eleventh);
  unsigned char key = cv::waitKey(0);
  if (key == 'q') cv::destroyWindow("Canny");

  return 0;
}
