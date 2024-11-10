// OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// std:
#include <math.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include "include/utils.h"

using namespace cv;
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void maxPooling(const Mat& image, int size, int stride, Mat& out) {
  int newSizeCols = floor(((image.cols + 2 * 0 - size) / stride) + 1);
  int newSizeRows = floor(((image.cols + 2 * 0 - size) / stride) + 1);

  cout << endl << "New Col:" << newSizeCols << endl;
  cout << "New Row:" << newSizeRows << endl;

  int max = 0;

  out = Mat(Size(newSizeCols, newSizeRows), CV_8UC1, Scalar(0));

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

void averagePooling(const Mat& image, int size, int stride, Mat& out) {
  int newSizeCols = floor(((image.cols + 2 * 0 - size) / stride) + 1);
  int newSizeRows = floor(((image.cols + 2 * 0 - size) / stride) + 1);

  out = Mat(Size(newSizeCols, newSizeRows), CV_8UC1, Scalar(0));

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

void convFloat(const Mat& image, const Mat& kernel, Mat& out, int stride = 1) {
  int newSizeCols = floor(((image.cols + 2 * 0 - kernel.cols) / stride) + 1);
  int newSizeRows = floor(((image.cols + 2 * 0 - kernel.rows) / stride) + 1);

  out = Mat(Size(newSizeCols, newSizeRows), CV_32FC1, Scalar(0));

  int offsetR = floor(kernel.rows) / 2;
  int offsetC = floor(kernel.cols) / 2;
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

void convInt(const Mat& image, const Mat& kernel, Mat& out, int stride = 1) {
  Mat support;
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

void gaussianKernel(float sigma, int radius, Mat& kernel) {
  kernel = Mat(1, 2 * radius + 1, CV_32F);
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

void sobel(const Mat& image, Mat& magnitude, Mat& orientation) {
  Mat Gx, Gy;
  magnitude = Mat(Size(image.cols, image.rows), CV_32F);
  orientation = Mat(Size(image.cols, image.rows), CV_32F);

  float vet1[9] = {1.0f, 0.0f, -1.0f, 2.0f, 0.0f, -2.0f, 1.0f, 0.0f, -1.0f};
  Mat Y = Mat(3, 3, CV_32F, vet1);

  // eseguo la convoluzione tra l'immagine e X (Y) e salvo il risulato in Gx
  // (Gy)
  convFloat(image, Y.t(), Gx, 1);
  convFloat(image, Y, Gy, 1);

  // scorro elemento per elemento di Gx e Gy e caclolo il valore assoluto che
  // poi salvo in magnitude
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

float bilinear(const cv::Mat& image, float r, float c) {
  float bilinearInterpolation;

  int cF = (int)floor(c);
  int rF = (int)floor(r);

  float s = r - rF;
  float t = c - cF;

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

int findPeaks(const Mat& magnitude, const Mat& orientation, Mat& out,
              float th0) {
  out = Mat(Size(magnitude.cols, magnitude.rows), CV_32F);
  float angle, pixel, e1x, e1y, e2x, e2y, e1, e2;

  for (int r = 0; r < magnitude.rows; r++) {
    for (int c = 0; c < magnitude.cols; c++) {
      pixel = magnitude.at<float>(r, c);
      angle = orientation.at<float>(r, c);

      e1x = c + 1 * cosf(angle);
      e1y = r + 1 * sinf(angle);
      e2x = c - 1 * cosf(angle);
      e2y = r - 1 * sinf(angle);

      e1 = bilinear(magnitude, e1y, e1x);
      e2 = bilinear(magnitude, e2y, e2x);

      if (pixel >= e1 && pixel >= e2 && pixel >= th0) {
        out.at<float>(r, c) = pixel;
      } else {
        out.at<float>(r, c) = 0;
      }
    }
  }
  return 0;
}

int doubleTh(const Mat& magnitude, Mat& out, float th1, float th2) {
  out = Mat(Size(magnitude.cols, magnitude.rows), CV_8UC1);
  float val;

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

int canny(const Mat& image, Mat& out, float th0, float th1, float th2) {
  Mat magnitude, orientation, support;

  sobel(image, magnitude, orientation);
  findPeaks(magnitude, orientation, support, th0);
  doubleTh(support, out, th1, th2);

  return 0;
}

int runCannyEdgeDetector(ArgumentList args) {
  int frame_number = 0;
  char frame_name[256];
  bool exit_loop = false;

  // // TODO(Fabiola): rivedere questa cosa del parsing...
  //cout << "Simple program." << endl;

  while (!exit_loop) {
  //  // generating file name
  //  // multi frame case
  //  if (args.image_name.find('%') != std::string::npos)
  //    sprintf(frame_name, (const char*)(args.image_name.c_str()), frame_number);
  //  else  // single frame case
  //    sprintf(frame_name, "%s", args.image_name.c_str());

  //  // opening file
  //  // std::cout<<"Opening "<<frame_name<<std::endl;

    Mat image = imread("Lenna.png", CV_8UC1);

    if (image.empty()) {
      std::cout << "Unable to open " << frame_name << std::endl;
      return 1;
    }

    // display image
    // namedWindow("image", cv::WINDOW_NORMAL);
    // imshow("image", image);

    //////////////////////////////////////////////////////////////////////////////

    [[maybe_unused]] int size = 3;
    [[maybe_unused]] int stride = 2;
    [[maybe_unused]] int sigma = 5;
    [[maybe_unused]] int radius = 10;

    [[maybe_unused]] float th0 = 0.15f;
    [[maybe_unused]] float th1 = 0.3f;
    [[maybe_unused]] float th2 = 0.2f;

    float vet[9] = {0.0f, -1.0f, 0.0f, -1.0f, 5.0f, -1.0f, 0.0f, -1.0f, 0.0f};
    Mat kernel = Mat(3, 3, CV_32F, vet);

    float vet3[3] = {0.0f, 1.0f, 0.0f};
    Mat kernel1D = Mat(1, 3, CV_32F, vet3);

    ///////////// CODICE PER PROVARE LE VARIE FUNZIONI SINGOLARMENTE
    ///////////////

    // 1) MAX POOLLING
    /*Mat first(Size(image.cols, image.rows), CV_8UC1, Scalar(0));
    maxPooling(image, size, stride, first);
    namedWindow("Max Polling");
    imshow("Max Polling", first);*/

    // 2) AVERAGE POOLLING
    /*Mat second(Size(image.cols, image.rows), CV_8UC1, Scalar(0));
    averagePooling(image, size, stride, second);
    namedWindow("Everage Polling");
    imshow("Everage Polling", second);*/

    // 3) FLOAT CONVOLUTION
    /*Mat third;
    convFloat(image, kernel, third);
    namedWindow("Float convolution");
    imshow("Float convolution", third);*/

    // 4) INT CONVOLUTION
    /*Mat fourth;
    convInt(image, kernel, fourth);
    namedWindow("Float convolution");
    imshow("Float convolution", fourth);*/

    // 5) GAUSSIAN KERNEL
    /*Mat fifth;
    gaussianKernel(sigma, radius, kernel1D);
    convInt(image, kernel1D, fifth, 1);
    namedWindow("Orizontal gaussian Kernel");
    imshow("Orizontal gaussian Kernel", fifth);*/

    // 6) VARIUS FILTERS (decommentare il 5))
    /*Mat sixtB;
    convInt(image, kernel1D.t(), sixtB, 1);
    namedWindow("Vertical gaussian Kernel");
    imshow("Vertical gaussian Kernel", sixtB);

    Mat sixt;
    convInt(fifth, kernel1D.t(), sixt, 1);
    namedWindow("Bidimensional gaussian Kernel");
    imshow("Bidimensional gaussian Kernel", sixt);*/

    // 7) MAGNITUDE & ORIENTATION
    Mat magnitude, orientation, Gx, Gy;

    sobel(image, magnitude, orientation);

    namedWindow("Magnitude");
    imshow("Magnitude", magnitude);

    Mat adjMap;
    convertScaleAbs(orientation, adjMap, 255 / (2 * M_PI));
    Mat falseColorsMap;
    applyColorMap(adjMap, falseColorsMap, COLORMAP_AUTUMN);
    namedWindow("Orientation");
    imshow("Orientation", falseColorsMap);

    // 8) BILINEAR INTERPOLATION
    /*float r = 27.8f;
    float c = 11.4f;
    float eight = bilinear(image, r, c);
    cout<<"Bilinear interpolation between: " << r <<  " and " << c << " = " <<
    eight << endl;*/

    // 9) FIND PEAKS
    Mat ninth(Size(image.cols, image.rows), CV_32F);
    findPeaks(magnitude, orientation, ninth, th0);

    namedWindow("Non maxima suppression");
    imshow("Non maxima suppression", ninth);

    // 10) HISTERESIS THRESHOLD
    /*Mat tenth(Size(image.cols, image.rows), CV_8U);
    doubleTh(magnitude, tenth, th1, th2);

    namedWindow("Histeresis");
    imshow("Histeresis", tenth);*/

    // 11) CANNY
    // //////////////////////////////////////////////////////////////////

    Mat eleventh(Size(image.cols, image.rows), CV_8UC1);
    canny(image, eleventh, th0, th1, th2);

    namedWindow("Canny");
    imshow("Canny", eleventh);

    //////////////////////////////////////////////////////////////////////////////

    // wait for key or timeout
    unsigned char key = cv::waitKey(args.wait_t);
    std::cout << "key " << int(key) << std::endl;

    // here you can implement some looping logic using key value:
    if (key == 'q') exit_loop = true;

    frame_number++;
  }

  return 0;
}
