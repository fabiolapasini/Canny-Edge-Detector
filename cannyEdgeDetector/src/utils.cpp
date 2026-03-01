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
#include "include/Image.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Canny
{
	Graphics::Image<unsigned char> maxPooling(
		const Graphics::Image<unsigned char>& image, int size, int stride)
	{
		int newSizeCols = std::floor(((image.width() + 2 * 0 - size) / stride) + 1);
		int newSizeRows = std::floor(((image.height() + 2 * 0 - size) / stride) + 1);

		Graphics::Image<unsigned char> out(newSizeCols, newSizeRows, 1);
		std::fill(out.data(), out.data() + out.totalElements(), 0);

		int max = 0;
		for (int v = 0; v <= image.height() - size; v += stride) {
			for (int u = 0; u <= image.width() - size; u += stride) {
				max = 0;
				for (int i = 0; i < size; i++) {
					for (int j = 0; j < size; j++) {
						if (image(u + j, v + i) > max) {
							max = image(u + j, v + i);
						}
					}
				}
				out(u / stride, v / stride) = static_cast<unsigned char>(max);
			}
		}
		return out;
	}

	Graphics::Image<unsigned char> averagePooling(const Graphics::Image<unsigned char>& image, int size, int stride)
	{
		int newSizeCols = std::floor(((image.width() + 2 * 0 - size) / stride) + 1);
		int newSizeRows = std::floor(((image.height() + 2 * 0 - size) / stride) + 1);

		Graphics::Image<unsigned char> out(newSizeCols, newSizeRows, 1);
		std::fill(out.data(), out.data() + out.totalElements(), 0);

		float average = 0.0f;
		for (int v = 0; v <= image.height() - size; v += stride) {
			for (int u = 0; u <= image.width() - size; u += stride) {
				average = 0.0f;
				for (int i = 0; i < size; i++) {
					for (int j = 0; j < size; j++) {
						average += image(u + j, v + i);
					}
				}
				average /= (size * size);
				out(u / stride, v / stride) = static_cast<unsigned char>(average);
			}
		}
		return out;
	}

	Graphics::Image<float> convFloat(const Graphics::Image<unsigned char>& image,
		const Graphics::Image<float>& kernel, int stride)
	{
		int newSizeCols = std::floor(((image.width() + 2 * 0 - kernel.width()) / stride) + 1);
		int newSizeRows = std::floor(((image.height() + 2 * 0 - kernel.height()) / stride) + 1);

		Graphics::Image<float> out(newSizeCols, newSizeRows, 1);
		std::fill(out.data(), out.data() + out.totalElements(), 0);

		int offsetC = std::floor(kernel.width()) / 2;
		int offsetR = std::floor(kernel.height()) / 2;
		int pixelIm;
		float pixelElem;
		float sum = 0.0f;
		const float* fker = kernel.data();

		for (int v = 0; v < out.height(); v++) {
			for (int u = 0; u < out.width(); u++) {
				sum = 0.0f;
				for (int i = -offsetR; i <= offsetR; i++) {
					for (int j = -offsetC; j <= offsetC; j++) {
						pixelIm = image((u * stride) + j, ((v * stride) + i));
						pixelElem = kernel(offsetC + j, offsetR + i);
						sum += (pixelElem * pixelIm);
					}
				}
				out(u,v) = sum / 255.0f;
			}
		}
		return out;
	}


	Graphics::Image<unsigned char> convInt(const Graphics::Image<unsigned char>& image,
		const Graphics::Image<float>& kernel, int stride)
	{
		Graphics::Image<float> support = convFloat(image, kernel, stride);
		Graphics::Image<unsigned char> out(support.width(), support.height(), 1);
		std::fill(out.data(), out.data() + out.totalElements(), 0);

		float floatVal;
		int intVal;
		for (int v = 0; v < out.height(); v++) {
			for (int u = 0; u < out.width(); u++) {
				floatVal = support(u, v);
				intVal = floatVal * 255;
				if (intVal > 255) intVal = 255;
				if (intVal < 0) intVal = 0;
				out(u, v) = static_cast<unsigned char>(intVal);
			}
		}
		return out;
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

		float vet1[9] = { 1.0f, 0.0f, -1.0f, 2.0f, 0.0f, -2.0f, 1.0f, 0.0f, -1.0f };
		cv::Mat Y = cv::Mat(3, 3, CV_32F, vet1);

		// perform convolution btw img and X (Y), then save the result in Gx (Gy)
		// convFloat(image, Y.t(), Gx, 1);
		// convFloat(image, Y, Gy, 1);

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
		}
		else if (image.type() == CV_32F) {
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
				}
				else {
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
				}
				else if (val <= th1 && val > th2) {
					out.data[u + v * magnitude.cols] = 128.0f;
				}
				else if (val <= th2) {
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

}