#pragma once

#include <filesystem>
#include "Image.h"

namespace Canny 
{
	void maxPooling(const cv::Mat& image, int size, int stride, cv::Mat& out);

	void averagePooling(const cv::Mat& image, int size, int stride, cv::Mat& out);

	void convFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out,
		int stride = 1);

	void convInt(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out,
		int stride = 1);

	void gaussianKernel(float sigma, int radius, cv::Mat& kernel);

	void sobelFilter(const cv::Mat& image, cv::Mat& magnitude,
		cv::Mat& orientation);

	float bilinearInterpolation(const cv::Mat& image, float r, float c);

	int findPeaks(const cv::Mat& magnitude, const cv::Mat& orientation,
		cv::Mat& out, float th0);

	int doubleTh(const cv::Mat& magnitude, cv::Mat& out, float th1, float th2);
}  // namespace Canny
