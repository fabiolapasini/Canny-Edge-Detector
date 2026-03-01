#pragma once

#include <filesystem>
#include "Image.h"

namespace Canny
{
	Graphics::Image<unsigned char> maxPooling(const Graphics::Image<unsigned char>& image, int size, int stride);

	Graphics::Image<unsigned char> averagePooling(const Graphics::Image<unsigned char>& image, int size, int stride);

	Graphics::Image<float> convFloat(const Graphics::Image<unsigned char>& image, 
		const Graphics::Image<float>& kernel, int stride = 1);

	Graphics::Image<unsigned char> convInt(const Graphics::Image<unsigned char>& image,
		const Graphics::Image<float>& kernel, int stride = 1);

	void gaussianKernel(float sigma, int radius, cv::Mat& kernel);

	void sobelFilter(const cv::Mat& image, cv::Mat& magnitude,
		cv::Mat& orientation);

	float bilinearInterpolation(const cv::Mat& image, float r, float c);

	int findPeaks(const cv::Mat& magnitude, const cv::Mat& orientation,
		cv::Mat& out, float th0);

	int doubleTh(const cv::Mat& magnitude, cv::Mat& out, float th1, float th2);
}  // namespace Canny
