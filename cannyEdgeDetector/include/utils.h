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

	Graphics::Image<float> gaussianKernel(float sigma, int radius);

	void sobelFilter(const Graphics::Image<unsigned char>& image,
		Graphics::Image<float>& magnitude, Graphics::Image<float>& orientation);


	template <typename T>
	float bilinearInterpolation(const Graphics::Image<T>& image, float r, float c) {
		int cF = static_cast<int>(std::floor(c));
		int rF = static_cast<int>(std::floor(r));
		float s = r - rF;
		float t = c - cF;
		if (cF < 0 || rF < 0 || cF + 1 >= image.width() || rF + 1 >= image.height()) {
			return 0.0f;
		}
		float f00 = static_cast<float>(image(cF, rF));
		float f10 = static_cast<float>(image(cF, rF + 1));
		float f01 = static_cast<float>(image(cF + 1, rF));
		float f11 = static_cast<float>(image(cF + 1, rF + 1));
		return (1 - t) * (1 - s) * f00 + s * (1 - t) * f10 +
			(1 - s) * t * f01 + s * t * f11;
	}

	Graphics::Image<unsigned char> findPeaks(const Graphics::Image<float>& magnitude,
		const Graphics::Image<float>& orientation, float th0);

	Graphics::Image<unsigned char> doubleTh(const Graphics::Image<unsigned char>& magnitude,
		float th1, float th2);

	Graphics::Image<unsigned char> cannyEdgeDetector(const Graphics::Image<unsigned char>& image,
		float th0, float th1, float th2);

	Graphics::Image<unsigned char> edgeLinking(const Graphics::Image<unsigned char>& image);
}  // namespace Canny
