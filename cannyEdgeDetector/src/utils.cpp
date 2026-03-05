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

	Graphics::Image<float> gaussianKernel(float sigma, int radius)
	{
		int size = 2 * radius + 1;
		Graphics::Image<float> kernel(size, 1, 1);
		float sum = 0.0f;
		float g;

		for (int c = 0; c < size; c++) {
			g = (1.0f / (2.0f * M_PI * powf(sigma, 2))) *
				expf(-powf(c, 2) / (2.0f * powf(sigma, 2)));
			kernel(c, 0) = g;
			sum += g;
		}
		for (int c = 0; c < size; c++) {
			kernel(c, 0) /= sum;
		}
		return kernel;
	}


	void sobelFilter(const Graphics::Image<unsigned char>& image,
		Graphics::Image<float>& magnitude, Graphics::Image<float>& orientation)
	{
		// Sobel Kernel
		float vet1[] = { 1.0f, 0.0f, -1.0f, 2.0f, 0.0f, -2.0f, 1.0f, 0.0f, -1.0f };
		Graphics::Image<float> Y(3, 3, 1);
		std::copy(vet1, vet1 + 9, Y.data());

		Graphics::Image<float> Gx = convFloat(image, Y.transpose());	// horizontal gradient
		Graphics::Image<float> Gy = convFloat(image, Y);				// vertical gradient

		magnitude = Graphics::Image<float>(Gx.width(), Gx.height(), 1);
		orientation = Graphics::Image<float>(Gx.width(), Gx.height(), 1);
		std::fill(magnitude.data(), magnitude.data() + magnitude.totalElements(), 0.0f);
		std::fill(orientation.data(), orientation.data() + orientation.totalElements(), 0.0f);

		float max_orientation = 0.0f, min_orientation = 0.0f;
		float max_magnitude = 0.0f, min_magnitude = 0.0f;
		float pixelX, pixelY, mag, ori;

		for (int v = 0; v < Gx.height(); v++) {
			for (int u = 0; u < Gx.width(); u++) {
				pixelX = Gx(u, v);
				pixelY = Gy(u, v);

				// Magnitude of the gradient vector
				mag = sqrtf(powf(pixelX, 2) + powf(pixelY, 2));
				if (mag >= max_magnitude) max_magnitude = mag;
				if (mag <= min_magnitude) min_magnitude = mag;
				magnitude(u, v) = mag;

				// Direction of the gradient vector
				ori = atan2f(pixelY, pixelX);
				if (ori >= max_orientation) max_orientation = ori;
				if (ori <= min_orientation) min_orientation = ori;
				orientation(u, v) = ori;
			}
		}

		// Normalization
		for (int v = 0; v < orientation.height(); v++) {
			for (int u = 0; u < orientation.width(); u++) {
				orientation(u, v) = 2.0f * M_PI * (orientation(u, v) - min_orientation) / (max_orientation - min_orientation);
			}
		}
		for (int v = 0; v < magnitude.height(); v++) {
			for (int u = 0; u < magnitude.width(); u++) {
				magnitude(u, v) = (magnitude(u, v) - min_magnitude) / (max_magnitude - min_magnitude);
			}
		}
	}

	Graphics::Image<unsigned char> findPeaks(const Graphics::Image<float>& magnitude,
		const Graphics::Image<float>& orientation, float th0)
	{
		Graphics::Image<unsigned char> out(magnitude.width(), magnitude.height(), 1);
		std::fill(out.data(), out.data() + out.totalElements(), 0);
		float angle, pixel, e1x, e1y, e2x, e2y, e1, e2;

		for (int r = 0; r < magnitude.height(); r++) {
			for (int c = 0; c < magnitude.width(); c++) {
				pixel = magnitude(c, r);
				angle = orientation(c, r);
				e1x = c + cosf(angle);
				e1y = r + sinf(angle);
				e2x = c - cosf(angle);
				e2y = r - sinf(angle);
				e1 = bilinearInterpolation(magnitude, e1y, e1x);
				e2 = bilinearInterpolation(magnitude, e2y, e2x);
				if (pixel >= e1 && pixel >= e2 && pixel >= th0) {
					out(c, r) = static_cast<unsigned char>(pixel * 255);
				}
				else {
					out(c, r) = 0;
				}
			}
		}

		return out;
	}

	Graphics::Image<unsigned char> doubleTh(const Graphics::Image<unsigned char>& magnitude,
		float th1, float th2)
	{
		Graphics::Image<unsigned char> out(magnitude.width(), magnitude.height(), 1);
		std::fill(out.data(), out.data() + out.totalElements(), 0);
		float val;
		for (int v = 0; v < magnitude.height(); v++) {
			for (int u = 0; u < magnitude.width(); u++) {
				val = static_cast<float>(magnitude(u, v)) / 255.0f;
				if (val > th1) {
					out(u, v) = 255;
				}
				else if (val > th2) {
					out(u, v) = 128;
				}
				else {
					out(u, v) = 0;
				}
			}
		}
		return out;
	}

	Graphics::Image<unsigned char> cannyEdgeDetector(const Graphics::Image<unsigned char>& image,
		float th0, float th1, float th2)
	{
		Graphics::Image<float> magnitude, orientation;
		sobelFilter(image, magnitude, orientation);
		auto support = findPeaks(magnitude, orientation, th0);
		return doubleTh(support, th1, th2);
	}

	Graphics::Image<unsigned char> edgeLinking(const Graphics::Image<unsigned char>& image)
	{
		Graphics::Image<unsigned char> out = image.clone();

		bool changed = true;
		while (changed) 
		{
			changed = false;
			for (int v = 1; v < image.height() - 1; v++) {
				for (int u = 1; u < image.width() - 1; u++) {
					if (out(u, v) == 128) {
						bool nearStrong = false;
						for (int dv = -1; dv <= 1 && !nearStrong; dv++) {
							for (int du = -1; du <= 1 && !nearStrong; du++) {
								if (out(u + du, v + dv) == 255) {
									nearStrong = true;
								}
							}
						}
						if (nearStrong) {
							out(u, v) = 255;
							changed = true;
						}
					}
				}
			}
		}

		for (int v = 0; v < out.height(); v++) {
			for (int u = 0; u < out.width(); u++) {
				if (out(u, v) == 128) out(u, v) = 0;
			}
		}

		return out;
	}
}