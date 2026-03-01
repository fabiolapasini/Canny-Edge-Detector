
#include "include/HelloVisionWorld.h"

#include "src/utils.cpp"

using namespace cv;
using namespace std;

int runCannyEdgeDetector(Graphics::Image<unsigned char> img)
{
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;


	cv::Mat image = cv::imread("Lenna.png", CV_8UC1);
	if (image.empty()) {
		std::cout << "Unable to open " << frame_name << std::endl;
		return 1;
	}


	cv::Mat mat(img.height(), img.width(), CV_8UC1, img.data());
	cv::namedWindow("Lenna", cv::WINDOW_NORMAL);
	cv::imshow("Lenna", mat);
	cv::waitKey(5000);
	cv::destroyWindow("Lenna");

	int size = 3;
	int stride = 2;
	int sigma = 5;
	int radius = 10;

	float th0 = 0.15f;
	float th1 = 0.3f;
	float th2 = 0.2f;

	Graphics::Image<float> kernel_3x3(3, 3, 1);
	float data_3x3[] = { 0.0f, -1.0f, 0.0f, -1.0f, 5.0f, -1.0f, 0.0f, -1.0f, 0.0f };
	std::copy(data_3x3, data_3x3 + 9, kernel_3x3.data());

	Graphics::Image<float> kernel_1x3(3, 1, 1);
	float data_1x3[] = { 0.0f, 1.0f, 0.0f };
	std::copy(data_1x3, data_1x3 + 3, kernel_1x3.data());

	// 1) MAX POOLLING
	auto first = Canny::maxPooling(img, size, stride);

	// 2) AVERAGE POOLLING
	auto second = Canny::averagePooling(img, size, stride);

	// 3) FLOAT CONVOLUTION
	auto third = Canny::convFloat(img, kernel_3x3, stride);

	// 4) INT CONVOLUTION
	auto fourth = Canny::convInt(img, kernel_3x3);

	//// 5) GAUSSIAN KERNEL
	//cv::Mat fifth;
	//Canny::gaussianKernel(sigma, radius, kernel1D);
	//Canny::convInt(image, kernel1D, fifth, 1);

	//// 6) VARIUS FILTERS
	//cv::Mat sixtB;
	//Canny::convInt(image, kernel1D.t(), sixtB, 1);

	//cv::Mat sixt;
	//Canny::convInt(fifth, kernel1D.t(), sixt, 1);

	//// 7) MAGNITUDE & ORIENTATION
	//cv::Mat magnitude, orientation, Gx, Gy;
	//Canny::sobelFilter(image, magnitude, orientation);

	//cv::Mat adjMap;
	//convertScaleAbs(orientation, adjMap, 255 / (2 * M_PI));
	//cv::Mat falseColorsMap;
	//applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_AUTUMN);

	//// 8) BILINEAR INTERPOLATION
	//float r = 27.8f;
	//float c = 11.4f;
	//float eight = Canny::bilinearInterpolation(image, r, c);
	//std::cout << "Bilinear interpolation between: " << r << " and " << c << " = "
	//          << eight << std::endl;

	//// 9) FIND PEAKS
	//cv::Mat ninth(cv::Size(image.cols, image.rows), CV_32F);
	//Canny::findPeaks(magnitude, orientation, ninth, th0);

	//// 10) HISTERESIS THRESHOLD
	//cv::Mat tenth(cv::Size(image.cols, image.rows), CV_8U);
	//Canny::doubleTh(magnitude, tenth, th1, th2);

	//// 11) CANNY
	//cv::Mat eleventh(cv::Size(image.cols, image.rows), CV_8UC1);
	//Canny::cannyEdgeDetector(image, eleventh, th0, th1, th2);
	//cv::namedWindow("Canny");
	//cv::imshow("Canny", eleventh);
	//unsigned char key = cv::waitKey(0);
	//if (key == 'q') cv::destroyWindow("Canny");

	//  // 12) EDGE LINKING

#ifdef _SAVE_IMG_
	Canny::saveRAW(first, "MaxPooling.raw");
	Canny::saveRAW(second, "AveragePooling.raw");
	Canny::saveRAW(third, "FloatConvolution.raw");
	Canny::saveRAW(fourth, "IntConvolution.raw");
	Canny::saveRAW(fifth, "HorizontalGaussian.raw");
	Canny::saveRAW(sixtB, "VerticalGaussian.raw");
	Canny::saveRAW(sixt, "BidimensionalGaussian.raw");
	Canny::saveRAW(magnitude, "Magnitude.raw");
	Canny::saveRAW(falseColorsMap, "Orientation.raw");
	Canny::saveRAW(ninth, "NonMaxSuppression.raw");
	Canny::saveRAW(tenth, "Hysteresis.raw");
	Canny::saveRAW(eleventh, "Canny.raw");
#endif
}

int main(int argc, char** argv) {
	Graphics::Image<unsigned char> img =
		Graphics::Image<unsigned char>::loadRAW("Lenna.raw", 512, 512, 3);

	Graphics::Image<unsigned char> gray_img(512, 512, 1);
	for (int y = 0; y < img.height(); ++y) {
		for (int x = 0; x < img.width(); ++x) {
			unsigned char r = img(x, y, 0);
			unsigned char g = img(x, y, 1);
			unsigned char b = img(x, y, 2);
			// Y' = 0.299·R + 0.587·G + 0.114·B
			gray_img(x, y, 0) =
				static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
		}
	}

	runCannyEdgeDetector(std::move(gray_img));

	return 0;
}
