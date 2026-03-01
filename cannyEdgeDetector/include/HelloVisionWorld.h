#pragma once

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "Image.h"
#include "utils.h"

int runCannyEdgeDetector(Graphics::Image<unsigned char> img);