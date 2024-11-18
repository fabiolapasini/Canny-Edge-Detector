#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <opencv2/opencv.hpp>

#include "src/utils.cpp"
#include "include/utils.h"

using namespace cv;
using namespace std;


int main(int argc, char** argv) {
  runCannyEdgeDetector();
  return 0;
}
