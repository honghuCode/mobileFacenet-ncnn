#ifndef TestDll_H_
#define TestDll_H_
#ifdef MYLIBDLL
#define MYLIBDLL extern "C" _declspec(dllimport) 
#else
#define MYLIBDLL extern "C" _declspec(dllexport) 
#endif
#include <opencv2/opencv.hpp>
#include "net.h"
MYLIBDLL float* getFeatByMobileFaceNetNCNN(ncnn::Extractor ex,cv::Mat img);
//You can also write like this:
//extern "C" {
//_declspec(dllexport) int Add(int plus1, int plus2);
//};
#endif