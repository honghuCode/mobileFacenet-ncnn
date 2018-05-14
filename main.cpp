
#include<iostream>
#include<ncnn_mobileFace.h>
#include "net.h"
using namespace std;




std::vector<std::string> splitString_1(const std::string &str,
	const char delimiter) {
	std::vector<std::string> splited;
	std::string s(str);
	size_t pos;

	while ((pos = s.find(delimiter)) != std::string::npos) {
		std::string sec = s.substr(0, pos);

		if (!sec.empty()) {
			splited.push_back(s.substr(0, pos));
		}

		s = s.substr(pos + 1);
	}

	splited.push_back(s);

	return splited;
}



float simd_dot_1(const float* x, const float* y, const long& len) {
	float inner_prod = 0.0f;
	__m128 X, Y; // 128-bit values
	__m128 acc = _mm_setzero_ps(); // set to (0, 0, 0, 0)
	float temp[4];

	long i;
	for (i = 0; i + 4 < len; i += 4) {
		X = _mm_loadu_ps(x + i); // load chunk of 4 floats
		Y = _mm_loadu_ps(y + i);
		acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));
	}
	_mm_storeu_ps(&temp[0], acc); // store acc into an array
	inner_prod = temp[0] + temp[1] + temp[2] + temp[3];

	// add the remaining values
	for (; i < len; ++i) {
		inner_prod += x[i] * y[i];
	}
	return inner_prod;
}
float CalcSimilarity_1(const float* fc1,
	const float* fc2,
	long dim) {

	return simd_dot_1(fc1, fc2, dim)
		/ (sqrt(simd_dot_1(fc1, fc1, dim))
			* sqrt(simd_dot_1(fc2, fc2, dim)));
}



void main()
{
	ncnn::Net squeezenet;
	squeezenet.load_param("mobilenet_ncnn.param");
	squeezenet.load_model("mobilenet_ncnn.bin");
	ncnn::Extractor ex = squeezenet.create_extractor();
	ex.set_light_mode(true);

	cv::Mat m1 = cv::imread("2_1.jpg", CV_LOAD_IMAGE_COLOR);
	cv::Mat m2 = cv::imread("2.jpg", CV_LOAD_IMAGE_COLOR);
		
	//cout << "lfw-112X112/" + img_L << endl;

	float* feat1 = getFeatByMobileFaceNetNCNN(ex, m1);
	float *feat2 = getFeatByMobileFaceNetNCNN(ex, m2);



	float sim = CalcSimilarity_1(feat1, feat2, 128);
	fprintf(stderr, "%f\n", sim);

	//fstream in("pairs_1.txt");
	//string line;
	//while (in >> line)
	//{
	//	//cout <<line<<endl;
	//	std::vector<std::string>  rs = splitString_1(line, ',');

	//	string img_L = rs[0];
	//	string img_R = rs[1];
	//	string flag = rs[2];
	//	//cout <<img_L<<endl;
	//	std::vector<float> cls_scores;
	//	cv::Mat m1 = cv::imread("lfw-112X112/" + img_L, CV_LOAD_IMAGE_COLOR);
	//	cv::Mat m2 = cv::imread("lfw-112X112/" + img_R, CV_LOAD_IMAGE_COLOR);
	//	
	//	//cout << "lfw-112X112/" + img_L << endl;

	//	float* feat1 = getFeatByMobileFaceNetNCNN(ex, m1);
	//	float *feat2 = getFeatByMobileFaceNetNCNN(ex, m2);



	//	float sim = CalcSimilarity_1(feat1, feat2, 128);
	//	fprintf(stderr, "%s,%f\n", flag.c_str(), sim);
	//	//out << flag.c_str() << sim << endl;
	//}


	//float* getFeatByMobileFaceNetNCNN(ncnn::Extractor ex, cv::Mat img);
	//cout << "ssssss" << endl;
}