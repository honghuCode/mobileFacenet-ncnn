
#include<iostream>
#include<ncnn_mobileFace.h>
#include "net.h"
#include<fstream>
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include "mtcnn.h"
using namespace std;
#define _MSC_VER 1900
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



int test_picture() {
	char *model_path = "./models";
	MTCNN mtcnn(model_path);

	clock_t start_time = clock();

	cv::Mat image;
	image = cv::imread("./sample.jpg");
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
	std::vector<Bbox> finalBbox;

#if(MAXFACEOPEN==1)
	mtcnn.detectMaxFace(ncnn_img, finalBbox);
#else
	mtcnn.detect(ncnn_img, finalBbox);
#endif

	const int num_box = finalBbox.size();
	std::vector<cv::Rect> bbox;
	bbox.resize(num_box);
	for (int i = 0; i < num_box; i++) {
		bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1);

		for (int j = 0; j<5; j = j + 1)
		{
			cv::circle(image, cvPoint(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), CV_FILLED);
		}
	}
	for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
		rectangle(image, (*it), cv::Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow("face_detection", image);
	clock_t finish_time = clock();
	double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
	std::cout << "time" << total_time * 1000 << "ms" << std::endl;

	cv::waitKey(0);
	return 1;
}


cv::Mat getsrc_roi(std::vector<cv::Point2f> x0, std::vector<cv::Point2f> dst)
{
	int size = dst.size();
	cv::Mat A = cv::Mat::zeros(size * 2, 4, CV_32FC1);
	cv::Mat B = cv::Mat::zeros(size * 2, 1, CV_32FC1);

	//[ x1 -y1 1 0] [a]       [x_1]
	//[ y1  x1 0 1] [b]   =   [y_1]
	//[ x2 -y2 1 0] [c]       [x_2]
	//[ y2  x2 0 1] [d]       [y_2]	

	for (int i = 0; i < size; i++)
	{
		A.at<float>(i << 1, 0) = x0[i].x;// roi_dst[i].x;
		A.at<float>(i << 1, 1) = -x0[i].y;
		A.at<float>(i << 1, 2) = 1;
		A.at<float>(i << 1, 3) = 0;
		A.at<float>(i << 1 | 1, 0) = x0[i].y;
		A.at<float>(i << 1 | 1, 1) = x0[i].x;
		A.at<float>(i << 1 | 1, 2) = 0;
		A.at<float>(i << 1 | 1, 3) = 1;

		B.at<float>(i << 1) = dst[i].x;
		B.at<float>(i << 1 | 1) = dst[i].y;
	}

	cv::Mat roi = cv::Mat::zeros(2, 3, A.type());
	cv::Mat AT = A.t();
	cv::Mat ATA = A.t() * A;
	cv::Mat R = ATA.inv() * AT * B;

	//roi = [a -b c;b a d ];

	roi.at<float>(0, 0) = R.at<float>(0, 0);
	roi.at<float>(0, 1) = -R.at<float>(1, 0);
	roi.at<float>(0, 2) = R.at<float>(2, 0);
	roi.at<float>(1, 0) = R.at<float>(1, 0);
	roi.at<float>(1, 1) = R.at<float>(0, 0);
	roi.at<float>(1, 2) = R.at<float>(3, 0);
	return roi;

}


cv::Mat faceAlign(cv::Mat image, MTCNN *mtcnn)
{
	double dst_landmark[10] = {
		38.2946, 73.5318, 55.0252, 41.5493, 70.7299,
		51.6963, 51.5014, 71.7366, 92.3655, 92.2041 };
	vector<cv::Point2f>coord5points;
	vector<cv::Point2f>facePointsByMtcnn;
	for (int i = 0; i < 5; i++) {
		coord5points.push_back(cv::Point2f(dst_landmark[i], dst_landmark[i + 5]));
	}
	char *model_path = "./models";
	(model_path);
	clock_t start_time = clock();

	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
	std::vector<Bbox> finalBbox;

#if(MAXFACEOPEN==1)
	mtcnn->detectMaxFace(ncnn_img, finalBbox);
#else
	mtcnn->detect(ncnn_img, finalBbox);
#endif

	const int num_box = finalBbox.size(); //人脸的数量（默认一张脸）
	std::vector<cv::Rect> bbox;
	bbox.resize(num_box);
	for (int i = 0; i < num_box; i++) {
		for (int j = 0; j<5; j = j + 1)
		{
			//cv::circle(image, cvPoint(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), CV_FILLED);
			facePointsByMtcnn.push_back(cvPoint(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]));
		}
	}

	cv::Mat warp_mat = cv::estimateRigidTransform(facePointsByMtcnn, coord5points, false);
	if (warp_mat.empty()) {
		warp_mat = getsrc_roi(facePointsByMtcnn, coord5points);
	}
	warp_mat.convertTo(warp_mat, CV_32FC1);
	cv::Mat alignFace = cv::Mat::zeros(112, 112, image.type());
	warpAffine(image, alignFace, warp_mat, alignFace.size());
	return alignFace;
}


void main()
{
	cv::Mat image = cv::imread("./wanghan.jpg");
	MTCNN *mtcnn = new MTCNN("./models");
	cv::Mat alignedFace1 = faceAlign(image, mtcnn);
	
	image = cv::imread("./wanghan_2.jpg");

	cv::Mat alignedFace2 = faceAlign(image, mtcnn);

	//cv::imshow("alignedFace1", alignedFace1);
	//cv::waitKey(0);

	//cv::imshow("alignedFace2", alignedFace2);
	//cv::waitKey(0);

	ncnn::Net squeezenet;
	//98.83
	/*squeezenet.load_param("mobilenet_ncnn.param");
	squeezenet.load_model("mobilenet_ncnn.bin");*/
	//99.4
	squeezenet.load_param("mobilefacenet.param");
	squeezenet.load_model("mobilefacenet.bin"); 
	ncnn::Extractor ex = squeezenet.create_extractor();
	ex.set_light_mode(true);

		
	//cout << "lfw-112X112/" + img_L << endl;
	long t1 = clock();
	float* feat1 = getFeatByMobileFaceNetNCNN(ex, alignedFace1);
	float *feat2 = getFeatByMobileFaceNetNCNN(ex, alignedFace2);
	long t2 = clock();


	float sim = CalcSimilarity_1(feat1, feat2, 128);
	fprintf(stderr, "time:%f,sim:%f\n", (t2 - t1) / 2.0,sim);



	//cv::imshow("alignedFace", alignedFace);
	//cv::waitKey(0);
	////人脸对齐
	//
	//ncnn::Net squeezenet;
	////98.83
	///*squeezenet.load_param("mobilenet_ncnn.param");
	//squeezenet.load_model("mobilenet_ncnn.bin");*/
	////99.4
	//squeezenet.load_param("mobilefacenet.param");
	//squeezenet.load_model("mobilefacenet.bin"); 
	//ncnn::Extractor ex = squeezenet.create_extractor();
	//ex.set_light_mode(true);

	//cv::Mat m1 = cv::imread("2_1.jpg", CV_LOAD_IMAGE_COLOR);
	//cv::Mat m2 = cv::imread("2.jpg", CV_LOAD_IMAGE_COLOR);
	//	
	////cout << "lfw-112X112/" + img_L << endl;

	//float* feat1 = getFeatByMobileFaceNetNCNN(ex, m1);
	//float *feat2 = getFeatByMobileFaceNetNCNN(ex, m2);



	//float sim = CalcSimilarity_1(feat1, feat2, 128);
	//fprintf(stderr, "%f\n", sim);
	////LFW: 99.50, CFP_FP: 88.94, AgeDB30: 95.91
	//fstream in("pairs_1.txt");
	//fstream out("rs_lfw_99.50.txt",ios::out);
	//string line;
	//long t1 = clock();
	//int count = 0;
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
	//	out << flag.c_str() << "\t"<<sim << endl;
	//	long t2 = clock();
	//	if (count++ % 10 == 0)
	//	{
	//	
	//		cout << t2 - t1 << "s"<<endl;
	//		t1 = t2;
	//	}
	//}


	//float* getFeatByMobileFaceNetNCNN(ncnn::Extractor ex, cv::Mat img);
	//cout << "ssssss" << endl;
}