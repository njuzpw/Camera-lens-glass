#include <iostream>
#include "cv.h"
#include "highgui.h"
#include <string>
using namespace std;
using namespace cv;

#define debug

string fold_path = "./1124/";
string img_name = "1.bmp";
string img_path = fold_path + img_name;
string write_path = "~/Project/Camera-lens-glassFiles/save";


void show(const vector<Point>& vec)
{
	for (int i = 0; i != vec.size(); ++i)
	{
		cout << "(" << vec[i].x << "," << vec[i].y << ")" << endl;
	}
}


Mat  findCircleMask(const Mat& _src)
{
	Mat src, src_gray;
	//src = imread(img_path, 0);
	src = _src.clone();
	//pyrDown(src, src);
	//pyrDown(src, src);
	threshold(src, src, 0, 255, THRESH_BINARY | CV_THRESH_OTSU);
	Mat	binary_src = src.clone();
	Mat flood_binary_src = src.clone();

	imshow("src", src);
	imwrite(write_path + "src.bmp", src);
	cvtColor(src, src, CV_GRAY2BGR);
	if (!src.data)
	{
		return Mat();
	}
	cvtColor(src, src_gray, CV_BGR2GRAY);
	GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);
	vector<Vec3f> circles;
	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, 10, 200, 100, 0, 0);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		circle(src, center, 3, Scalar(0, 0, 255), -1, 8, 0);
		circle(src, center, radius, Scalar(0, 0, 255), 1, 8, 0);
	}
	Point center;
	int radius(INT_MAX);
	assert(circles.size());
	for (int i = 0; i != circles.size(); ++i)
	{
		if (circles[i][2] < radius)
		{
			radius = circles[i][2];
			center.x = circles[i][0];
			center.y = circles[i][1];
		}
	}
	cout << "randius" << radius << endl;
	cout << "(" << center.x << "," << center.y << ")" << endl<<endl;
	vector<Point> seedPoints;
	Point TempSeedPoint;
	int dis = int(radius / 2 / 1.414);
	//1
	TempSeedPoint.x = center.x - dis;
	TempSeedPoint.y = center.y - dis;
	seedPoints.push_back(TempSeedPoint);
	//2
	TempSeedPoint.x = center.x;
	TempSeedPoint.y = center.y - dis;
	seedPoints.push_back(TempSeedPoint);
	//3
	TempSeedPoint.x = center.x + dis;
	TempSeedPoint.y = center.y - dis;
	seedPoints.push_back(TempSeedPoint);
	//4
	TempSeedPoint.x = center.x + dis;
	TempSeedPoint.y = center.y;
	seedPoints.push_back(TempSeedPoint);
	//5
	TempSeedPoint.x = center.x + dis;
	TempSeedPoint.y = center.y + dis;
	seedPoints.push_back(TempSeedPoint);
	//6
	TempSeedPoint.x = center.x;
	TempSeedPoint.y = center.y + dis;
	seedPoints.push_back(TempSeedPoint);
	//7
	TempSeedPoint.x = center.x - dis;
	TempSeedPoint.y = center.y + dis;
	seedPoints.push_back(TempSeedPoint);
	//8
	TempSeedPoint.x = center.x - dis;
	TempSeedPoint.y = center.y;
	seedPoints.push_back(TempSeedPoint);

	for (int i = 0; i != seedPoints.size(); ++i)
	{
		circle(src, seedPoints[i], 3, Scalar(0, 255, 0), -1, 8, 0);
	}
#ifdef debug
	imshow("pointsPosition", src);
	imwrite(write_path + "pointsPosition.bmp", src);
#endif

	for (int i = 0; i != seedPoints.size(); ++i)
	{
		floodFill(flood_binary_src, seedPoints[i], Scalar(255, 255, 255));
	}
#ifdef debug
	imshow("floodresults", flood_binary_src);
	imwrite(write_path + "floodresults.bmp", flood_binary_src);
#endif
	Mat pic_dis = flood_binary_src - binary_src;
	
	//下面是获取所含形状的外接圆
	Point2f enclosing_cecter;
	float enclosing_radius(0);
	threshold(pic_dis, pic_dis, 150, 255, THRESH_BINARY);
#ifdef debug
	imwrite(write_path + "pic_dis.bmp", pic_dis);
	//imshow("pic_dis", pic_dis);
#endif
	vector< vector<Point> > cont;
	findContours(pic_dis, cont, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	Mat canvas = Mat::zeros(src.size(), CV_8UC3);
	Mat ans = canvas.clone();
	drawContours(canvas, cont, -1, Scalar(255, 255, 255));
	minEnclosingCircle(cont[0], enclosing_cecter, enclosing_radius);
	cout << "enclosing_radius" << enclosing_radius << endl;
	circle(ans, enclosing_cecter, enclosing_radius, Scalar(255, 255, 255), -1, 8, 0);
	threshold(ans, ans, 150, 255, THRESH_BINARY);
#ifdef debug
	imwrite(write_path + "ans.bmp",ans);
	//imshow("ans", ans);
#endif
	show(seedPoints);
	return ans;
}

void process(char** argv)
{
	Mat src = imread(argv[1], 0);
	pyrDown(src, src);
	pyrDown(src, src);
	Mat mask = findCircleMask(src);
	cvtColor(mask, mask, CV_BGR2GRAY);
#ifdef debug
	imshow("mask", mask);
#endif
	Mat and_img = Mat::zeros(src.size(),CV_8UC1);
	erode(mask, mask, Mat());
	erode(mask, mask, Mat());
	bitwise_and(mask, src, and_img);
#ifdef debug
	imwrite(write_path + "seg_img.bmp", and_img);
	imshow("seg_img", and_img);
#endif
	adaptiveThreshold(and_img, and_img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, -3);
	Mat res;
	erode(mask, mask, Mat());
	erode(mask, mask, Mat());
	bitwise_and(and_img, mask, res);
#ifdef debug
	//imshow("and_img", res);
	imwrite(write_path + "and_img.bmp",res);
	imshow("res", res);
	imwrite(write_path + "res.bmp",res);
#endif
}

int main(int argc,char** argv)
{
	/*Mat mask = findCircleMask();
	imshow("mask", mask);*/
	process(argv);
	waitKey(0);
	return 0;
}

