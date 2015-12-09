#include <iostream>
#include "cv.h"
#include "highgui.h"
#include <string>
#include <ctime>
using namespace std;
using namespace cv;

#define debug

// string fold_path = "./1124/";
// string img_name = "1.bmp";
// string img_path = fold_path + img_name;
string write_path = "../Camera-lens-glassFiles/save/";


void show(const vector<Point>& vec)
{
	for (int i = 0; i != vec.size(); ++i)
	{
		cout << "(" << vec[i].x << "," << vec[i].y << ")" << endl;
	}
}
void my_boxFilter(Mat src, Mat& mean,int blockWidth,int blockHeight)
{
	int halfwidth = (blockWidth - 1)/2;
	int halfheight = (blockHeight - 1)/2;
	for(int i = 0; i != src.rows; ++i)
		for(int j = 0; j != src.cols; ++j)
		{
			int xstart = (j - halfwidth >= 0) ? (j - halfwidth) : 0;
			int ystart = (i - halfheight >= 0) ? (i - halfheight) : 0;
			int xend = (j + halfwidth < src.cols) ? (j + halfwidth) : src.cols - 1;
			int yend = (i + halfheight < src.rows) ? (i + halfheight) : src.rows - 1;
			int count(0);
			int sum(0);
			for(int m = xstart; m <= xend; ++m)
				for(int n = ystart; n <= yend; ++n)
				{
					if(src.at<uchar>(n,m) != 0)
					{
						sum += src.at<uchar>(n,m);
						++count;
					}
				}
			if(count == 0)
				mean.at<uchar>(i,j) = 0;
			else
				mean.at<uchar>(i,j) = sum / count;
		}
#ifdef debug
	imshow("mean",mean);
#endif
}


void my_adaptiveThreshold(Mat src,Mat& dst,int blockWidth,int blockHeight,int c)
{
	Mat mean = Mat::zeros(src.size(),CV_8UC1);
	my_boxFilter(src,mean,blockWidth,blockHeight);
	for(int i = 0; i != src.rows; ++i)
		for(int j = 0; j != src.cols; ++j)
		{
			if(int(src.at<uchar>(i,j)) - int(mean.at<uchar>(i,j)) >= c)
				dst.at<uchar>(i,j) = 255;
			else
				dst.at<uchar>(i,j) = 0;
		}
}

Mat get_pic_dis_withCircleSeedPoints(Mat res, Mat orign)
{
	Mat orignSave = orign.clone();
	erode(res,res,Mat());
	erode(res,res,Mat());
	erode(res,res,Mat());
	erode(res,res,Mat());
	erode(res,res,Mat());
	erode(res,res,Mat());
	erode(res,res,Mat());
	erode(res,res,Mat());
	erode(res,res,Mat());
	erode(res,res,Mat());
	erode(res,res,Mat());
	erode(res,res,Mat());
	erode(res,res,Mat());
	vector<vector<Point> > CircleSeedPoints;
	cvtColor(res, res, CV_BGR2GRAY);
	findContours(res, CircleSeedPoints, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	for (int i = 0; i < CircleSeedPoints[0].size(); ++i)
	{
		if(orign.at<uchar>(CircleSeedPoints[0][i].y,CircleSeedPoints[0][i].x) == 0)
			floodFill(orign, CircleSeedPoints[0][i], Scalar(255, 255, 255));
	}
	Mat pic_dis = orign - orignSave;
#ifdef debug
	imshow("get_pic_dis_withCircleSeedPoints",pic_dis);
	imwrite(write_path + "get_pic_dis_withCircleSeedPoints.bmp", pic_dis);	
#endif
	return pic_dis;
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
	Mat flood_binary_src_withCircleSeedPoints = src.clone();
#ifdef debug
	imshow("src", src);
	imwrite(write_path + "src.bmp", src);
#endif
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
	//cout << "randius" << radius << endl;
	//cout << "(" << center.x << "," << center.y << ")" << endl<<endl;
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
	
	//ÏÂÃæÊÇ»ñÈ¡Ëùº¬ÐÎ×´µÄÍâ½ÓÔ²
	Point2f enclosing_cecter;
	float enclosing_radius(0);
	threshold(pic_dis, pic_dis, 150, 255, THRESH_BINARY);
#ifdef debug
	imwrite(write_path + "pic_dis.bmp", pic_dis);
	//imshow("pic_dis", pic_dis);
#endif

//**********************according Circumcircle to get mask****************************
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
	//show(seedPoints);
	//return ans;
//***********************************************************************************

//**********************according convexHull to get mask*****************************
	//Mat pic_disCopy = pic_dis.clone();
	Mat pic_disCopy = get_pic_dis_withCircleSeedPoints(ans,flood_binary_src_withCircleSeedPoints);
	vector< vector<Point> > convexHullcont;
	vector<Vec4i> hierarchy;
	findContours(pic_disCopy,convexHullcont,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<vector<Point> > hull(convexHullcont.size());
	for( int i = 0; i < convexHullcont.size(); i++ )
    {  
      	convexHull( Mat(convexHullcont[i]), hull[i], false ); 
    }
	Mat drawing = Mat::zeros(pic_disCopy.size(), CV_8UC3 );
	for(int i = 0; i != convexHullcont.size(); ++i)
	{
		drawContours( drawing, hull, i, Scalar(255,255,255), -1, 8, vector<Vec4i>(), 0, Point() );
	}
#ifdef debug
	imshow("drawing",drawing);
	imwrite(write_path + "drawing.bmp",drawing);
	//waitKey(0);
#endif
	//pyrUp(drawing,drawing);
	//pyrUp(drawing,drawing);
	return drawing;
}

void getRidOfThreshEdges(Mat& thresholdResult)
{
#ifdef debug
	imshow("thresholdResult",thresholdResult);
#endif
	Point2f center;
	float radius(0);
	Mat src = thresholdResult.clone();
	Mat cont_src = thresholdResult.clone();
	Mat cont_canvas = Mat::zeros(thresholdResult.size(),CV_8UC3);
	vector< vector<Point> > cont;
	findContours(cont_src,cont,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	minEnclosingCircle(cont[0],center,radius);
	Mat canvas = Mat::zeros(thresholdResult.size(),CV_8UC3);
	circle(canvas,center,radius,Scalar(0,255,0),-1,8,0);
#ifdef debug
	imshow("fill_canva",canvas);
#endif
	return;
}

Mat findThreshResult(Mat src,Mat mask)
{
	cvtColor(mask, mask, CV_BGR2GRAY);
#ifdef debug
	imshow("mask", mask);
#endif
	Mat and_img = Mat::zeros(src.size(),CV_8UC1);
	//erode(mask, mask, Mat());
	//erode(mask, mask, Mat());
	bitwise_and(mask, src, and_img);

	// Mat mean = Mat::zeros(and_img.size(),CV_8UC1);
	// my_boxFilter(and_img,mean,7,7);

#ifdef debug
	imwrite(write_path + "seg_img.bmp", and_img);
	imshow("seg_img", and_img);
#endif
	//adaptiveThreshold(and_img, and_img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, -3);
	Mat adaptiveThresholdResult = Mat::zeros(src.size(),CV_8UC1);
	my_adaptiveThreshold(and_img,adaptiveThresholdResult,7,7,5);
	Mat res;
	erode(mask, mask, Mat());
	erode(mask, mask, Mat());
	bitwise_and(adaptiveThresholdResult, mask, res);
#ifdef debug
	imshow("and_img", adaptiveThresholdResult);
	imwrite(write_path + "and_img_1.bmp",adaptiveThresholdResult);
	imwrite(write_path + "and_img.bmp",res);
	imshow("res", res);
	imwrite(write_path + "res.bmp",res);
#endif
	return res;
}

void process(char** argv)
{
	Mat src = imread(argv[1], 0);
	//pyrDown(src, src);
	//pyrDown(src, src);
	Mat mask = findCircleMask(src);
	Mat threshResult = findThreshResult(src,mask);
}

// Mat getRidOfBoundsMask(Mat& src)
// {
// 	vector<Point> whitePos;
// 	for(int i = 0; i != src.rows;++i)
// 		for(int j = 0; j != src.cols; ++j)
// 		{
// 			if(src.at<uchar>(i,j) == 255)
// 				whitePos.push_back(Point(j,i));
// 		}
// 	Point2f center;
// 	float radius;
// 	minEnclosingCircle(whitePos,center,radius);
// 	Mat canvas = Mat::zeros(src.size(),CV_8UC1);
// 	circle(canvas, center, radius, Scalar(255, 255, 255), -1, 8, 0);
// 	imshow("drawCanvas",canvas);
// 	return canvas;
// }

int main(int argc,char** argv)
{
	double t = (double)getTickCount();
	process(argv);
	t = ((double)getTickCount() - t)/getTickFrequency();
	cout <<"program time = " << t*1000.0 << "ms" << endl;
#ifdef debug
	waitKey(0);
#endif
	return 0;
}

