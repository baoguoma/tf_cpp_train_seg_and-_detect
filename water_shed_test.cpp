#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "stdafx.h"
#include "tf_detect.h"
#include "water_shed_test.h"
using namespace std;
using namespace cv;


//
/******************************************************************************
* Function: 分水岭算法
* InPut   : width 原图像的宽
high 原图像的高
src_image 指向原图像数据的指针，
border_image指向边缘图像的指针
watershed_thres_hold 将图像二值化时的阈值
* OutPut  : 输出是一个带标号的图像，石头从128开始，边界和外部为0
* Return  : 与输出值相同
* Other   : 该函数与DL_test中的water_shed完全一致,仅用来测试
* Author  : lichao %{CurrentDate:yyyy.MM.dd}
*******************************************************************************/
Mat seg_watershed(int width, int high, uchar* src_image, uchar* border_image)
{
	Mat src(high, width, CV_8U, src_image);
	Mat border(high, width, CV_8U, border_image);

	src = src > 128;
	border = border > 128;

	imwrite("src.bmp", src);

	Mat img_foreground = src - border;		//前景减掉边界;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));

	Mat img_fg_mask;	//前景mask;
	erode(img_foreground, img_fg_mask, element);
	

	Mat img_fg_label(src.size(), CV_32S);	//前景label;
	connectedComponents(img_fg_mask, img_fg_label);
	img_fg_label = img_fg_label + 128;

	Mat img_bg_mask;	//背景mask;
	bitwise_not(src, img_bg_mask);

	erode(img_bg_mask, img_bg_mask, element);
	

	//label添加背景标记;
	int* p_data_fg_label = (int*)img_fg_label.data;
	for (int i = 0; i < img_fg_label.rows*img_fg_label.cols; i++)
	{
		if (p_data_fg_label[i] == 128)
			p_data_fg_label[i] = 0;

		if (img_bg_mask.data[i] > 0)
			p_data_fg_label[i] = 1;
	}

	//----------test--------------




	Mat src0;
	cvtColor(src, src0, CV_GRAY2RGB);
	watershed(src0, img_fg_label);

	
	int width0 = src0.cols;
	int* plabel = (int*)img_fg_label.data;
	

	for (int i = 1; i < src0.rows - 1; i++)
		for (int j = 1; j < src0.cols - 1; j++)
		{
			if (plabel[i*width0 + j] < 128)
			{
				plabel[i*width0 + j] == 0;
				//img_fg_label.data[i*width0 + j] = 0;
			}

		}

	

	return img_fg_label;
	
}

/******************************************************************************
* Function: 切分图像的函数
* InPut   : pimage输入为原图像的数据
block_width 每一个小块的宽
block_height 每一个小块的高
step切分的步长
* OutPut  : 输出是一个vector<mat>，内部存放了切分结果，结果为若干张小图
* Return  : 与输出值相同
* Other   :
* Author  : kebin peng %{CurrentDate:yyyy.MM.dd}
*******************************************************************************/
vector<Mat> slice_image(uchar* pimage, int block_width, int block_height, int origin_width, int origin_height, int step) {
	
	Mat image = Mat(origin_height, origin_width, CV_8UC1, (void*)pimage);
	
	if (step > block_width)
		step = block_width;
	if (step > block_height)
		step = block_height;

	int row_num, col_num;

	row_num = (image.rows + step - 1) / step;//计算行方向有多少块
       	col_num = (image.cols + step - 1) / step;//计算列方向有多少块

	vector<Mat> vimgOut;

	Mat imageROI;

	int row_start, col_start;
	for (int j = 0; j<row_num; j++)
	{
		for (int i = 0; i<col_num; i++)
		{

			//计算每一个小块应该从什么地方开始
			row_start = j*step;
			if (row_start + block_height > image.rows) {//如果超出了图像范围，则从图像边界往反方向取一块
				row_start = image.rows - block_height;
			}

			col_start = i*step;
			if (col_start + block_width > image.cols) {
				col_start = image.cols - block_width;
			}


			imageROI = image(Rect(col_start, row_start, block_width, block_height));//rect(x, y, width, height)选定感兴趣区域  
			vimgOut.push_back(imageROI);
			

		}
	}

	return vimgOut;
}


/******************************************************************************
* Function: 合并图像的函数
* InPut   : image存放待合并图像的vector
origin_width 每一个小块的宽
origin_height 每一个小块的高
step切分的步长
bi_thres_hold二值化时的阈值
* OutPut  : 输出是一个mat，存放了合并后的图像
* Return  : 与输出值相同
* Other   : 该函数与DL_detect中的函数不同，其区别在于该函数没有进行二值化化操作
            因为这里仅仅时测试拼接功能的正常，不需要卡阈值
* Author  : kebin peng %{CurrentDate:yyyy.MM.dd}
*******************************************************************************/
Mat merge_image(vector<Mat> image, int origin_width, int origin_height, int step, float bi_thres_hold) {


	Mat result(origin_height, origin_width, CV_8UC1, Scalar(0));


	int block_height = image[0].rows;
	int block_width = image[0].cols;

	if (step > block_width)
		step = block_width;
	if (step > block_height)
		step = block_height;

	int row_num, col_num;

	row_num = (origin_height + step - 1) / step;//计算行方向有多少块
	col_num = (origin_width + step - 1) / step;//计算列方向有多少块


	vector<Mat> imgOut;

	Mat imageROI;
	Mat mid = Mat(block_width, block_height, CV_8UC1, Scalar(0));
	int row_start, col_start;
	for (int j = 0; j<row_num; j++)
	{
		for (int i = 0; i<col_num; i++)
		{


			row_start = j*step;//计算每一个小块在原图中应该处于什么位置，与切分图像的操作是一致的
			if (row_start + block_height > origin_height) {
				row_start = origin_height - block_height;
			}



			col_start = i*step;
			if (col_start + block_width > origin_width) {
				col_start = origin_width - block_width;
			}

			Mat imageROI = result(Rect(col_start, row_start, block_width, block_height));
			
			Mat bi_img = image[j*col_num + i];
			mid = (bi_img | imageROI);
			mid.copyTo(imageROI);
			
		}
	}
	
	return result;
}
