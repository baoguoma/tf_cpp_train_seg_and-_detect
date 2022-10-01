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
* Function: ��ˮ���㷨
* InPut   : width ԭͼ��Ŀ�
high ԭͼ��ĸ�
src_image ָ��ԭͼ�����ݵ�ָ�룬
border_imageָ���Եͼ���ָ��
watershed_thres_hold ��ͼ���ֵ��ʱ����ֵ
* OutPut  : �����һ������ŵ�ͼ��ʯͷ��128��ʼ���߽���ⲿΪ0
* Return  : �����ֵ��ͬ
* Other   : �ú�����DL_test�е�water_shed��ȫһ��,����������
* Author  : lichao %{CurrentDate:yyyy.MM.dd}
*******************************************************************************/
Mat seg_watershed(int width, int high, uchar* src_image, uchar* border_image)
{
	Mat src(high, width, CV_8U, src_image);
	Mat border(high, width, CV_8U, border_image);

	src = src > 128;
	border = border > 128;

	imwrite("src.bmp", src);

	Mat img_foreground = src - border;		//ǰ�������߽�;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));

	Mat img_fg_mask;	//ǰ��mask;
	erode(img_foreground, img_fg_mask, element);
	

	Mat img_fg_label(src.size(), CV_32S);	//ǰ��label;
	connectedComponents(img_fg_mask, img_fg_label);
	img_fg_label = img_fg_label + 128;

	Mat img_bg_mask;	//����mask;
	bitwise_not(src, img_bg_mask);

	erode(img_bg_mask, img_bg_mask, element);
	

	//label��ӱ������;
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
* Function: �з�ͼ��ĺ���
* InPut   : pimage����Ϊԭͼ�������
block_width ÿһ��С��Ŀ�
block_height ÿһ��С��ĸ�
step�зֵĲ���
* OutPut  : �����һ��vector<mat>���ڲ�������зֽ�������Ϊ������Сͼ
* Return  : �����ֵ��ͬ
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

	row_num = (image.rows + step - 1) / step;//�����з����ж��ٿ�
       	col_num = (image.cols + step - 1) / step;//�����з����ж��ٿ�

	vector<Mat> vimgOut;

	Mat imageROI;

	int row_start, col_start;
	for (int j = 0; j<row_num; j++)
	{
		for (int i = 0; i<col_num; i++)
		{

			//����ÿһ��С��Ӧ�ô�ʲô�ط���ʼ
			row_start = j*step;
			if (row_start + block_height > image.rows) {//���������ͼ��Χ�����ͼ��߽���������ȡһ��
				row_start = image.rows - block_height;
			}

			col_start = i*step;
			if (col_start + block_width > image.cols) {
				col_start = image.cols - block_width;
			}


			imageROI = image(Rect(col_start, row_start, block_width, block_height));//rect(x, y, width, height)ѡ������Ȥ����  
			vimgOut.push_back(imageROI);
			

		}
	}

	return vimgOut;
}


/******************************************************************************
* Function: �ϲ�ͼ��ĺ���
* InPut   : image��Ŵ��ϲ�ͼ���vector
origin_width ÿһ��С��Ŀ�
origin_height ÿһ��С��ĸ�
step�зֵĲ���
bi_thres_hold��ֵ��ʱ����ֵ
* OutPut  : �����һ��mat������˺ϲ����ͼ��
* Return  : �����ֵ��ͬ
* Other   : �ú�����DL_detect�еĺ�����ͬ�����������ڸú���û�н��ж�ֵ��������
            ��Ϊ�������ʱ����ƴ�ӹ��ܵ�����������Ҫ����ֵ
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

	row_num = (origin_height + step - 1) / step;//�����з����ж��ٿ�
	col_num = (origin_width + step - 1) / step;//�����з����ж��ٿ�


	vector<Mat> imgOut;

	Mat imageROI;
	Mat mid = Mat(block_width, block_height, CV_8UC1, Scalar(0));
	int row_start, col_start;
	for (int j = 0; j<row_num; j++)
	{
		for (int i = 0; i<col_num; i++)
		{


			row_start = j*step;//����ÿһ��С����ԭͼ��Ӧ�ô���ʲôλ�ã����з�ͼ��Ĳ�����һ�µ�
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
