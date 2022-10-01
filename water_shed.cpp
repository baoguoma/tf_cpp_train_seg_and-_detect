#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "opencv2/opencv.hpp"
#include "stdafx.h"

using namespace std;
using namespace cv;

/******************************************************************************
* Function: watershed
* InPut   : width
            high
                        src_image ָ
                        border_imageָ
                        watershed_threshold
* OutPut  : segmented image
* Return  : CV_Mat
* Other   :
* Author  : kebin peng
*******************************************************************************/
Mat seg_watershed(uchar* src_image, uchar* border_image, int width, int high) {
  Mat src(high, width, CV_8UC1, src_image);
  Mat border(high, width, CV_8UC1, border_image);

  // src = src > watershed_thres_hold;
  // border = border > watershed_thres_hold;

  Mat element = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));

  Mat img_fg_mask;  // front ground mask;
  erode(img_foreground, img_fg_mask, element);

  Mat img_fg_label(src.size(), CV_32S);  // pixel label;
  connectedComponents(img_fg_mask, img_fg_label);
  img_fg_label = img_fg_label + 128;

  Mat img_bg_mask;  // background mask;
  bitwise_not(src, img_bg_mask);
  erode(img_bg_mask, img_bg_mask, element);

  // pointer to the front ground label;
  int* p_data_fg_label = (int*)img_fg_label.data;
  for (int i = 0; i < img_fg_label.rows * img_fg_label.cols; i++) {
    if (p_data_fg_label[i] == 128) p_data_fg_label[i] = 0;

    if (img_bg_mask.data[i] > 0) p_data_fg_label[i] = 1;
  }

  //----------test--------------

  Mat src0;
  cvtColor(src, src0, CV_GRAY2RGB);
  watershed(src0, img_fg_label);

  int* plabel = (int*)img_fg_label.data;
  for (int i = 0; i < src.rows * src.cols; i++) {
    if (plabel[i] < 128) plabel[i] = 0;
  }

  for (int i = 1; i < src0.rows - 1; i++)
    for (int j = 1; j < src0.cols - 1; j++) {
      if (plabel[i * width0 + j] < 128) {
        plabel[i * width0 + j] = 0;
        // img_fg_label.data[i*width0 + j] = 0;
      }
    }

  return img_fg_label;
}

/******************************************************************************
* Function: slice whole image into small block and combine again to remove the
edge
* InPut   : pimage
            block_width
                        block_height
                        step
* OutPut  : vector<mat>
* Return  :
* Other   :
* Author  : kebin peng %{CurrentDate:yyyy.MM.dd}
*******************************************************************************/
vector<Mat> slice_image(int width, int height, uchar* pimage, int block_width,
                        int block_height, int step) {
  Mat image = Mat(height, width, CV_8UC1, (void*)pimage);

  if (step > block_width) step = block_width;
  if (step > block_height) step = block_height;

  int row_num = (image.rows + step - 1) / step;  //�����з����ж��ٿ�
  int col_num = (image.cols + step - 1) / step;  //�����з����ж��ٿ�

  cout << "row_num: " << row_num << " ";
  cout << "col_num: " << col_num << endl;

  vector<Mat> vimgOut;

  for (int j = 0; j < row_num; j++) {
    for (int i = 0; i < col_num; i++) {
      int row_start = j * step;
      if (row_start + block_height >= image.rows) {
        row_start = image.rows - block_height;
      }

      int col_start = i * step;
      if (col_start + block_width >= image.cols) {
        col_start = image.cols - block_width;
      }

      Mat imageROI = image(Rect(col_start, row_start, block_width,
                                block_height));  // rect(x, y, width, height)
      vimgOut.push_back(imageROI);
    }
  }

  return vimgOut;
}

/******************************************************************************
* Function: merge image
* InPut   : image vector<mat>
            origin_width
            origin_height
            step
                        bi_threshold
* OutPut  : cv::mat
* Return  :
* Other   :
* Author  : kebin peng %{CurrentDate:yyyy.MM.dd}
*******************************************************************************/
Mat merge_image(vector<Mat>& image, int origin_width, int origin_height,
                int step, float bi_threshold) {
  Mat result(origin_height, origin_width, CV_8UC1, Scalar(0));

  int block_height = image[0].rows;
  int block_width = image[0].cols;

  if (step > block_width) step = block_width;
  if (step > block_height) step = block_height;

  int row_num = (origin_height + step - 1) / step;
  int col_num = (origin_width + step - 1) / step;

  // Mat mid = Mat(block_width, block_height, CV_8UC1, Scalar(0));

  for (int j = 0; j < row_num; j++) {
    for (int i = 0; i < col_num; i++) {
      int row_start = j * step;  //
      if (row_start + block_height >= origin_height) {
        row_start = origin_height - block_height;
      }

      int col_start = i * step;
      if (col_start + block_width >= origin_width) {
        col_start = origin_width - block_width;
      }

      Mat imageROI =
          result(Rect(col_start, row_start, block_width, block_height));

      Mat bi_img = image[j * col_num + i] > bi_threshold;
      Mat mid = (bi_img | imageROI);

      mid.copyTo(imageROI);
    }
  }

  return result;
}
