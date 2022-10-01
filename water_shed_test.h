#pragma once

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include<iostream>
using namespace std;


cv::Mat seg_watershed(int width, int high, uchar* src_image, uchar* border_image);

cv::Mat merge_image(vector<cv::Mat> image, int origin_width, int origin_height, int step, float bi_thres_hold);

vector<cv::Mat> slice_image(uchar* pimage, int block_width, int block_height, int origin_width, int origin_height, int step);