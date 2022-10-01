#pragma once

#include<iostream>
#include"stdafx.h"
#include "opencv2/opencv.hpp"
using namespace std;

typedef unsigned char uchar;

struct PARAM_MODEL
{
	char* filename_model;
	char* filename_image_mean;

	char* input_tensor_name;
	char* output_tensor_name;

	int input_width, input_height, input_nband;
	bool flag_model_encoded;
};

int load_model(void*& pmodel, PARAM_MODEL& param_model);
int detect(void* pmodel, vector<cv::Mat>& v_input_image, vector<cv::Mat>& v_output_image);
int release_model(void* pmodel);
