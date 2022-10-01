#pragma once

#include"tf_detect.h"
#include"stdafx.h"

#ifdef TF_DLL_EXPORT
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __declspec(dllimport)
#endif



extern "C"
{

	DLL_EXPORT int DL_init(void** ppmodel);


	DLL_EXPORT int DL_detect(void* pmodel, int width, int height, unsigned char* pimg, int* plabel,
		                     int step_size, int nbatchsize, float bi_thres_hold);


	DLL_EXPORT int DL_close(void* pmodel);


}

