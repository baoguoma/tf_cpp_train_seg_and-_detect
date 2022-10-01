// tfTest.cpp :
#include <time.h>

#include <iostream>

#include "..\tf_detect\DL_detect.h"
#include "..\tf_detect\water_shed.h"
#include "opencv2/opencv.hpp"
#include "stdafx.h"
#include "water_shed_test.h"

#ifdef TF_GPU
#pragma comment(lib, "..\\x64\\Release\\DL_detect.lib")
#else
#pragma comment(lib, "..\\x64\\Release_CPU\\DL_detectCPU.lib")
#endif

using namespace std;
using namespace cv;

int main() {
  char* image_name[] = {"0.jpg", "1.jpg", "2.jpg"};
  void* p;

  void* pmodel = NULL;
  DL_init(&pmodel);  //

  for (int x = 0; x < 1; x++) {
    Mat test = imread(image_name[x % 3], IMREAD_GRAYSCALE);
    int* plabel = new int[test.cols * test.rows];
    memset(plabel, 0, sizeof(int) * test.cols * test.rows);

    uchar* image_data = test.data;
    clock_t start = clock();
    DL_detect(pmodel, test.cols, test.rows, image_data, plabel, 500, 10,
              0.3);  //
    clock_t end = clock();
    cout << "Running Time : " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    //

    Mat result0(test.rows, test.cols, CV_32F, plabel);
    Mat watershedImage(result0.size(), CV_8UC1);
    for (int i = 0; i < result0.rows; i++) {
      for (int j = 0; j < result0.cols; j++) {
        int index = result0.at<int>(i, j);
        if (index == 0)
          watershedImage.at<uchar>(i, j) = 0;

        else if (index >= 128)
          watershedImage.at<uchar>(i, j) = 255;
      }

      imwrite("result2.bmp", watershedImage);
    }

    delete[] plabel;
    plabel = NULL;
  }
  DL_close(pmodel);

  return 0;
}