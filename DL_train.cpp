#include "DL_train.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "stdafx.h"
#include "tf_train.h"

using namespace std;
using namespace cv;

struct model_ptr {
  void* ptr1;
  void* ptr2;
};

const int net_width = 512;
const int net_height = 512;
const int net_input_nband = 1;
const float net_mean_value = 0.38;

int encode_model_file(char* filename_model, char* filename_encode_model,
                      int n_key = 0, char* skey = nullptr, int n_siv = 0,
                      char* siv = nullptr);
int decode_model(char* filename_encode_model, uint64_t& n_data_decode,
                 char*& p_data_decode, int n_key = 0, char* skey = nullptr,
                 int n_siv = 0, char* siv = nullptr);
void write2File(const std::string path, char* buf, size_t len);

/******************************************************************************
 * Function: encpyt model
 * InPut   :
 * OutPut  :
 * Return  :
 * Other   :
 * Author  : kebin peng %{CurrentDate:yyyy.MM.dd}
 *******************************************************************************/
int cpytomodel() {
  encode_model_file("unet.pb", "unet_membrane.enc");

  uint64_t nsize = 0;
  char* pdata = nullptr;
  decode_model("unet_membrane.enc", nsize, pdata);

  write2File("unet_membrane.pb2", pdata, nsize);

  free(pdata);

  encode_model_file("unet_side.pb", "unet_side_membrane.enc");

  nsize = 0;
  pdata = nullptr;
  decode_model("unet_side_membrane.enc", nsize, pdata);

  write2File("unet_side_membrane.pb2", pdata, nsize);

  free(pdata);

  return 0;
}

/******************************************************************************
 * Function: initial tensorflow model
 * InPut   : pointer to the model
 * OutPut  :
 * Return  :
 * Other   :
 * Author  : kebin peng
 *******************************************************************************/
int DL_init(void** ppmodel) {
  //��ʼ��ģ��
  void* pmodel = nullptr;
  PARAM_MODEL p_init;
  p_init.filename_model = "graph.pb";  //"tf-unet.pb";
  p_init.filename_image_mean = "";
  p_init.input_tensor_name = "input_1:0";
  p_init.output_tensor_name = "conv2d_24/Sigmoid:0";  //"conv2d_23/truediv:0";
  p_init.input_width = net_width;
  p_init.input_height = net_height;
  p_init.input_nband = net_input_nband;
  p_init.flag_model_encoded = 0;
  p_init.mean_value[0] = net_mean_value;

  load_model(pmodel, p_init);
  *ppmodel = pmodel;

  return 0;
}

/******************************************************************************
* Function: training tensorflow by c++
* InPut   : pointer to the model
*           width
*           height
*           pimgָ pointer to image
*           plabel
*           step_size
*           nbatchsize
*           bi_thres_hold
*           mean_value
*           watershed_thres_hold

* OutPut  :
* Return  :
* Other   :
* Author  : kebin peng
*******************************************************************************/
int DL_train(void* pmodel, vector<Mat> pimg, vector<Mat> plabel, int batchsize,
             int epoch) {
  train(pmodel, pimg, plabel, batchsize, epoch, 0);
  return 0;
}

/******************************************************************************
 * Function: save tensorflow model
 * InPut   : pointer to the model
 * OutPut  :
 * Return  :
 * Other   :
 * Author  : kebin peng %{CurrentDate:yyyy.MM.dd}
 *******************************************************************************/
int DL_close(void* pmodel) {
  release_model(pmodel);
  return 0;
}
