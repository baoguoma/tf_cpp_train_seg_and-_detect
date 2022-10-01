// TfDetect.cpp;
//

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#define COMPILER_MSVC
#define NOMINMAX

#include "DL_train.h"
#include "google/protobuf/message.h"
#include "stdafx.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/data_flow_ops.h"
#include "tensorflow/cc/ops/io_ops.h"
#include "tensorflow/cc/ops/parsing_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tensorflow/core/public/session.h"
#include "tf_train.h"

#ifdef TF_GPU
#pragma comment(lib, "tensorflow.lib")
#else
#pragma comment(lib, "tensorflowCPU.lib")
#endif

#pragma comment(lib, "opencv_world341.lib")
#pragma comment(lib, "libprotobuf.lib")

typedef std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_dict;

using namespace std;
using namespace cv;
using namespace tensorflow;

int encode_model_file(char* filename_model, char* filename_encode_model,
                      int n_key = 0, char* skey = nullptr, int n_siv = 0,
                      char* siv = nullptr);
int decode_model(char* filename_encode_model, uint64_t& n_data_decode,
                 char*& p_data_decode, int n_key = 0, char* skey = nullptr,
                 int n_siv = 0, char* siv = nullptr);

struct TFMODEL {
  Session* session;
  GraphDef graphdef;

  string input_tensor_name;
  string output_tensor_name;

  int input_width, input_height, input_band;
  float mean_value[4];

  Mat* img_mean;
};
/******************************************************************************
 * Function: readPB model
 * InPut   : ָref to the model
 * OutPut  : void
 * Return  :
 * Other   :
 * Author  : kebin peng %{CurrentDate:yyyy.MM.dd}
 *******************************************************************************/
void readPB(GraphDef& graph_def) {
  int i;

  graph_def.node(0).PrintDebugString();
  graph_def.node(212).PrintDebugString();
  graph_def.node(213).PrintDebugString();
  graph_def.node(214).PrintDebugString();
  graph_def.node(215).PrintDebugString();
  int x;
  cin >> x;

  for (i = 0; i < graph_def.node_size(); i++) {
    // if (graph_def.node(i).name() == "input_1")
    { graph_def.node(i).PrintDebugString(); }
  }
}

/******************************************************************************
 * Function: initial tf model
 * InPut   : TFMODEL
 * OutPut  :
 * Return  :
 * Other   :
 * Author  : kebin peng %{CurrentDate:yyyy.MM.dd}
 *******************************************************************************/

int init_tf_session(TFMODEL*& pmodel) {
  pmodel = new TFMODEL;

  //- create session;
  Status status_sess_new = NewSession(SessionOptions(), &(pmodel->session));
  if (!status_sess_new.ok()) {
    cout << "[]Error: creating a session: " << status_sess_new.ToString()
         << endl;
    return 1;
  }

  cout << "[]Success: created session;" << endl;

  //- init;
  pmodel->input_width = pmodel->input_height = pmodel->input_band = 0;
  pmodel->img_mean = nullptr;

  return 0;
}

/******************************************************************************
 * Function: load tf model
 * InPut   : pmodel
 *           filename_model
 *           filename_image_mean
 *           flag_model_encoded
 * OutPut  :
 * Return  :
 * Other   :
 * Author  : kebin peng %{CurrentDate:yyyy.MM.dd}
 *******************************************************************************/

int load_tf_model(TFMODEL*& pmodel, char* filename_model,
                  bool flag_model_encoded = false) {
  pmodel = new TFMODEL;

  //- create session;
  Status status_sess_new = NewSession(SessionOptions(), &(pmodel->session));
  if (!status_sess_new.ok()) {
    cout << "[]Error: creating a session: " << status_sess_new.ToString()
         << endl;
    return 1;
  }

  cout << "[]Success: created session;" << endl;

  //- init;
  pmodel->input_width = pmodel->input_height = pmodel->input_band = 0;
  pmodel->img_mean = nullptr;

  //- load graphdef;
  // GraphDef graphdef;

  if (!flag_model_encoded)  // not encoded;
  {
    Status status_load =
        ReadBinaryProto(Env::Default(), filename_model, &(pmodel->graphdef));

    if (!status_load.ok()) {
      cout << "[]Error: loading model from file: " << status_load.ToString()
           << endl;

      return 2;
    }

  } else  // model is encoded;
  {
    uint64_t n_decode_model = 0;
    char* p_decode_model = nullptr;
    decode_model(filename_model, n_decode_model, p_decode_model);

    bool state =
        pmodel->graphdef.ParseFromArray(p_decode_model, n_decode_model);

    free(p_decode_model);

    if (!state) {
      cout << "[]Error: loading encoded model from file: " << filename_model
           << " ;" << endl;
      return 2;
    }
  }

  cout << "[]Success: loaded model from: " << filename_model << ";" << endl;

  // readPB(graphdef);

  //- create graph in session;
  // Status status_create = pmodel->session->Create(
  // *(pmodel->graphdef.mutable_graph_def())  );
  Status status_create = pmodel->session->Create(pmodel->graphdef);
  if (!status_create.ok()) {
    cout << "[]Error: creating graph in session: " << status_create.ToString()
         << endl;
    return 3;
  }

  cout << "[]Success: created graph in session;" << endl;
  /*


  */
  //- load image mean;
  /*
  if (filename_image_mean)
  {
          Mat img0 = imread(filename_image_mean, IMREAD_GRAYSCALE);


          if (img0.rows < 1)
          {
                  cout << "[]Error: loading image mean from: " <<
  filename_image_mean << ";" << endl; return 4;
          }

          pmodel->img_mean = new Mat;
          img0.convertTo(*(pmodel->img_mean), CV_32F, 1. / 255);

          cout << "[]Success: loaded image mean from: " << filename_image_mean
  << ";" << endl;
  }
  */
  return 0;
}
/******************************************************************************
* Function: object detection model
* InPut   : pmodel
            input_image
            v_output
* OutPut  : v_output
* Return  :
* Other   :
* Author  : kebin peng %{CurrentDate:yyyy.MM.dd}
*******************************************************************************/
int detect_tf_model(TFMODEL* pmodel, Tensor& input_image, Tensor& input_label) {
  /*Tensor input_tensor(DT_FLOAT, TensorShape({ 1, pmodel->input_height,
  pmodel->input_width, 1 })); float* pdata = input_tensor.flat<float>().data();

  Mat mat_tensor(pmodel->input_height, pmodel->input_width, CV_32FC1, pdata);
  v_input_image[0].convertTo(mat_tensor, CV_32FC1);*/

  // TF_CHECK_OK(pmodel->session->Run({}, {}, { "init_all_vars_op" }, nullptr));
  vector<Tensor> v_output;
  float cost;
  Status status_run;

  //

  vector<Tensor> w_1;
  TF_CHECK_OK(((TFMODEL*)pmodel)
                  ->session->Run({{"input/input_images", input_image},
                                  {"input/input_labels", input_label}},
                                 {"layer_1/init_w/w_1"}, {},
                                 &w_1));           // "init_all_vars_op"
  auto w_1_value = w_1[0].AsProtoTensorContent();  //   <float>()(0);
  cout << *(w_1_value.data() + 1) << endl;
  //

  TF_CHECK_OK(pmodel->session->Run(
      {{"input/input_images", input_image},
       {"input/input_labels", input_label}},  //"input","label
      {"softmax_loss/all_loss"},              //  "loss"
      {}, &v_output));

  cost = v_output[0].scalar<float>()(0);
  cout << "Loss: " << cost << " ";

  vector<Tensor> v_acc;
  TF_CHECK_OK(pmodel->session->Run({{"input/input_images", input_image},
                                    {"input/input_labels", input_label}},
                                   {"accuracy/acc_op"},  // "acc_op"
                                   {}, &v_acc));
  float acc;
  acc = v_acc[0].scalar<float>()(0);
  cout << "Acc: " << acc * 100 << "%" << endl;

  TF_CHECK_OK(pmodel->session->Run(
      {{"input/input_images", input_image},
       {"input/input_labels", input_label}},
      {}, {"Gradient_Descent/Optimizer"},  //,	 "Optimizer"
      nullptr));

  /*
  if (!status_run.ok())
  {
          cout << "[]Error: session running: " << status_run.ToString() << endl;
          return 1;
  }

  cout << "[]Success: run session for " << input_image.shape().DebugString()
          << " and output " << v_output[0].shape().DebugString() << ";" << endl;

*/
  return 0;
}

/******************************************************************************
* Function: trainfer cv_mat to tensor in tf
* InPut   : pmodelָ
            v_input_image
* OutPut  :
* Return  :
* Other   :
* Author  : kebin peng %{CurrentDate:yyyy.MM.dd}
*******************************************************************************/
Tensor imgToTensor(void* pmodel, vector<Mat> v_input_image, bool readLabel) {
  TFMODEL* model = (TFMODEL*)pmodel;

  int n_batch_size = v_input_image.size();

  //- input tensor;
  Tensor input_tensor(DT_FLOAT,
                      TensorShape({n_batch_size, model->input_height,
                                   model->input_width, model->input_band}));

  auto Eig_input_tensor = input_tensor.tensor<float, 4>();
  if (readLabel) {
    auto Eig_input_tensor = input_tensor.tensor<int, 4>();
  }

  for (int k = 0; k < n_batch_size; k++) {
    //-- image;
    Mat img;

    if (((TFMODEL*)pmodel)->input_height != v_input_image[k].rows ||
        ((TFMODEL*)pmodel)->input_width != v_input_image[k].cols) {
      resize(v_input_image[k], img,
             cv::Size(((TFMODEL*)pmodel)->input_height,
                      ((TFMODEL*)pmodel)->input_width));
    } else {
      img = v_input_image[k];
    }

    Mat img_norm;

    if (readLabel) {
      img.convertTo(img_norm, CV_32SC1, 1 / 255);
    } else {
      img.convertTo(img_norm, CV_32FC1, 1. / 255);
    }

    //-- abstract mean;
    if (!readLabel) {
      img_norm = img_norm - model->mean_value[0];
    }

    //-- concatenate image batch;

    if (readLabel) {
    } else {
      Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>> imgTensor(
          (float*)(img_norm.data), ((TFMODEL*)pmodel)->input_height,
          ((TFMODEL*)pmodel)->input_width, 1);
      Eig_input_tensor.chip(k, 0) = imgTensor;
    }
  }

  return input_tensor;
}

//------------------------------- interface
//------------------------------------------------;

/******************************************************************************
* Function: load tf model
* InPut   : pmodel
            para_model
* OutPut  :
* Return  :
* Other   :
* Author  : lichao %{CurrentDate:yyyy.MM.dd}
*******************************************************************************/
int load_model(void*& pmodel, PARAM_MODEL& param_model) {
  TFMODEL*& model = (TFMODEL*&)pmodel;

  int rv = load_tf_model(model, param_model.filename_model,
                         param_model.flag_model_encoded);
  if (rv) return rv;

  model->input_tensor_name = string(param_model.input_tensor_name);
  model->output_tensor_name = string(param_model.output_tensor_name);
  model->input_width = param_model.input_width;
  model->input_height = param_model.input_height;
  model->input_band = param_model.input_nband;
  model->mean_value[0] = param_model.mean_value[0];

  return 0;
}

void SaveOrRestore(TFMODEL* pmodel, const string& checkpoint_prefix,
                   const string& op_name) {
  Tensor t(tensorflow::DT_STRING, tensorflow::TensorShape());
  t.scalar<string>()() = checkpoint_prefix;
  TF_CHECK_OK(
      pmodel->session->Run({{"save/Const", t}}, {}, {op_name}, nullptr));
}

void Restore(TFMODEL* pmodel, const string& checkpoint_prefix) {
  SaveOrRestore(pmodel, checkpoint_prefix, "save/restore_all");

  /*
  vector<string> vNames;
  int node_count = pmodel->graphdef.node_size();


  for (int i = 0; i < node_count; i++)
  {
          auto n = pmodel->graphdef.node(i);
          if (n.name().find("nWeights") != std::string::npos)
          {
                  vNames.push_back(n.name());
          }
}

  vector<Tensor> nnOutput;
  TF_CHECK_OK( pmodel->session->Run({}, vNames, {}, &nnOutput));
*/
}

void Checkpoint(TFMODEL* pmodel, const string& checkpoint_prefix) {
  SaveOrRestore(pmodel, checkpoint_prefix, "save/control_dependency");
}

/******************************************************************************
* Function: train tf model
* InPut   : pmodel
            v_inut_image
* OutPut  :
* Return  :
* Other   :
* Author  : kebin peng %{CurrentDate:yyyy.MM.dd}
*******************************************************************************/
int train(void* pmodel, vector<Mat>& v_input_image, vector<Mat>& v_input_label,
          int batchsize, int epoch, bool restore) {
  //
  int j = 0;
  long long int k = 0;
  int start, end;
  int iter = (v_input_image.size() + batchsize - 1) / batchsize;

  vector<Mat> batch_image;
  vector<Mat> batch_label;

  start = 0;

  const string checkpoint_prefix = "./";
  if (!restore) {
    TF_CHECK_OK(((TFMODEL*)pmodel)
                    ->session->Run({}, {}, {"init/init_all_vars_op"},
                                   nullptr));  // "init_all_vars_op"
  } else {
    cout << "Restoring traning" << endl;
    const string checkpoint_prefix = "./";
    Restore((TFMODEL*)pmodel, checkpoint_prefix);
  }

  while (k < epoch) {
    if (k % 10 == 0 && k != 0) {
      const string checkpoint_prefix = "./";
      cout << "Saving checkpoint\n";
      Checkpoint((TFMODEL*)pmodel, checkpoint_prefix);

      /*
      string saveFileName = "unet-train" + to_string(k) + ".pb";
      WriteBinaryProto(Env::Default(), saveFileName,
      ((TFMODEL*)pmodel)->graphdef);
  */

      /*
      tensorflow::Tensor checkpointPathTensor(tensorflow::DT_STRING,
      tensorflow::TensorShape()); checkpointPathTensor.scalar<std::string>()() =
      "./";


      TFMODEL* model = (TFMODEL*)pmodel;


      tensor_dict feed_dict = { {
      model->graphdef.mutable_saver_def()->filename_tensor_name() ,
      checkpointPathTensor }, }; Status status =
      ((TFMODEL*)pmodel)->session->Run(feed_dict,
      {},
      { model->graphdef.mutable_saver_def()->save_tensor_name() },
              nullptr);
      */
    }
    j = 0;
    while (j < iter) {
      if ((start + batchsize) < v_input_image.size())
        end = start + batchsize;
      else
        end = v_input_image.size();

      for (int i = start; i < end; i++) {
        batch_image.push_back(v_input_image[i]);
        batch_label.push_back(v_input_image[i]);
      }
      start = end;

      //���end == v_input_image.size() �����Ѿ�����ѵ���Ѿ��������ݼ������һ��batch
      //��һ��ѵ��Ӧ�����¿�ʼ�����԰�start����Ϊ0
      if (end == v_input_image.size()) start = 0;

      // Tensor input_tensor_image = imgToTensor(pmodel, batch_image,false);
      // Tensor input_tensor_label = imgToTensor(pmodel, batch_label,false);

      Tensor input_tensor_image(DT_FLOAT, TensorShape({1, 512, 512, 1}));
      Tensor input_tensor_label(DT_INT32, TensorShape({1, 512, 512}));

      cout << "Epoch: " << k << " ";
      int rv = detect_tf_model((TFMODEL*)pmodel, input_tensor_image,
                               input_tensor_label);
      if (rv) {
        return 1;  // session run failed;
      }

      batch_image.clear();
      batch_label.clear();
      j++;
    }
    k++;
  }
  return 0;
}

/******************************************************************************
 * Function: release tf model
 * InPut   : pmodel
 * OutPut  :
 * Return  :
 * Other   :
 * Author  : kebin peng %{CurrentDate:yyyy.MM.dd}
 *******************************************************************************/
int release_model(void* pmodel) {
  ((TFMODEL*)pmodel)->session->Close();
  delete ((TFMODEL*)pmodel)->img_mean;

  return 0;
}
