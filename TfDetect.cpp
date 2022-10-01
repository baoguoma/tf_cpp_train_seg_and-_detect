// TfDetect.cpp;
//

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

#define COMPILER_MSVC
#define NOMINMAX

#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/public/session.h"
#include "google/protobuf/message.h"
#include "tensorflow/cc/ops/io_ops.h"
#include "tensorflow/cc/ops/data_flow_ops.h"
#include "tensorflow/cc/ops/parsing_ops.h"
#include "tensorflow/cc/framework/scope.h"

#pragma comment(lib, "opencv_world341.lib")
#pragma comment(lib, "libprotobuf.lib")

#ifdef TF_CPU
	#pragma comment(lib, "tensorflowCPU.lib")
#else
	#pragma comment(lib, "tensorflow.lib")
#endif

using namespace std;
using namespace cv;
using namespace tensorflow;

int encode_model_file(char* filename_model, char* filename_encode_model,
	int n_key = 0, char* skey = nullptr, int n_siv = 0, char* siv = nullptr);
int decode_model(char* filename_encode_model, uint64_t& n_data_decode, char*& p_data_decode,
	int n_key = 0, char* skey = nullptr, int n_siv = 0, char* siv = nullptr);

struct TFMODEL
{
	Session* session;

	string input_tensor_name;
	string output_tensor_name;

	int input_width, input_height, input_band;

	Mat* img_mean;
};

void readPB(GraphDef& graph_def)
{
	int i;

	graph_def.node(220).PrintDebugString();
	int x;
	cin >> x;

	for (i = 0; i < graph_def.node_size(); i++)
	{
		//if (graph_def.node(i).name() == "input_1")
		{
			graph_def.node(i).PrintDebugString();
		}
	}
}

int init_tf_session(TFMODEL*& pmodel)
{
	pmodel = new TFMODEL;

	//- create session;
	Status status_sess_new = NewSession(SessionOptions(), &(pmodel->session));
	if (!status_sess_new.ok())
	{
		cout << "[]Error: creating a session: " << status_sess_new.ToString() << endl;
		return 1;
	}

	cout << "[]Success: created session;" << endl;

	//- init;
	pmodel->input_width = pmodel->input_height = pmodel->input_band = 0;
	pmodel->img_mean = nullptr;

	return 0;
}

int load_tf_model(TFMODEL*& pmodel, char* filename_model, char* filename_image_mean, bool flag_model_encoded = false)
{
	pmodel = new TFMODEL;

	//- create session;
	Status status_sess_new = NewSession(SessionOptions(), &(pmodel->session));
	if (!status_sess_new.ok())
	{
		cout << "[]Error: creating a session: " << status_sess_new.ToString() << endl;
		return 1;
	}

	cout << "[]Success: created session;" << endl;

	//- init;
	pmodel->input_width = pmodel->input_height = pmodel->input_band = 0;
	pmodel->img_mean = nullptr;


	//- load graphdef;
	GraphDef graphdef;

	

	if (!flag_model_encoded)  //not encoded;
	{
		
		Status status_load = ReadBinaryProto(Env::Default(), filename_model, &graphdef);
		if (!status_load.ok())
		{
			cout << "[]Error: loading model from file: " << status_load.ToString() << endl;
			return 2;
		}
	}
	else  //model is encoded;
	{
		uint64_t n_decode_model = 0;
		char* p_decode_model = nullptr;
		decode_model(filename_model, n_decode_model, p_decode_model);

		bool state = graphdef.ParseFromArray(p_decode_model, n_decode_model);

		free(p_decode_model);

		if (!state)
		{
			cout << "[]Error: loading encoded model from file: " << filename_model << " ;" << endl;
			return 2;
		}
	}

	cout << "[]Success: loaded model from: " << filename_model << ";" << endl;

	//readPB(graphdef);

	//- create graph in session;
	Status status_create = pmodel->session->Create(graphdef);
	if (!status_create.ok())
	{
		cout << "[]Error: creating graph in session: " << status_create.ToString() << endl;
		return 3;
	}

	cout << "[]Success: created graph in session;" << endl;

	//- load image mean;
	if (filename_image_mean)
	{
		Mat img0 = imread(filename_image_mean);

		if (img0.rows < 1)
		{
			cout << "[]Error: loading image mean from: " << filename_image_mean << ";" << endl;
			return 4;
		}

		pmodel->img_mean = new Mat;
		img0.convertTo(*(pmodel->img_mean), CV_32F, 1. / 255);

		cout << "[]Success: loaded image mean from: " << filename_image_mean << ";" << endl;
	}

	return 0;
}

int detect_tf_model(TFMODEL* pmodel, Tensor& input_image, vector<Tensor>& v_output)
{
	/*Tensor input_tensor(DT_FLOAT, TensorShape({ 1, pmodel->input_height, pmodel->input_width, 1 }));
	float* pdata = input_tensor.flat<float>().data();

	Mat mat_tensor(pmodel->input_height, pmodel->input_width, CV_32FC1, pdata);
	v_input_image[0].convertTo(mat_tensor, CV_32FC1);*/
	
	Status status_run = pmodel->session->Run({ { pmodel->input_tensor_name, input_image } },
	{ pmodel->output_tensor_name },
	{},
		&v_output);

	if (!status_run.ok())
	{
		cout << "[]Error: session running: " << status_run.ToString() << endl;
		return 1;
	}

	cout << "[]Success: run session for " << input_image.shape().DebugString()
		<< " and output " << v_output[0].shape().DebugString() << ";" << endl;
	

	

	return 0;
}

//------------------------------- interface ------------------------------------------------;

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


int load_model(void*& pmodel, PARAM_MODEL& param_model)
{
	TFMODEL*& model = (TFMODEL*&)pmodel;

	int rv = load_tf_model(model, param_model.filename_model, param_model.filename_image_mean, param_model.flag_model_encoded);
	if (rv) return rv;

	model->input_tensor_name = string(param_model.input_tensor_name);
	model->output_tensor_name = string(param_model.output_tensor_name);
	model->input_width = param_model.input_width;
	model->input_height = param_model.input_height;
	model->input_band = param_model.input_nband;

	return 0;
}


int detect(void* pmodel, vector<Mat>& v_input_image, vector<Mat>& v_output_image)
{
	TFMODEL* model = (TFMODEL*)pmodel;

	int n_batch_size = v_input_image.size();

	//- input tensor;
	Tensor input_tensor(DT_FLOAT, TensorShape({ n_batch_size, model->input_height, model->input_width, model->input_band }));
	auto Eig_input_tensor = input_tensor.tensor<float, 4>();

	for (int k = 0; k < n_batch_size; k++)
	{
		//-- image;
		Mat img;
		resize(v_input_image[k], img, cv::Size(((TFMODEL*)pmodel)->input_height, ((TFMODEL*)pmodel)->input_width));
		img.convertTo(img, CV_32FC1, 1. / 255);

		//-- abstract mean;
		if (((TFMODEL*)pmodel)->img_mean)
		{
			img = img - *((TFMODEL*)pmodel)->img_mean;
		}

		//-- concatenate image batch;
		Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>> imgTensor((float*)(img.data), ((TFMODEL*)pmodel)->input_height, ((TFMODEL*)pmodel)->input_width, 1);
		Eig_input_tensor.chip(k, 0) = imgTensor;
	}

	//- detecting;
	vector<Tensor> v_output;
	int rv = detect_tf_model((TFMODEL*)pmodel, input_tensor, v_output);
	if (rv)
	{
		return 1;	//session run failed;
	}

	//- output;
	TensorShape shape = v_output[0].shape();
	//cout << shape.DebugString();

	Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>, Eigen::Aligned16> output_tensor = v_output[0].tensor<float, 4>();
	Eigen::Tensor<float, 3, Eigen::RowMajor> prob_image;

	//-- for each output of the batch;
	for (int k = 0; k < output_tensor.dimension(0); k++)
	{
		prob_image = output_tensor.chip(k, 0); //for each input image;

		int outW = prob_image.dimension(0);
		int outH = prob_image.dimension(1);
		int outband = prob_image.dimension(2);

		//Mat* mat_output = new Mat(outH, outW, CV_32FC(outband));
		Mat mat_output(outH, outW, CV_32FC(outband));
		memcpy(mat_output.data, prob_image.data(), outW * outH * outband * sizeof(float));

		v_output_image.push_back(Mat(mat_output));

		//--;
		/*Eigen::Tensor<float, 2, Eigen::RowMajor> prob_class_0 = prob_image.chip(0, 2);  //class 0;

		Mat output_prb_0(prob_class_0.dimension(0), prob_class_0.dimension(1), CV_32FC1, prob_class_0.data());
		output_prb_0.convertTo(output_prb_0, CV_8UC1, 255.);
		imwrite("c0.tif", output_prb_0);*/
	}

	return 0;
}


int detect(void* pmodel, int n_batch_size, char** input_filename_images, vector<Mat>& v_output_image)
{
	vector<Mat> v_input_mat;

	for (int k = 0; k < n_batch_size; k++)
	{
		Mat img = imread(input_filename_images[k], IMREAD_GRAYSCALE);
		v_input_mat.push_back(Mat(img));
	}

	detect(pmodel, v_input_mat, v_output_image);

	return 0;
}

int release_model(void* pmodel)
{
	((TFMODEL*)pmodel)->session->Close();
	delete ((TFMODEL*)pmodel)->img_mean;

	return 0;
}


//-------------- test -----------------------------;
void write2File(const std::string path, char* buf, size_t len);

int test_tf_detect()
{
	/*{//cpypto model;

	encode_model_file("unet_membrane.pb",
	"unet_membrane.enc");

	uint64_t nsize = 0;
	char* pdata = nullptr;
	decode_model("unet_membrane.enc", nsize, pdata);

	write2File("unet_membrane.pb2", pdata, nsize);

	free(pdata);

	return 0;
	}*/

	void* pmodel = nullptr;

	PARAM_MODEL param_model;
	param_model.filename_model = "unet.pb";
	param_model.filename_image_mean = NULL;
	param_model.input_tensor_name = "input_1:0";
	param_model.output_tensor_name = "conv2d_24/Sigmoid:0";
	param_model.input_width = 512;
	param_model.input_height = 512;
	param_model.input_nband = 1;
	param_model.flag_model_encoded = false;

	load_model(pmodel, param_model);
	
	char* imgs[] = { "0.png", "0.png" };
	vector<Mat> v_output;
	detect(pmodel, 2, imgs, v_output);

	if (!v_output.empty())
	{
		vector<Mat> v_prob_class;
		split(v_output[0], v_prob_class);
		v_prob_class[0].convertTo(v_prob_class[0], CV_8UC1, 255.);
		imwrite("c0.tif", v_prob_class[0]);
	}

	//mode 2
	
	void* pmodel_side = nullptr;

	PARAM_MODEL param_model_side;
	param_model_side.filename_model = "unet_side.pb";
	param_model_side.filename_image_mean = NULL;
	param_model_side.input_tensor_name = "input_1:0";
	param_model_side.output_tensor_name = "conv2d_24/Sigmoid:0";
	param_model_side.input_width = 512;
	param_model_side.input_height = 512;
	param_model_side.input_nband = 1;
	param_model_side.flag_model_encoded = false;

	load_model(pmodel_side, param_model);

	//char* imgs[] = { "0.png", "0.png" };
	vector<Mat> v_output_side;
	detect(pmodel_side, 2, imgs, v_output);

	if (!v_output.empty())
	{
		vector<Mat> v_prob_class;
		split(v_output_side[0], v_prob_class);
		v_prob_class[0].convertTo(v_prob_class[0], CV_8UC1, 255.);
		imwrite("c0_side.tif", v_prob_class[0]);
	}
	
	release_model(pmodel);
	release_model(pmodel_side);

	return 0;
}