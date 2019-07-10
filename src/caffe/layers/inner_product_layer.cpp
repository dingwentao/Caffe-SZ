#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "../SZ/sz/include/sz.h"
#include "../SZ/sz/include/rw.h"

size_t r5=0,r4=0,r3=0,r2=0,r1=0;
size_t outSize=0;
int time_tool_inner = 0;

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  pruning_coeff_ = this->layer_param_.pruning_param().coeff();
  CHECK_GE(pruning_coeff_, 0);
  CHECK_GT(1, pruning_coeff_);
  pruned_ = (pruning_coeff_ == 0);
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Deep compression pruning
    if (pruning_coeff_ > 0) {
      masks_.resize(this->blobs_.size());
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    if (pruning_coeff_ != 0) {
      masks_[0].reset(new Blob<Dtype>(weight_shape));
      caffe_set<Dtype>(this->blobs_[0]->count(), (Dtype)1.,
          masks_[0]->mutable_cpu_data());
    }
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
      if (pruning_coeff_ != 0) {
        masks_[1].reset(new Blob<Dtype>(bias_shape));
        caffe_set<Dtype>(this->blobs_[1]->count(), (Dtype)1.,
            masks_[1]->mutable_cpu_data());
      }
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    //SZ_Init("../SZ/example/sz.config");
    //r1 = this->blobs_[0]->count();
    //float temp[r1];
    //for (int i = 0; i < r1; i++) {
    //    temp[i] = this->blobs_[0]->mutable_cpu_data()[i];
    //}
    //float *data = temp;
    //unsigned char *bytes = SZ_compress(SZ_FLOAT, &temp[0], &outSize, r5, r4, r3, r2, r1);    
    //r1 = this->blobs_[0]->count();
    //void *decData = SZ_decompress(SZ_FLOAT, bytes, outSize, r5, r4, r3, r2, r1);
    //float *decData2 = (float *)decData;
    //SZ_decompress(SZ_FLOAT, bytes, outSize, r5, r4, r3, r2, r1);
    //printf("weights[0] = %f %f \n", this->blobs_[0]->mutable_cpu_data()[2], *(float*)(decData2+2));
    //for (int i = 0; i < r1; i++) {
    //    this->blobs_[0]->mutable_cpu_data()[i] = *(float*)(decData2+i);
    //}
    //free(bytes);
    //free(decData);
    //free(decData2);
    //free(data);
    //SZ_Finalize();

  //printf("weights[0-2] = %f %f %f \n", this->blobs_[0]->mutable_cpu_data()[0], this->blobs_[0]->mutable_cpu_data()[1], this->blobs_[0]->mutable_cpu_data()[2]);

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  // prune only once after loading the caffemodel
  if (!pruned_) {
    caffe_cpu_prune(this->blobs_[0]->count(), pruning_coeff_,
          this->blobs_[0]->mutable_cpu_data(), masks_[0]->mutable_cpu_data());
    if (bias_term_) {
      caffe_cpu_prune(this->blobs_[1]->count(), pruning_coeff_,
          this->blobs_[1]->mutable_cpu_data(), masks_[1]->mutable_cpu_data());
    }
    pruned_ = true;
  }
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
 
    if (bias_term_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
          bias_multiplier_.cpu_data(),
          this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
    }

    /*if (N_ > 100) { 
     SZ_Init("../SZ/example/sz.config");
     r1 = N_*M_;
     float temp[r1];
     //printf("test = %d %d %d\n", M_, N_, K_);
    //if (r1 == 400000)     
    //    top_data[300000] = 0.00;
     for (int i = 0; i < r1; i++) {
         temp[i] = top_data[i];
     }
     //float *data = temp;
     unsigned char *bytes = SZ_compress(SZ_FLOAT, &temp[0], &outSize, r5, r4, r3, r2, r1);    
     //r1 = N_;i
     time_tool_inner += 1;
     if (time_tool_inner % 100 == 0)
       printf("Current compression ratio of fc layers = %d to %d\n", r1*32, outSize);
     void *decData = SZ_decompress(SZ_FLOAT, bytes, outSize, r5, r4, r3, r2, r1);
     float *decData2 = (float *)decData;
     //SZ_decompress(SZ_FLOAT, bytes, outSize, r5, r4, r3, r2, r1);
     //printf("weights[0] = %f %f \n", this->blobs_[0]->mutable_cpu_data()[2]    , *(float*)(decData2+2));
     for (int i = 0; i < r1; i++) {
        //top_data[i] = *(float*)(decData2+i);
        top[0]->mutable_cpu_data()[i] = *(float*)(decData2+i);
     }
     free(bytes);
     //free(decData);
     free(decData2);
     //free(data);

     SZ_Finalize();
     //printf("compression ratio of fc layers = %f\n", (r1/outSize));
    }*/
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    
     //SZ_Init("../SZ/example/sz.config");
     //r1 = 40000;
     //float temp[r1];
     //for (int i = 0; i < r1; i++) {
     //    temp[i] = bottom[0]->cpu_data()[i];
     //}
     //unsigned char *bytes = SZ_compress(SZ_FLOAT, &temp[0], &outSize, r5, r4,     r3, r2, r1);    
     //r1 = 40000;
     //void *decData = SZ_decompress(SZ_FLOAT, bytes, outSize, r5, r4, r3, r2,     r1);
     //float *decData2 = (float *)decData;
     //printf("weights[0] = %d %f %f \n", r1, bottom[0]->cpu_data()[2], *(float*)(decData2+2));
    //double temp2[r1]; 
     //for (int i = 0; i < r1; i++) {
     //    temp[i] = *(float*)(decData2+i);
     //}
     const Dtype* bottom_data = bottom[0]->cpu_data();
     Dtype *p_var = NULL;
     p_var = const_cast <Dtype*>(bottom_data);
 
SZ_Init("../SZ/example/sz.config");
r1 = K_;
r2 = M_;
float temp[K_*M_];
//printf("test = %d %d %d\n", M_, N_, K_);
for (int i = 0; i < K_*M_; i++) {
  temp[i] = p_var[i];
}
unsigned char *bytes = SZ_compress(SZ_FLOAT, &temp[0], &outSize, r5, r4, r3, r2, r1);    
//r1 = N_;i
time_tool_inner += 1;
if (time_tool_inner % 100 == 0 || time_tool_inner % 100 == 1)
  printf("Current compression ratio of fc layers = %d to %d\n", K_*M_/250, outSize/1000);
void *decData = SZ_decompress(SZ_FLOAT, bytes, outSize, r5, r4, r3, r2, r1);
float *decData2 = (float *)decData;
for (int i = 0; i < K_*M_; i++) {
  //top_data[i] = *(float*)(decData2+i);
  p_var[i] = *(float*)(decData2+i);
}
free(bytes);
//free(decData);
free(decData2);
//free(data);

SZ_Finalize();


     //for (int i = 0; i < M_*K_; i++)
     //  *(p_var+i) = 0.0;
     //printf("test = %d %d %d\n", M_, K_, N_);
     //free(bytes);
     //free(decData2);
     //SZ_Finalize();

    //const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
  if (pruning_coeff_ > 0) {
    if (this->param_propagate_down_[0]) {
      caffe_mul(this->blobs_[0]->count(), this->blobs_[0]->cpu_diff(),
          masks_[0]->cpu_data(), this->blobs_[0]->mutable_cpu_diff());
    }
    if (bias_term_ && this->param_propagate_down_[1]) {
      caffe_mul(this->blobs_[1]->count(), this->blobs_[1]->cpu_diff(),
          masks_[1]->cpu_data(), this->blobs_[1]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
