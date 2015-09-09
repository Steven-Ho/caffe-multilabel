#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithMaskLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.hierarchical_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  // sum_multiplier_.Reshape(mult_dims);
  // Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  // caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  // scale_dims[softmax_axis_] = 1;
  label_num_ = this->layer_param_.hierarchical_param().tree_size() / 2;
  vector<int> split_dims = bottom[0]->shape();
  split_dims.clear();
  split_dims.push_back(2);
  split_dims.push_back(label_num_);
  split_.Reshape(split_dims);
  scale_.Reshape(scale_dims);
  Dtype* split_data = split_.mutable_cpu_data();
  for (int i = 0; i < label_num_; i++) {
    split_data[2 * i] = this->layer_param_.hierarchical_param().tree(2 * i);
    split_data[2 * i + 1] = this->layer_param_.hierarchical_param().tree(2 * i + 1);
  }
}

template <typename Dtype>
void SoftmaxWithMaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  // const Dtype* label_vec = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  const Dtype* split_data = split_.cpu_data();
  int channels = bottom[0]->shape(softmax_axis_);
  int dim = bottom[0]->count() / outer_num_;
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    /*caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }*/
    // below assume inner_num=1
    // max
    caffe_copy(dim, bottom_data + i * dim, scale_data);
    for (int j = 0; j < label_num_; j++) {
      int start = split_data[2 * j];
      int end = split_data[2 * j + 1];
      Dtype dmax = bottom_data[i * dim + start];
      for (int k = start; k <= end; k++) {
        dmax = std::max(dmax, bottom_data[i * dim + k]);
      }
      for (int k = start; k <= end; k++) {
        scale_data[k] = dmax;
      }      
    }
    Dtype* mult = new Dtype(1);
    // subtraction
    /*caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);*/
    // subtract
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, 1, 1, -1.,
        scale_data, mult, 1., top_data);
    // exponentiation
    // caffe_exp<Dtype>(dim, top_data, top_data);
    // exp
    caffe_exp<Dtype>(dim, top_data, top_data);
    // sum after exp
    /*caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);*/
    // sum
    for (int j = 0; j < label_num_; j++) {
      int start = split_data[2 * j];
      int end = split_data[2 * j + 1];
      Dtype dsum = 0;
      for (int k = start; k <= end; k++) {
        dsum += top_data[k];
      }
      for (int k = start; k <= end; k++) {
        scale_data[k] = dsum;
      }
    }    
    // division
    /*for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
    }*/
    // divide
    caffe_div(dim, top_data, scale_data, top_data);
    top_data += dim;   
  }
}

template <typename Dtype>
void SoftmaxWithMaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithMaskLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithMaskLayer);

}  // namespace caffe
