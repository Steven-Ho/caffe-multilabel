#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void HierarchicalSoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("SoftmaxWithMask");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  // softmax_bottom_vec_.push_back(bottom[1]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();

  vector<int> split_dims = bottom[0]->shape();
  split_dims.clear();
  split_dims.push_back(2);
  split_dims.push_back(label_num_);
  split_.Reshape(split_dims);
  Dtype* split_data = split_.mutable_cpu_data();
  for (int i = 0; i < label_num_; i++) {
    split_data[2 * i] = this->layer_param_.hierarchical_param().tree(2 * i);
    split_data[2 * i + 1] = this->layer_param_.hierarchical_param().tree(2 * i + 1);
  }
}

template <typename Dtype>
void HierarchicalSoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.hierarchical_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  /*CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";*/
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void HierarchicalSoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* split_data = split_.cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;

  LOG(INFO) << "dim: " << dim << ", labelsize: " << bottom[1]->count();
  for (int i = 0; i < outer_num_; i++) {
    for (int j = 0; j < label_num_; j++) {
      int start = split_data[2 * j];
      int end = split_data[2 * j + 1];
      for (int k = start; k <= end; k++) {
        if (label[i * dim + k] == 1) {
          loss -= log(std::max(prob_data[i * dim + k], Dtype(FLT_MIN)));
          count++;
          break;
        }
      }
    }
  }

  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = loss / count;
  } else {
    top[0]->mutable_cpu_data()[0] = loss / outer_num_;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void HierarchicalSoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* split_data = split_.cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;

    for (int i = 0; i < outer_num_; i++) {
      for (int j = 0; j < label_num_; j++) {
        int start = split_data[2 * j];
        int end = split_data[2 * j + 1];
        bool flag = false;
        int index = 0;
        for (int k = start; k <= end; k++) {
          if (label[i * dim + k] == 1) {
            flag = true;
            index = k;
            break;
          }
        }
        if (flag) {
          bottom_diff[i * dim + index] -= 1;
          count++;
        } else {
          for (int k = start; k <= end; k++) {
            bottom_diff[i * dim + k] = 0;
          }
        }
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(HierarchicalSoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(HierarchicalSoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(HierarchicalSoftmaxWithLoss);

}  // namespace caffe
