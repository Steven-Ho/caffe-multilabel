#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <fstream>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void AccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  //vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  vector<int> top_shape(1);
  top_shape[0] = 6;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::ofstream of;
  of.open("/media/DATA/BigVision/NLTK/caffe/tags_classifier/hierarchical_classification/result.txt", ios::app);
  // Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  int count = 0;
  int true_positive = 0;
  int true_negative = 0;
  int false_positive = 0;
  int false_negative = 0;
  const int auc_pts = 20;
  vector<int> auc_tp(2 * auc_pts + 1, 0);
  vector<int> auc_tn(2 * auc_pts + 1, 0);
  vector<int> auc_fp(2 * auc_pts + 1, 0);
  vector<int> auc_fn(2 * auc_pts + 1, 0);
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      // Top-k accuracy
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < num_labels; ++k) {
        bottom_data_vector.push_back(std::make_pair(
            bottom_data[i * dim + k * inner_num_ + j], k));
      }
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      // check if true label is in top k predictions
      count++;
      if (label_value == 0) {
        if (bottom_data_vector[0].second == 0) {
          true_negative++;
        } else {
          false_positive++;
        }
      } else {
        if (bottom_data_vector[0].second == 0) {
          false_negative++;
        } else {
          true_positive++;
        }
      }
      //for (int k = 0; k < 1; k++) {//top_k_ modified for binary classifier
      //}
      for (int k = 0; k < 2 * auc_pts + 1; k++) {
          int p = k - auc_pts;
          Dtype inc = (1 - exp(-p)) / (1 + exp(-p));
          bottom_data_vector.clear();
          for (int l = 0; l < num_labels; l++) {
            bottom_data_vector.push_back(std::make_pair(
                bottom_data[i * dim + l * inner_num_ + j], l));
          }
          bottom_data_vector[1].first += inc;
          // LOG(INFO) << "first: " << bottom_data_vector[0].first << ", second: " << bottom_data_vector[1].first;
          std::partial_sort(
              bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
              bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
          if (label_value == 0) {
            if (bottom_data_vector[0].second == 0) {
              auc_tn[k]++;
            } else {
              auc_fp[k]++;
            }
          } else {
            if (bottom_data_vector[0].second == 0) {
              auc_fn[k]++;
            } else {
              auc_tp[k]++;
            }
          }
      }
    }
    if (i==0) {
      //LOG(INFO) << "correct: " << correct << ", error: " << error;
      //LOG(INFO) << "positive1: " << positive1 << ", negative1: " <<negative1;
      //LOG(INFO) << "positive2: " << positive2 << ", negative2: " <<negative2;      
    }
  }
  //LOG(INFO) << "accuracy: " << accuracy << ", count: " <<count;
  //LOG(INFO) << "accuracy rate: " << accuracy / count;
  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = Dtype(true_positive) / (true_positive + false_negative);
  top[0]->mutable_cpu_data()[1] = Dtype(true_negative) / (true_negative + false_positive);
  top[0]->mutable_cpu_data()[2] = Dtype(false_positive) / (true_negative + false_positive);
  top[0]->mutable_cpu_data()[3] = Dtype(false_negative) / (true_positive + false_negative);
  top[0]->mutable_cpu_data()[4] = Dtype(true_positive) / (true_positive + false_positive);
  top[0]->mutable_cpu_data()[5] = Dtype(true_negative) / (true_negative + false_negative);

  // int l = auc_pts / 2;
  // of << Dtype(sqrt(Dtype(auc_tp[l] * auc_tn[l]) / ((auc_tp[l] + auc_fn[l]) * (auc_tn[l] + auc_fp[l])))) << std::endl;
  for(int i = 0; i < 2 * auc_pts + 1; i++) {
    of << Dtype(auc_tp[i]) / (auc_tp[i] + auc_fn[i]) << " ";
    of << Dtype(auc_fp[i]) / (auc_fp[i] + auc_tn[i]) << " ";
    of << std::endl;
  }
  of << std::endl;
  //LOG(INFO) << "Write in result.txt";
  /*for (int i = 0; i < auc_pts; i++) {
    of << auc_tp[i] << " " << auc_fn[i] << " " << auc_tn[i] << " " << auc_fp[i] << std::endl;
  }
  of << std::endl;*/
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(AccuracyLayer);
REGISTER_LAYER_CLASS(Accuracy);

}  // namespace caffe
