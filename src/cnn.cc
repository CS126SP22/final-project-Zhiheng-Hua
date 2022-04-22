#include "cnn.h"
#include "util.h"



CNN::CNN(int kernel_size) {
  for (auto & conv_kernel : conv_kernels_) {
    conv_kernel = MatrixXf::Random(kernel_size, kernel_size);
  }
}

void CNN::loadImageFromDataset(const string &path, int img_width, int img_height) {
  // init labels_, c_ (label count), image_width_, image_height_
  labels_ = Util::getLabelVectorFromDataset(path);
  c_ = labels_.size();
  
  // init expected_map_
  for (int i = 0; i < c_; i++) {
    VectorXf curr = VectorXf::Zero(c_);
    curr[i] = 1;
    expected_map_.insert({labels_[i], curr});
  }

  // read from dataset and store them in images_
  for (auto & label : labels_) {
    images_.insert({label, vector<MatrixXf*>()});
  }
  
  n_ = 0; // increment image count when loading image
  for (const auto & label_folder : fs::directory_iterator(path)) {
    // add images to each label
    vector<MatrixXf*>& curr_vector = images_[label_folder.path().filename().string()];
    
    for (const auto & img_iter : fs::directory_iterator(label_folder.path().string())) {
      string img_path = img_iter.path().string();
      MatrixXf* rgb_image = Util::imageToMatrix(img_path, img_width, img_height);
      curr_vector.push_back(rgb_image);
      ++n_;
    }
  }
}

MatrixXf CNN::softmaxJacobian(const VectorXf& y_hat) {
  assert(y_hat.size() == c_);
  
  MatrixXf res = MatrixXf::Zero(c_, c_);
  
  for (int r = 0; r < c_; r++) {
    float sr = y_hat[r];
    for (int c = 0; c < c_; c++) {
      float sc = y_hat[c];
      res(r, c) = (r == c) ? sr * (1 - sc) : -sr * sc;
    }
  }
  
  return res;
}

map<string, vector<VectorXf>> CNN::featureForwardPropagation() {
  // FIXME: should be in loadimage 
  s_ = 25;
  t_ = 16;
  W1_ = MatrixXf::Random(s_, t_);
  W2_ = MatrixXf::Random(t_, c_);

  map<string, vector<VectorXf>> Xs;
  
  for (const auto& entry : images_) {
    // init map
    Xs.insert({entry.first, vector<VectorXf>()});
    // loop through image
    for (const MatrixXf* img : entry.second) {
      MatrixXf conv_res = Util::convolution3D(img, conv_kernels_);
      Util::Relu(conv_res);
      MatrixXf pooling_res = Util::maxPooling(conv_res, 5, 5, 5, 5);  // FIXME: hardcoded
      VectorXf X = Util::flatten(pooling_res);
      Xs[entry.first].push_back(X);
    }
  }
  
  return Xs;
}

vector<VectorXf> CNN::FcForwardPropagation(const VectorXf& X) {
  VectorXf Z2 = X.transpose() * W1_;
  VectorXf a2 = Util::sigmoid(Z2);
  VectorXf Z3 = a2.transpose() * W2_;
  VectorXf y_hat = Util::softmax(Z3);
  return {Z2, a2, y_hat};
}

pair<MatrixXf, MatrixXf> CNN::costFunctionPrime(const map<string, vector<VectorXf>>& Xs, float* error) {
  // maximum ascend directions for each weight matrix
  MatrixXf dJdW1 = MatrixXf::Zero(s_, t_);
  MatrixXf dJdW2 = MatrixXf::Zero(t_, c_);
  
  *error = 0;
  
  // loop over all input images
  for (const auto& key_val : Xs) {
    // y: expected probability vector of current label
    VectorXf y = expected_map_[key_val.first];
    
    for (const auto& X: key_val.second) {
      // find {Z2, a2, y_hat} using forward propagation
      vector<VectorXf> forward_result = FcForwardPropagation(X);
      
      VectorXf Z2 = forward_result[0];
      VectorXf a2 = forward_result[1];
      VectorXf y_hat = forward_result[2];
      
      // softmax Jacobian Matrix
      MatrixXf dYdZ3 = softmaxJacobian(y_hat);  // (c_ x c_)
      // Error vector
      VectorXf err_vec = (y_hat - y);           // (1 x c_).T
      *error += pow(err_vec.norm(), 2);

      /* dJdW2 */
      VectorXf delta3 = err_vec.transpose() * dYdZ3;  // (1 x c_) * (c_ x c_)
      for (int t_iter = 0; t_iter < t_; t_iter++) {   // (1 x c_) * [t_ x (c_ x c_)]
        dJdW2.row(t_iter) += delta3 * a2[t_iter];
      }

      /* dJdW1 */ 
      MatrixXf delta2 = err_vec.transpose() * dYdZ3 * W2_.transpose() * Util::sigmoidPrime(Z2); // (1 x t_)
      for (int s_iter = 0; s_iter < s_; s_iter++) {
        dJdW1.row(s_iter) += delta2 * X[s_iter];
      }
    }
  }
  
  *error /= (float) n_;
  
  // average dJdW2 and dJdW1
  dJdW2 *= (2.0f / (float) n_);
  dJdW1 *= (2.0f / (float) n_);
  
  return {dJdW1, dJdW2};
}

void CNN::updateW1(const MatrixXf& dJdW1) {
  W1_ -= dJdW1;
}

void CNN::updateW2(const MatrixXf& dJdW2) {
  W2_ -= dJdW2;
}

const vector<string>& CNN::getLabels() {
  return labels_;
}

const map<string, vector<MatrixXf*>>& CNN::getImages() {
  return images_;
}

int CNN::getTotalImageCount() const {
  return n_;
}
