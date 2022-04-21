#include "cnn.h"
#include "util.h"



CNN::CNN(int kernel_size) {
  for (auto & conv_kernel : conv_kernels) {
    conv_kernel = MatrixXf::Random(kernel_size, kernel_size);
  }
}

void CNN::loadImageFromDataset(const string &path, int img_count, int img_width, int img_height) {
  labels_ = Util::getLabelVectorFromDataset(path);
  
  for (auto & label : labels_) {
    images_.insert(pair<string, vector<MatrixXf*>>(label, {}));
  }

  for (const auto & label_folder : fs::directory_iterator(path)) {
    // add images to each label
    vector<MatrixXf*>& curr_vector = images_[label_folder.path().filename().string()];
    
    for (const auto & img_iter : fs::directory_iterator(label_folder.path().string())) {
      string img_path = img_iter.path().string();
      MatrixXf* rgb_image = Util::imageToMatrix(img_path, img_width, img_height);
      curr_vector.push_back(rgb_image);
    }
  }
}

const vector<string>& CNN::getLabels() {
  return labels_;
}

const map<string, vector<MatrixXf*>>& CNN::getImages() {
  return images_;
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

vector<VectorXf> CNN::forwardPropagation(const VectorXf& X) {
  VectorXf Z2 = W1_ * X;
  VectorXf a2 = Util::sigmoid(Z2);
  VectorXf Z3 = W2_ * a2;
  VectorXf y_hat = Util::softmax(Z3);
  return {Z2, a2, y_hat};
}

pair<MatrixXf, MatrixXf> CNN::costFunctionPrime() {
  // maximum ascend directions for each weight matrix
  MatrixXf dJdW1 = MatrixXf::Zero(t_, c_);
  MatrixXf dJdW2 = MatrixXf::Zero(t_, c_);
  
  // loop over all input images
  for (const auto& key_val : Xs_) {
    // y: expected probability vector of current label
    VectorXf y = expected_map_[key_val.first];
    
    for (const auto& X: key_val.second) {
      // find {Z2, a2, y_hat} using forward propagation
      vector<VectorXf> forward_result = forwardPropagation(X);
      
      VectorXf Z2 = forward_result[0];
      VectorXf a2 = forward_result[1];
      VectorXf y_hat = forward_result[2];
      
      // softmax Jacobian Matrix
      MatrixXf dYdZ3 = softmaxJacobian(y_hat);
      // Error vector
      VectorXf err_vec = (y_hat - y);     // (1 x c_)

      /* dJdW2 */
      VectorXf delta3 = err_vec * dYdZ3;  // (1 x c_) * (c_ x c_)
      for (int t_iter = 0; t_iter < t_; t_iter++) { // (1 x c_) * [t_ x (c_ x c_)]
        dJdW2.row(t_iter) += delta3 * a2[t_iter];
      }

      /* dJdW1 */
      MatrixXf delta2 = err_vec * dYdZ3 * W2_.transpose() * Util::sigmoidPrime(Z2);
      for (int s_iter = 0; s_iter < s_; s_iter++) {
        dJdW1.row(s_iter) += delta3 * X[s_iter];
      }
    }
  }
  
  // average dJdW2 and dJdW1
  dJdW2 *= (2.0f / (float) n_);
  dJdW1 *= (2.0f / (float) n_);
  
  return {dJdW1, dJdW2};
}


