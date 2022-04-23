#include "cnn.h"
#include "util.h"


CNN::CNN() : lw_(0), lh_(0), sw_(0), sh_(0), c_(0), n_(0), s_(0), t_(0), 
            W1_(MatrixXf::Random(3, 3)), W2_(MatrixXf::Random(3, 3)), image_width_(0), image_height_(0) 
{
  initKernels(3);
}

CNN::CNN(int kernel_size, const string &path, int img_width, int img_height, 
         int lw, int lh, int sw, int sh) 
   : lw_(lw), lh_(lh), sw_(sw), sh_(sh), c_(0), n_(0), s_(0), t_(0), 
     image_width_(img_width), image_height_(img_height)
{
  initKernels(kernel_size);
  loadImageFromDataset(path, img_width, img_height);
}

void CNN::loadImageFromDataset(const string &path, int img_width, int img_height) {
  // init labels_, c_ (label count), image_width_, image_height
  labels_ = Util::getLabelVectorFromDataset(path);
  c_ = labels_.size();
  image_width_ = img_width;
  image_height_ = img_height;
  
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
  map<string, vector<VectorXf>> Xs;
  for (const auto& entry : images_) {
    // init map
    Xs.insert({entry.first, vector<VectorXf>()});
    // loop through image
    for (const MatrixXf* img : entry.second) {
      VectorXf X = featureForwardHelper(img);
      Xs[entry.first].push_back(X);
    }
  }
  
  s_ = Xs.begin()->second[1].size();
  t_ = (s_ + c_) / 2;
  W1_ = MatrixXf::Random(s_, t_);
  W2_ = MatrixXf::Random(t_, c_);
  
  return Xs;
}

VectorXf CNN::featureForwardHelper(const MatrixXf* img) {
  MatrixXf conv_res1 = Util::convolution3D(img, conv_kernels_);
  Util::Relu(conv_res1);
  MatrixXf pooling_res1 = Util::maxPooling(conv_res1, lw_, lh_, sw_, sh_);
  
  MatrixXf conv_res2 = Util::convolution(pooling_res1, second_conv_kernel_);
  Util::Relu(conv_res2);
  MatrixXf pooling_res2 = Util::maxPooling(conv_res2, lw_, lh_, sw_, sh_);

  return Util::flatten(pooling_res2);
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

void CNN::initKernels(int kernel_size) {
  for (auto & conv_kernel : conv_kernels_) {
    conv_kernel = MatrixXf::Random(kernel_size, kernel_size);
  }
  second_conv_kernel_ = MatrixXf::Random(kernel_size, kernel_size);
}

VectorXf CNN::predict(MatrixXf* image) {
  VectorXf X = featureForwardHelper(image);
  return FcForwardPropagation(X)[2];
}

void CNN::trainModel(int max_iter)
{
  auto Xs = featureForwardPropagation();

  for (int iter = 0; iter < max_iter; iter++) {
    float error;
    pair<MatrixXf, MatrixXf> result = costFunctionPrime(Xs, &error);
    updateW1(result.first);
    updateW2(result.second);
  }
}

string CNN::classifyImage(const VectorXf &prob) {
  // obtain idx of highest probability
  int pred_idx = 0;
  for (int i = 0; i < prob.size(); i++) {
    pred_idx = (prob[i] > prob[pred_idx]) ? i : pred_idx;
  }
  return labels_[pred_idx];
}

void CNN::saveModel(const string &filename) {
  ofstream file("data/" + filename);
  
  // save image_width, image_height
  file << image_width_ << "\n" << image_height_ << "\n";
  
  // lw_, lh_, sw_, sh_, c_, n_, s_, t_ 
  file << lw_ << "\n" << lh_ << "\n" << sw_ << "\n" << sh_ << "\n" 
       << c_ << "\n" << n_ << "\n" << s_ << "\n" << t_ << "\n";
  
  // #labels
  file << labels_.size() << "\n";
  for (const auto& label : labels_) {
    file << label << "\n";
  }
  
  // kernels size
  file << second_conv_kernel_.rows() << "\n";
  
  // conv kernels
  for (int channel = 0; channel < CNN::CHANNEL_COUNT; channel++) {
    const MatrixXf& curr_kernel = conv_kernels_[channel];
    for (int i = 0; i < curr_kernel.rows(); i++) {
      for (int j = 0; j < curr_kernel.cols(); j++) {
        file << curr_kernel(i, j) << "\n";
      }
    }
  }
  
  // second conv kernel
  for (int i = 0; i < second_conv_kernel_.rows(); i++) {
    for (int j = 0; j < second_conv_kernel_.cols(); j++) {
      file << second_conv_kernel_(i, j) << "\n";
    }
  }
  
  // save W1
  int w1_rows = W1_.rows();
  int w1_cols = W1_.cols();
  file << w1_rows << "\n" << w1_cols << "\n";
  for (int i = 0; i < w1_rows; i++) {
    for (int j = 0; j < w1_cols; j++) {
      file << W1_(i, j) << "\n";
    }
  }
  
  // save W2
  int w2_rows = W2_.rows();
  int w2_cols = W2_.cols();
  file << w2_rows << "\n" << w2_cols << "\n";
  for (int i = 0; i < w2_rows; i++) {
    for (int j = 0; j < w2_cols; j++) {
      file << W2_(i, j) << "\n";
    }
  }
  
  file.close();
}

void CNN::readModel(const string &path) {
  ifstream model(path);

  string line;
  
  // image width, height
  getline(model, line); image_width_ = stoi(line);
  getline(model, line); image_height_ = stoi(line);

  // lw_, lh_, sw_, sh_, c_, n_, s_, t_ 
  getline(model, line); lw_ = stoi(line);
  getline(model, line); lh_ = stoi(line);
  getline(model, line); sw_ = stoi(line);
  getline(model, line); sh_ = stoi(line);
  
  getline(model, line); c_ = stoi(line);
  getline(model, line); n_ = stoi(line);
  getline(model, line); s_ = stoi(line);
  getline(model, line); t_ = stoi(line);

  // #label
  getline(model, line);
  int label_size = stoi(line);
  labels_ = vector<string>(label_size);
  
  // labels
  for (int i = 0; i < label_size; i++) {
    getline(model, line); labels_[i] = line;
  }

  // kernels size
  getline(model, line); 
  int kernel_size = stoi(line);

  // conv kernels
  for (int channel = 0; channel < CNN::CHANNEL_COUNT; channel++) {
    MatrixXf curr_kernel(kernel_size, kernel_size);
    for (int i = 0; i < kernel_size; i++) {
      for (int j = 0; j < kernel_size; j++) {
        getline(model, line);
        curr_kernel(i, j) = stof(line);
      }
    }
    conv_kernels_[channel] = curr_kernel;
  }

  // second conv kernel
  MatrixXf sec_kernel(kernel_size, kernel_size);
  for (int i = 0; i < kernel_size; i++) {
    for (int j = 0; j < kernel_size; j++) {
      getline(model, line);
      sec_kernel(i, j) = stof(line);
    }
  }
  second_conv_kernel_ = sec_kernel;

  // #rows, #cols of W1
  getline(model, line);
  int r1 = stoi(line);
  getline(model, line);
  int c1 = stoi(line);
  W1_ = MatrixXf(r1, c1);

  // coeffs of W1
  for (int r = 0; r < r1; r++) {
    for (int c = 0; c < c1; c++) {
      getline(model, line);
      W1_(r, c) = stof(line);
    }
  }

  // #rows, #cols of W1
  getline(model, line);
  int r2 = stoi(line);
  getline(model, line);
  int c2 = stoi(line);
  W2_ = MatrixXf(r2, c2);

  // coeffs of W1
  for (int r = 0; r < r2; r++) {
    for (int c = 0; c < c2; c++) {
      getline(model, line);
      W2_(r, c) = stof(line);
    }
  }
  
  model.close();
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

const MatrixXf &CNN::getW1() {
  return W1_;
}

const MatrixXf &CNN::getW2() {
  return W2_;
}

int CNN::getTotalImageCount() const {
  return n_;
}

int CNN::getImageWidth() const {
  return image_width_;
}

int CNN::getImageHeight() const {
  return image_height_;
}
