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



