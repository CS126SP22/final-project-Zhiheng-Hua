#include "cnn.h"
#include "util.h"




void CNN::loadImageFromDataset(const string &path, int img_count, int img_width, int img_height) {
  cout << "loading image from dataset" << endl;
  
  labels_ = Util::getLabelVectorFromDataset(path);
  images_.reserve(img_count);

  for (const auto & label_folder : fs::directory_iterator(path)) {
    for (const auto & img_iter : fs::directory_iterator(label_folder.path().string())) {
      string img_path = img_iter.path().string();
      MatrixXi* rgb_image = Util::imageToMatrix(img_path, img_width, img_height);
      images_.push_back(rgb_image);
    }
  }
  
  cout << "successfully loaded " << images_.size() << "images"<< endl;
}

const vector<string>& CNN::getLabels() {
  return labels_;
}

const vector<MatrixXi *> &CNN::getImages() {
  return images_;
}

