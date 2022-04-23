#include "cnn.h"
#include <catch2/catch.hpp>
#include "util.h"


bool matrixApproxEqual(const MatrixXf& m1, const MatrixXf& m2) {
  if (m1.rows() != m2.rows() || m1.cols() != m2.cols()) {
    return false;
  }
  
  for (int i = 0; i < m1.rows(); i++) {
    for (int j = 0; j < m1.cols(); j++) {
      if (m1(i, j) != Approx(m2(i, j)) ) {
        return false; 
      }
    }
  }
  
  return true;
}

TEST_CASE("test loadImageFromDataset") {
  CNN cnn;
  cnn.loadImageFromDataset("data/test_image/", 256, 256);

  vector<string> expected_labels{"airplane","car","cat","dog","flower","fruit","motorbike","person"};
  REQUIRE(cnn.getLabels() == expected_labels);

  size_t count = 0;
  for (const auto& label: expected_labels) {
    count += cnn.getImages().at(label).size();
  }
  REQUIRE(count == 88);
  
  REQUIRE(cnn.getTotalImageCount() == 88);
}

TEST_CASE("test trainModel") {
  CNN cnn(5, "data/test_image/", 256, 256, 5, 5, 5, 5);
  cnn.trainModel(500);
  
  MatrixXf* image = Util::imageToMatrix("data/test_image/flower/flower_0000.jpg", 256, 256);
  VectorXf pred = cnn.predict(image);

  int pred_idx = 0;
  for (int i = 0; i < pred.size(); i++) {
    pred_idx = (pred[i] > pred[pred_idx]) ? i : pred_idx;
  }

  REQUIRE(pred_idx == 4);
}

TEST_CASE("test classifyImage") {
  CNN cnn;
  cnn.loadImageFromDataset("data/test_image/", 1, 1);
  
  VectorXf prob(8);
  prob << 1.0f, 0.4f, 0.5f, 0.2f, 0.0f, 0.1f, 0.8f, 0.8f;
  
  REQUIRE(cnn.classifyImage(prob) == "airplane");
}

TEST_CASE("test saveModel") {
  CNN cnn;
  cnn.loadImageFromDataset("data/test_image/", 1, 1);
  cnn.saveModel("natural-images-model-test.mdl");

  ifstream file("data/natural-images-model-test.mdl");
  REQUIRE(file.is_open());

  int count = 0;
  string line;
  while ( getline(file, line) ) {
    ++count;
  }
  
  file.close();
  REQUIRE(count == 41);

  remove("data/natural-images-model-test.mdl");
}

TEST_CASE("test readModel") {
  CNN cnn;
  cnn.loadImageFromDataset("data/test_image/", 1, 1);
  cnn.saveModel("natural-images-model-test.mdl");

  MatrixXf orig_W1 = cnn.getW1();
  MatrixXf orig_W2 = cnn.getW2();

  CNN cnn2;
  cnn2.readModel("data/natural-images-model-test.mdl");

  MatrixXf new_W1 = cnn2.getW1();
  MatrixXf new_W2 = cnn2.getW2();
  
  remove("data/natural-images-model-test.mdl");
  
  REQUIRE(matrixApproxEqual(orig_W1, new_W1));
  REQUIRE(matrixApproxEqual(orig_W2, new_W2));
}

TEST_CASE("playground") {
  CNN cnn(5, "data/test_image/", 256, 256, 5, 5, 5, 5);
  
  cnn.trainModel(500);
  cnn.saveModel("natural-images-model.mdl");
}
