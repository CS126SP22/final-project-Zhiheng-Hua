#include "cnn.h"
#include <catch2/catch.hpp>
#include "util.h"

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


