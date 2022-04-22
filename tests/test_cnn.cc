#include "cnn.h"
#include <catch2/catch.hpp>

TEST_CASE("test loadImageFromDataset") {
  CNN cnn(3);
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

TEST_CASE("test forward costFunctionPrime") {
  CNN cnn(5);
  cnn.loadImageFromDataset("data/test_image/", 25, 25);

  auto Xs = cnn.featureForwardPropagation();
  
  for (int iter = 0; iter < 1000; iter++) {
    float error1;
    pair<MatrixXf, MatrixXf> result1 = cnn.costFunctionPrime(Xs, &error1);

    cnn.updateW1(result1.first);
    cnn.updateW2(result1.second);
  }
  
  VectorXf pred = cnn.FcForwardPropagation(Xs["flower"][0])[2];
  
  int pred_idx = 0;
  for (int i = 0; i < pred.size(); i++) {
    pred_idx = (pred[i] > pred[pred_idx]) ? i : pred_idx;
  }
  
  REQUIRE(pred_idx == 4);
}
