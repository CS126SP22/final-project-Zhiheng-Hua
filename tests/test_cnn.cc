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

TEST_CASE("playground") {
  CNN cnn(5);
  cnn.loadImageFromDataset("data/test_image/", 25, 25);

  auto Xs = cnn.featureForwardPropagation();

  cout << "finish propagation" << endl;

  pair<MatrixXf, MatrixXf> result = cnn.costFunctionPrime(Xs);
  cout << "result" << endl;
  cout << result.first << endl;
}
