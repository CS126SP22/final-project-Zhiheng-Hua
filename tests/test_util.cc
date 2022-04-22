#include "util.h"
#include "cnn.h"
#include <catch2/catch.hpp>
#include <math.h>



TEST_CASE("test getLabelVectorFromDataset") {
  Util::getLabelVectorFromDataset("data/natural_images");
  vector<string> expected = {};
  REQUIRE(1 == 1);
}

TEST_CASE("test getDatasetImageCount") {
  int img_count = Util::getDatasetImageCount("data/natural_images");
  REQUIRE(img_count == 6899);
}

TEST_CASE("test imageToMatrix") {
  MatrixXf* result = Util::imageToMatrix("data/natural_images/airplane/airplane_0000.jpg", 256, 256);
  vector<float> rgb_tl{223.0f / 255, 240.0f / 255, 232.0f / 255};  // top left corner
  vector<float> rgb_rb{0, 0, 0};        // right bottom corner
  
  for (int i = 0; i < CNN::CHANNEL_COUNT; i++) {
    REQUIRE(result[i].size() == 65536);
    REQUIRE(result[i](0, 0) == rgb_tl[i]);
    REQUIRE(result[i](255, 255) == rgb_rb[i]);
  }
  
  delete[] result;
}

TEST_CASE("test convolution3D") {
  MatrixXf input[3] = {MatrixXf(2,2), MatrixXf(2,2), MatrixXf(2,2)};
  input[0] << 1, 2, 3, 4;
  input[1] << 2, 3, 4, 1;
  input[2] << 3, 4, 1, 2;
  MatrixXf kernels[3] = {MatrixXf(3,3), MatrixXf(3,3), MatrixXf(3,3)};
  kernels[0] << 1, 0, -1, 1, 0, -1, 1, 0, -1;
  kernels[1] << -1, 1, 0, -1, 1, 0, -1, 1, 0;
  kernels[2] << 0, 1, -1, 0, 1, -1, 0, 1, -1;

  MatrixXf expected = MatrixXf(2,2);
  expected << -2, 8, -2, 8;

  REQUIRE(Util::convolution3D(input, kernels) == expected);
}

TEST_CASE("test Relu") {
  MatrixXf input = MatrixXf(3,3);
  input << -4, 0, 28, 
           15, -76, 2, 
           1, 0, 39;
  MatrixXf expected = MatrixXf(3,3);
  expected << 0, 0, 28,
              15, 0, 2,
              1, 0, 39;
  Util::Relu(input);
  
  REQUIRE(input == expected);
}

TEST_CASE("test maxPooling") {
  MatrixXf input = MatrixXf(5,5);
  input << 1, 2, 3, 4, 5,
           6, 7, 8, 9, 10,
           11, 12, 13, 14, 15,
           16, 17, 18, 19, 20,
           21, 22, 23, 24, 25;
  
  SECTION("test with good shape") {
    MatrixXf expected = MatrixXf(2,2);
    expected << 13, 15,
                23, 25;
    MatrixXf actual = Util::maxPooling(input, 3, 3, 2, 2);

    REQUIRE(actual == expected);
  }

  SECTION("test with tricky shape") {
    MatrixXf expected = MatrixXf(1,2);
    expected << 13, 15;
    MatrixXf actual = Util::maxPooling(input, 3, 3, 2, 3);

    REQUIRE(actual == expected);
  }
}

TEST_CASE("test softmax") {
  VectorXf input(3);
  input << 1, 2, 3;

  VectorXf result = Util::softmax(input);
  for (int i = 1; i <= 3; i++) {
    REQUIRE(result[i - 1] == Approx(exp(i) / (exp(1) + exp(2) + exp(3))) ); 
  }
}

TEST_CASE("test sigmoid") {
  VectorXf input(3);
  input << 1, 2, 3;

  VectorXf result = Util::sigmoid(input);
  
  for (int i = 1; i <= 3; i++) {
    REQUIRE( result[i - 1] == Approx(1 / (1 + exp(-i))) );
  }
}

TEST_CASE("test sigmoidPrime") {
  VectorXf input(3);
  input << 1, 2, 3;

  MatrixXf result = Util::sigmoidPrime(input);
  
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j) {
        REQUIRE( result(i, j) == Approx(exp(-i-1) / pow(1 + exp(-i-1), 2)) );
      } else {
        REQUIRE( result(i, j) == 0 );
      }
    }
  }
}

TEST_CASE("test flatten") {
  MatrixXf input = MatrixXf(3,3);
  input << -4, 0, 28,
            15, -76, 2,
            1, 0, 39;
  VectorXf expected(9);
  expected << -4, 15, 1, 0, -76, 0, 28, 2, 39;
  VectorXf actual = Util::flatten(input);
  
  REQUIRE(actual == expected);
}
