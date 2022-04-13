#include "util.h"
#include <catch2/catch.hpp>



TEST_CASE("test getLabelVectorFromDataset") {
//  Util::testFunc();
//  Util::imageToMatrix("data/natural_images/airplane/airplane_0290.jpg");
  Util::getLabelVectorFromDataset("data/natural_images");
  vector<string> expected = {};
  REQUIRE(1 == 1);
}

TEST_CASE("test getDatasetImageCount") {
  int img_count = Util::getDatasetImageCount("data/natural_images");
  REQUIRE(img_count == 6899);
}

TEST_CASE("test imageToMatrix") {
  MatrixXi* result = Util::imageToMatrix("data/natural_images/airplane/airplane_0000.jpg", 256, 256);
  vector<int> rgb_tl{223, 240, 232};  // top left corner
  vector<int> rgb_rb{0, 0, 0};        // right bottom corner
  
  REQUIRE(result->size() == 3);
  for (int i = 0; i < 3; i++) {
    REQUIRE(result[i].rows() == 256);
    REQUIRE(result[i].cols() == 256);
    REQUIRE(result[i](0, 0) == rgb_tl[i]);
    REQUIRE(result[i](255, 255) == rgb_rb[i]);
  }
  
  delete[] result;
}