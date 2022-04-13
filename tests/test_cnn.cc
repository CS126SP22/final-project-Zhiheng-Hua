#include "cnn.h"
#include <catch2/catch.hpp>

TEST_CASE("test loadImageFromDataset") {
  CNN cnn;
  cnn.loadImageFromDataset("data/test_image/", 80, 256, 256);

  vector<string> expected_labels{"airplane","car","cat","dog","flower","fruit","motorbike","person"};
  REQUIRE(cnn.getLabels() == expected_labels);
  
  size_t count = 0;
  for (const auto& label: expected_labels) {
    count += cnn.getImages().at(label).size();
  }
  REQUIRE(count == 88);
}


