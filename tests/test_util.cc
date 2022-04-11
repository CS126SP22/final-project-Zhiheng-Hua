#include "util.h"
#include <catch2/catch.hpp>



TEST_CASE("place holder") {
//  Util::testFunc();
  Util::imageToMatrix("data/natural_images/airplane/airplane_0290.jpg");
//  Util::readDir("data/natural_images/airplane");
  REQUIRE(1 == 1);
}

