#include <util.h>
#include <cnn.h>
#include <iostream>


using std::cout;
using std::endl;

int main() {
//  CNN cnn(5, "data/NATURAL", 256, 256, 5, 5, 5, 5)
  CNN cnn(5, "data/intel_image/small_set", 150, 150, 5, 5, 5, 5);

  VectorXf res = cnn.trainModel(500);
  cnn.saveModel("intel-images-model.cnn");
  
  return 0;
};
