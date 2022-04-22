#include <util.h>
#include <cnn.h>

#include <iostream>

using std::cout;
using std::endl;

int main() {
  CNN cnn(7);
  cnn.loadImageFromDataset("data/test_image/", 252, 252);

  auto Xs = cnn.featureForwardPropagation();

  cout << "finish propagation" << endl;

  pair<MatrixXf, MatrixXf> result = cnn.costFunctionPrime(Xs);
  cout << "result" << endl;
  cout << result.first << endl;
  
  return 0;
};
