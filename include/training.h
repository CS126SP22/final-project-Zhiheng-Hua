#pragma once

#include "cnn.h"


#include <Eigen/Dense>
#include <string>
#include <map>
#include <stdexcept>
#include <utility>
#include <iostream>
#include <vector>


using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::map;
using std::pair;

using Eigen::DenseBase;
using Eigen::VectorXf;
using Eigen::RowVectorXf;
using Eigen::MatrixXf;  // matrix of int with dynamic size


class Training {
  public:
    Training(int kernel_size, const string& path, int img_width, int img_height,
             int lw, int lh, int sw, int sh, int max_iter);
    
    void trainModel();

  private:
    CNN cnn_;
};

