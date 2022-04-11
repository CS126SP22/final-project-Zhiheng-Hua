#pragma once

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/ImageIo.h"

#include <Eigen/Dense>
#include <string>

using Eigen::MatrixXi;  // matrix of int with dynamic size
using std::cout;
using std::endl;
using std::string;
using std::vector;
using Eigen::Vector3;


namespace Util {
  /**
   * out put a rgb representation of the image
   * @param path path from the working directory root
   * @return vector size3 with RGB matrix, empty vector if image failed to parsed
   */
  vector<MatrixXi> imageToMatrix(const string& path);

  
  /**
   * 
   * @param path path from the working directory 
   */
  void readDir(const string& path);
  
  
}