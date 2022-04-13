#pragma once

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/ImageIo.h"

#include <Eigen/Dense>
#include <string>
#include <vector>


namespace fs = std::experimental::filesystem;

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::ifstream;

using Eigen::Vector3;
using Eigen::MatrixXf;  // matrix of float with dynamic size
using Eigen::Matrix;
using Eigen::VectorXf;
using Eigen::Dynamic;


namespace Util {

  /**
   * read dataset folder and extract all label names
   * @param path path of the dataset, should have labels as inner folder names 
   */
  vector<string> getLabelVectorFromDataset(const string& path);
  
  /**
   * 
   * @param path @param path path of the dataset, with structure:
   * --- dataset folder name
   *    --- label_name
   *        --- img of this class
   * @return total image count in the dataset
   */
  int getDatasetImageCount(const string& path);
  
  /**
   * out put a rgb representation of the image
   * @param path path from the working directory root
   * @param exp_width expected width of the image, all image should have the same training size
   * @param exp_height expected height of the image, all image should have the same training size
   * @return list size 3 (CNN::CHANNEL_COUNT) with RGB matrix, empty vector if image failed to parsed
   */
  MatrixXf* imageToMatrix(const string& path, int exp_width, int exp_height);
  
  
  
  
}