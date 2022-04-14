#pragma once

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/ImageIo.h"
#include "cnn.h"

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <algorithm>


namespace fs = std::experimental::filesystem;

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::ifstream;
using std::max;

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

  /**
   * convolution operation and Relu activation function, return one MatrixXf
   * @param input array of 3 MatrixXf, representing the input
   * @param conv_kernels array of 3 MatrixXf, representing the kernels to use for each input 
   * @return MatrixXf representing the result, size should be the same as matrices in the input
   */
  MatrixXf convolution3D(const MatrixXf* input, const MatrixXf* conv_kernels);
  
  /**
   * take in a matrix reference, apply Relu on it
   * @param input matrix to apply Relu
   */
  void Relu(MatrixXf& input);
  
  /**
   * perform max pooling
   * @param input matrix to perform operation
   * @param lh layer height
   * @param lw layer width
   * @param sh stride height
   * @param sw stride width
   * @return max pooling result
   */
  MatrixXf maxPooling(const MatrixXf& input, int lh, int lw, int sh, int sw);
  
  /**
   * softmax function
   * @param input_layer input layer, a 1-dimensional vector with same size of labels counts
   * @return softmax result, probability vector of same size as input_layer
   */
  VectorXf softmax(const VectorXf& input_layer);
}