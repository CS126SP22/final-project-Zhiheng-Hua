#pragma once

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/ImageIo.h"

#include <Eigen/Dense>
#include <string>
#include <map>
#include <stdexcept>
#include <utility>


using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::map;
using std::pair;

using Eigen::DenseBase;
using Eigen::Vector3;
using Eigen::MatrixXf;  // matrix of int with dynamic size


class CNN {
  public:
    /**
     * constructor
     * @param kernel_size size of the kernel to use for convolution,
     * should be an odd number
     */
    CNN(int kernel_size);
    
    /**
     * parse all image from dataset into matrices, store them in the CNN class
     * also extract labels of the data using the dir names
     * @param path @param path path of the dataset, with structure:
     * --- dataset folder name
     *    --- label_name
     *        --- img of this class
     * @param img_count total image in the dataset
     * @param img_width expected width of the image, all image should have the same training size
     * @param img_height expected height of the image, all image should have the same training size
     */
    void loadImageFromDataset(const string& path, int img_count, int img_width, int img_height);
    
    
    /**
     * constants
     */
    const static int CHANNEL_COUNT = 3;
    
    /**
     * getters 
     */
    const vector<string>& getLabels();
    const map<string, vector<MatrixXf*>>& getImages();

  private:
    vector<string> labels_;
    map<string, vector<MatrixXf*>> images_;
    
    MatrixXf conv_kernels[CHANNEL_COUNT];
};


