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
using Eigen::VectorXf;
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
     * @param path path of the dataset, with structure:
     * --- dataset folder name
     *    --- label_name
     *        --- img of this class
     * @param img_count total image in the dataset
     * @param img_width expected width of the image, all image should have the same training size
     * @param img_height expected height of the image, all image should have the same training size
     */
    void loadImageFromDataset(const string& path, int img_count, int img_width, int img_height);

    /**
     * math reference:
     * https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
     * 
     * @param y_hat
     * @return 
     */
    MatrixXf softmaxJacobian(const VectorXf& y_hat);

    /**
     * obtain {Z2, a2, y_hat} of the input image X
     * @param X vector representation of image as fully connected layer input 
     * @return {Z2, a2, y_hat}
     */
    vector<VectorXf> forwardPropagation(const VectorXf& X);

    pair<MatrixXf, MatrixXf> costFunctionPrime();
    
    
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
    
    /**
     * mapping label name to expected prob vector
     */
    map<string, VectorXf> expected_map_;
    
    MatrixXf conv_kernels[CHANNEL_COUNT];
  
    // TODO: initialize them when loading image
    int c_;                   // label_count_
    int image_width_;
    int image_height_;
    int s_;                   // image_size_ (image_width_ * image_height_)
    int t_;                   // hidden_unit_number_ floor((s + label_count_) / 2)
    int n_;                   // total_image_count_ 

    
    MatrixXf W1_;             // shape: t x s
    MatrixXf W2_;             // shape: c x t

    VectorXf z2_;
    VectorXf z3_;
    
    /**
     * map label to vector of vector representation of images representing
     * inputs of fully connected layer, after max pooling 
     */
    map<string, vector<VectorXf>> Xs_;
};


