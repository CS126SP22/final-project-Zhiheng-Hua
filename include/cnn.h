#pragma once

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
     * @param img_width expected width of the image, all image should have the same training size
     * @param img_height expected height of the image, all image should have the same training size
     */
    void loadImageFromDataset(const string& path, int img_width, int img_height);

    /**
     * math reference:
     * https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
     * 
     * @param y_hat
     * @return 
     */
    MatrixXf softmaxJacobian(const VectorXf& y_hat);

    map<string, vector<VectorXf>> featureForwardPropagation();

    /**
     * forward propagation of the fully connected layer
     * @param X vector representation of image as fully connected layer input 
     * @return {Z2, a2, y_hat}
     */
    vector<VectorXf> FcForwardPropagation(const VectorXf& X);

    /**
     * this function should be used in backward propagation phase to obtain {dJdW1, dJdW2}
     * @param Xs mapping of labels to images, where images represented by flattened vectors
     * @return {dJdW1, dJdW2}, the gradient matrix with respect to the two weight matrices
     */
    pair<MatrixXf, MatrixXf> costFunctionPrime(const map<string, vector<VectorXf>>& Xs);

    /**
     * constants
     */
    const static int CHANNEL_COUNT = 3;

  private:
    // init in constructor
    MatrixXf conv_kernels_[CHANNEL_COUNT];   // array of convolutional layer kernels
    
    // init in loadImageFromDataset
    vector<string> labels_;                 // all labels name as vector
    map<string, vector<MatrixXf*>> images_; // mapping label to image, each [r, g, b] array (MatrixXf*)
    map<string, VectorXf> expected_map_;    // mapping label name to expected prob vector
    int c_;                                 // label_count_
    int n_;                                 // total_image_count_
    // TODO: init below in loadImage
    int s_;                   // image_size_ (size of the vector after flattened)
    int t_;                   // hidden_unit_number_ floor((s + label_count_) / 2)
    MatrixXf W1_;             // shape: s x t
    MatrixXf W2_;             // shape: t x c
    
  public:
    /**
     * getters
     */
    const vector<string>& getLabels();
    const map<string, vector<MatrixXf*>>& getImages();
    int getTotalImageCount() const;
};


