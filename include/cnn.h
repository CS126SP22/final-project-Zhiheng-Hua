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
    CNN();
    
    CNN(int kernel_size, const string& path, int img_width, int img_height,
        int lw, int lh, int sw, int sh);
    
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

    /**
     * forward propagation in the feature section
     * @param lw maxPooling layer width
     * @param lh maxPooling layer height
     * @param sw maxPooling stride width
     * @param sh maxPooling stride height
     * @return map of string to flattened image
     */
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
    pair<MatrixXf, MatrixXf> costFunctionPrime(const map<string, vector<VectorXf>>& Xs, float* error);
    
    /**
     * update W1
     * @param dJdW1 gradient of cost function with respect to W1 
     */
    void updateW1(const MatrixXf& dJdW1);

    /**
     * update W2
     * @param dJdW2 gradient of cost function with respect to W2 
     */
    void updateW2(const MatrixXf& dJdW2);
    
    /**
     * one single function that takes care all training process
     * when this function finish, W1, W2 will be trained    // TODO: update kernels as well
     * @param max_iter 
     */
    void trainModel(int max_iter);
    
    /**
     * make a prediction base on image
     * @param image input image, array of size 3 [r, g, b]
     * @return probability vector
     */
    VectorXf predict(MatrixXf* image);
    
    /**
     * get the name of the resulting class using probability vector provided
     * @param prob probability vector 
     * @return name of the classification result
     */
    string classifyImage(const VectorXf& prob);

    /**
     * constants
     */
    const static int CHANNEL_COUNT = 3;

  private:
    // init directly in constructor
    MatrixXf conv_kernels_[CHANNEL_COUNT];   // array of convolutional layer kernels
    MatrixXf second_conv_kernel_;
    int lw_;      // max pooling layer width
    int lh_;      // max pooling layer height
    int sw_;      // max pooling layer stride width
    int sh_;      // max pooling layer stride height
    
    // init in loadImageFromDataset
    vector<string> labels_;                 // all labels name as vector
    map<string, vector<MatrixXf*>> images_; // mapping label to image, each [r, g, b] array (MatrixXf*)
    map<string, VectorXf> expected_map_;    // mapping label name to expected prob vector
    int c_;                                 // label_count_
    int n_;                                 // total_image_count_
    
    // init in featureForwardPropagation 
    int s_;                   // image_size_ (size of the vector after flattened)
    int t_;                   // hidden_unit_number_ floor((s + c) / 2)
    MatrixXf W1_;             // shape: s x t
    MatrixXf W2_;             // shape: t x c
    
    /**
     * randomly init convolutional layer kernels
     * @param kernel_size size of conv_kernels
     */
    void initKernels(int kernel_size);
    
    /**
     * convert a [r, g, b] image to a flattened, smaller 1d vector
     * @param img image represented using [r, g, b]
     * @return converted image representation, input of fully connected layer
     */
    VectorXf featureForwardHelper(const MatrixXf* img);
    
  public:
    /**
     * getters
     */
    const vector<string>& getLabels();
    const map<string, vector<MatrixXf*>>& getImages();
    int getTotalImageCount() const;
};


