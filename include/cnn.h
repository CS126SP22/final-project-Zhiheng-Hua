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


class CNN {
  public:
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
     * getters 
     */
    const vector<string>& getLabels();
    const vector<MatrixXi*>& getImages();

  private:
    vector<string> labels_;
    vector<MatrixXi*> images_;
};


