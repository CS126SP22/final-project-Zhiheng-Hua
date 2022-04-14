#include "util.h"

#include "cinder/gl/Texture.h"
#include "cinder/Surface.h"

#include <iostream>
#include <filesystem>

extern "C" {
  #define STB_IMAGE_IMPLEMENTATION
  #include "stb_image.h"
}


namespace Util {
  
  vector<string> getLabelVectorFromDataset(const string& path) {
    vector<string> res;
    for (const auto & entry : fs::directory_iterator(path)) {
      res.push_back(entry.path().filename().string());
    }
    return res;
  }

  int getDatasetImageCount(const string& path) {
    int count = 0;
    for (const auto & label_folder : fs::directory_iterator(path)) {
      for (const auto & img : fs::directory_iterator(label_folder.path().string())) {
        ++count;
      }
    }
    return count;
  }
  
  // reference: https://www.cplusplus.com/forum/beginner/267364/
  MatrixXf* imageToMatrix(const string& path, int exp_width, int exp_height) {
    // channel_num = # 8-bit components (channel) per pixel, in this case channel is forced, so not used
    int width, height, channel_num;
    unsigned char *data = stbi_load(path.c_str(), &width, &height, &channel_num, 3);
    
    MatrixXf* img = nullptr;
    
    if (data != nullptr && width > 0 && height > 0) {
      img = new MatrixXf[3];
      
      MatrixXf matR(exp_height, exp_width);
      MatrixXf matG(exp_height, exp_width);
      MatrixXf matB(exp_height, exp_width);
      
      // zero padding excessive image part
      for (int r = 0; r < exp_height; r++) {
        if (r >= height) {
          matR.row(r) = VectorXf::Zero(exp_width);
          matG.row(r) = VectorXf::Zero(exp_width);
          matB.row(r) = VectorXf::Zero(exp_width);
          continue;
        }
        for (int c = 0 ; c < exp_width; c++) {
          matR(r, c) = (c < width) ? static_cast<float>(data[r * width + c + 0]) : 0;
          matG(r, c) = (c < width) ? static_cast<float>(data[r * width + c + 1]) : 0;
          matB(r, c) = (c < width) ? static_cast<float>(data[r * width + c + 2]) : 0;
        }
      }
      
      img[0] = matR;
      img[1] = matG;
      img[2] = matB;
      
//      cout << matR(0, 0) << " ";
//      cout << matG(0, 0) << " ";
//      cout << matB(0, 0) << " ";
    } else {
      cout << "error: invalid image at: " << path << endl;
    }
    
    stbi_image_free(data);
    return img;
  }


  MatrixXf convolution3D(const MatrixXf* input, const MatrixXf* conv_kernels) {
    int input_height = input[0].rows();
    int input_width = input[0].cols();
    int kernel_height = conv_kernels[0].rows();
    int kernel_width = conv_kernels[0].cols();
    
    MatrixXf result = MatrixXf::Zero(input_height, input_width);
    
    // prepare padding, reused in the loop
    MatrixXf padding_mat = MatrixXf::Zero(input_height + kernel_height - 1, input_width + kernel_width - 1);
    
    for (int channel = 0; channel < CNN::CHANNEL_COUNT; channel++) {
      // convolution for each input kernel pair
      const MatrixXf& curr_input = input[channel];
      const MatrixXf& curr_kernel = conv_kernels[channel];
      
      // change padding for input matrix first
      padding_mat.block(1, 1, input_height, input_width) = curr_input;
      
      for (int r = 0 ; r < input_height; r++) {
        for (int c = 0; c < input_width; c++) {
          MatrixXf curr_blk = padding_mat.block(r, c, kernel_height, kernel_width);
          result(r, c) += (curr_blk.array() * conv_kernels[channel].array()).sum();
        }
      }
    }
    
    return result;
  }

  void Relu(MatrixXf& input) {
    for (int r = 0 ; r < input.rows(); r++) {
      for (int c = 0; c < input.cols(); c++) {
        input(r, c) = max(0.0f, input(r, c));
      }
    }
  }
  
  MatrixXf maxPooling(const MatrixXf& input, int lh, int lw, int sh, int sw) {
    int input_height = input.rows();
    int input_width = input.cols();
    assert(lh <= input_height && lw <= input_width);

    int res_h = (input_height - lh) / sh + 1;
    int res_w = (input_width - lw) / sw + 1;
    MatrixXf result = MatrixXf(res_h, res_w);
    
    int h_idx = 0;
    int w_idx = 0;
    for (int r = 0; r < input_height && r + lh <= input_height; r += sh) {
      for (int c = 0; c < input_width && c + lw <= input_width; c += sw) {
        result(h_idx, w_idx) = input.block(r, c, lh, lw).maxCoeff();
        w_idx = (w_idx + 1) % res_w;
      }
      h_idx = (h_idx + 1) % res_h;
    }
    
    return result;
  }
};

