#include "util.h"


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
      
      // zero padding or crop excessive image part to fit exp_width and exp_height
      for (int r = 0; r < exp_height; r++) {
        if (r >= height) {
          matR.row(r) = VectorXf::Zero(exp_width);
          matG.row(r) = VectorXf::Zero(exp_width);
          matB.row(r) = VectorXf::Zero(exp_width);
          continue;
        }
        for (int c = 0 ; c < exp_width; c++) {
          matR(r, c) = (c < width) ? static_cast<float>(data[r * width + c + 0]) / 255 : 0;
          matG(r, c) = (c < width) ? static_cast<float>(data[r * width + c + 1]) / 255 : 0;
          matB(r, c) = (c < width) ? static_cast<float>(data[r * width + c + 2]) / 255 : 0;
        }
      }
      
      img[0] = matR;
      img[1] = matG;
      img[2] = matB;
    } else {
      cout << "error: invalid image at: " << path << endl;
    }
    
    stbi_image_free(data);
    return img;
  }


  MatrixXf convolution3D(const MatrixXf* input, const MatrixXf* conv_kernel) {
    int input_height = input[0].rows();
    int input_width = input[0].cols();
    int kernel_height = conv_kernel[0].rows();
    int kernel_width = conv_kernel[0].cols();
    
    MatrixXf result = MatrixXf::Zero(input_height, input_width);
    
    // prepare padding, reused in the loop
    MatrixXf padding_mat = MatrixXf::Zero(input_height + kernel_height - 1, input_width + kernel_width - 1);
    
    for (int channel = 0; channel < CNN::CHANNEL_COUNT; channel++) {
      // convolution for each input kernel pair
      const MatrixXf& curr_input = input[channel];
      const MatrixXf& curr_kernel = conv_kernel[channel];
      
      // change padding for input matrix first
      padding_mat.block(1, 1, input_height, input_width) = curr_input;
      
      for (int r = 0 ; r < input_height; r++) {
        for (int c = 0; c < input_width; c++) {
          MatrixXf curr_blk = padding_mat.block(r, c, kernel_height, kernel_width);
          result(r, c) += (curr_blk.array() * conv_kernel[channel].array()).sum();
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
  
  MatrixXf maxPooling(const MatrixXf& input, int lw, int lh, int sw, int sh) {
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

  VectorXf softmax(const VectorXf& input_layer) {
    return input_layer.array().exp() / input_layer.array().exp().sum();
  }

  VectorXf flatten(MatrixXf mat) {
    return Map<const VectorXf>(mat.data(), mat.size());
  }

  VectorXf sigmoid(const VectorXf& vec) {
    return 1 / ((-vec.array()).exp() + 1);
  }

  MatrixXf sigmoidPrime(const VectorXf& vec) {
    MatrixXf res = MatrixXf::Zero(vec.size(), vec.size());
    res.diagonal() = (-vec.array()).exp() / ((-vec.array()).exp() + 1).pow(2);
    return res;
  }
};

