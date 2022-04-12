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
  MatrixXi* imageToMatrix(const string& path, int exp_width, int exp_height) {
    // channel_num = # 8-bit components (channel) per pixel, in this case channel is forced, so not used
    int width, height, channel_num;
    unsigned char *data = stbi_load(path.c_str(), &width, &height, &channel_num, 3);
    
    MatrixXi* img = NULL;
    
    if (data != nullptr && width > 0 && height > 0) {
      img = new MatrixXi[3];
      
      MatrixXi matR(exp_height, exp_width);
      MatrixXi matG(exp_height, exp_width);
      MatrixXi matB(exp_height, exp_width);
      
      // zero padding excessive image part
      for (int r = 0; r < exp_height; r++) {
        if (r >= height) {
          matR.row(r) = VectorXi::Zero(exp_width);
          matG.row(r) = VectorXi::Zero(exp_width);
          matB.row(r) = VectorXi::Zero(exp_width);
          continue;
        }
        for (int c = 0 ; c < exp_width; c++) {
          matR(r, c) = (c < width) ? static_cast<int>(data[r * width + c + 0]) : 0;
          matG(r, c) = (c < width) ? static_cast<int>(data[r * width + c + 1]) : 0;
          matB(r, c) = (c < width) ? static_cast<int>(data[r * width + c + 2]) : 0;
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
  
  
  
};

