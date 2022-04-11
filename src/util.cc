#include "util.h"
#include <vector>
#include <iostream>

extern "C" {
  #define STB_IMAGE_IMPLEMENTATION
  #include "stb_image.h"
}

#include "cinder/gl/Texture.h"
#include "cinder/Surface.h"

#include <iostream>
#include <filesystem>
namespace fs = std::experimental::filesystem;

using Eigen::VectorXi;

using ci::loadImage;
using ci::Area;
using ci::Surface;
using std::ifstream;
using ci::gl::Texture;


namespace Util {
  void readDir(const string& path) {
    for (const auto & entry : fs::directory_iterator(path))
      std::cout << entry.path() << std::endl;
  }
  
  // reference: https://www.cplusplus.com/forum/beginner/267364/
  vector<MatrixXi> imageToMatrix(const string& path) {
    // channel_num = # 8-bit components (channel) per pixel, in this case channel is forced, so not used
    int width, height, channel_num;
    unsigned char *data = stbi_load(path.c_str(), &width, &height, &channel_num, 3);

    vector<MatrixXi> img;

//    matR.conservativeResizeLike(MatrixXi::Zero(2, 1));
//    cout << matR << endl;

    if (data != nullptr && width > 0 && height > 0) {
      MatrixXi matR(height, width);
      MatrixXi matG(height, width);
      MatrixXi matB(height, width);
      
      for (int r = 0; r < height; r++) {
        for (int c = 0 ; c < width; c++) {
          matR(r, c) = static_cast<int>(data[r * width + c + 0]);
          matG(r, c) = static_cast<int>(data[r * width + c + 1]);
          matB(r, c) = static_cast<int>(data[r * width + c + 2]);
        }
      }
//      cout << matR(0, 0) << " ";
//      cout << matG(0, 0) << " ";
//      cout << matB(0, 0) << " ";

      img.push_back(matR);
      img.push_back(matG);
      img.push_back(matB);
    }
    
    stbi_image_free(data);
    return img;
  }
};

