#pragma once

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include <Eigen/Dense>
#include "util.h"

using ci::gl::Texture2d;
using glm::vec2;
using Eigen::MatrixXf;
using ci::ColorT;
using std::string;
using ci::gl::Texture2dRef;
using ci::Font;

/**
 * An app for image classification
 */
class ImageClassificationApp : public ci::app::App {
  public:
    ImageClassificationApp();
    
    void draw() override;
    void fileDrop( ci::app::FileDropEvent event ) override;

    // provided that you can see the entire UI on your screen.
    const int kWindowSizeX = 600;
    const int kWindowSizeY = 600;

  private:
    Texture2dRef mTex;
    string image_path_;
    string message_;
    string prediction_;
    ColorT<float> message_color_;
    
    CNN cnn_;
};

