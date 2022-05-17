#pragma once

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include <Eigen/Dense>
#include "util.h"
#include "analysis.h"
#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <algorithm>

using ci::gl::Texture2d;
using glm::vec2;
using Eigen::MatrixXf;
using ci::ColorT;
using std::string;
using ci::gl::Texture2dRef;
using ci::Font;
using ci::app::KeyEvent;
using std::min;
using std::max;
using std::to_string;

/**
 * An app for image classification
 */
class ImageClassificationApp : public ci::app::App {
  public:
    ImageClassificationApp();
    
    void draw() override;
    void fileDrop( ci::app::FileDropEvent event ) override;
    void keyDown( KeyEvent event ) override;

    // provided that you can see the entire UI on your screen.
    const int kWindowSizeX = 600;
    const int kWindowSizeY = 600;
    const int k_MIN_CONFIG_IDX = 0;
    const int k_MAX_CONFIG_IDX = 6;
    
    const int MAX_ITER = 500;
    const vec2 configTopLeft = vec2(60, 60); // top left corner of config window
    const vec2 configSize = vec2(480, 480);  // config window size
    const vec2 configCursorSize = vec2(10, 10);
    const int configCursorX = 100;
    const vector<vec2> configOptionPos = {vec2(150, 100), vec2(150, 150), vec2(150, 200), vec2(150, 250),
                                          vec2(150, 300), vec2(150, 350), vec2(150, 400)};

  private:
    void drawConfigInterface();
    
    void configIncrement();
    
    void configDecrement();
    
    Texture2dRef mTex;
    string image_path_;
    string message_;
    string prediction_;
    ColorT<float> message_color_;

    VectorXf training_errors_;
    
    // config (see CNN constructor)
    int kernel_size_;
    int img_width_; 
    int img_height_;
    int lw_;
    int lh_;
    int sw_;
    int sh_;
    
    int configCursorIndex_; // use to indicate which is the selected field
    
    CNN cnn_;
    bool haveModel_;
    bool showAnalysis_;
    bool showConfig_;
    
    Analysis analysis_;
};

