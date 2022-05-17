#pragma once

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include <Eigen/Dense>
#include <string>
#include <map>
#include <stdexcept>
#include <utility>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::map;
using std::pair;

using ci::gl::clear;
using ci::Color;
using ci::gl::color;
using glm::vec2;
using ci::gl::drawSolidRect;
using ci::Rectf;

using Eigen::VectorXf;

class Analysis {
  public:
    Analysis(int windowSizeX, int windowSizeY);
    
    void drawErrorLineGraph(const VectorXf& ys);

  private:
    int windowSizeX_;
    int windowSizeY_;
    vec2 top_left_;
    vec2 bottom_left_;
    vec2 graph_size_;
};
