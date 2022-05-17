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
using ci::ColorT;
using glm::vec2;
using ci::gl::drawSolidRect;
using ci::Rectf;
using ci::Font;
using std::to_string;

using Eigen::VectorXf;

class Analysis {
  public:
    Analysis(int windowSizeX, int windowSizeY);
    
    /**
     * draw error analysis line graph according to error verctor
     * @param ys error vector
     * @return maximum error in the vector for future usage
     */
    float drawErrorLineGraph(const VectorXf& ys);
    
    /**
     * draw labels and x axis
     * @param x_lim max x value
     * @param label x label
     */
    void xAxisLabel(int x_lim, const string& label);

    /**
     * draw labels and y axis
     * @param y_lim max y value
     * @param label y label
     */
    void yAxisLabel(float y_lim, const string& label);

  private:
    int windowSizeX_;
    int windowSizeY_;
    vec2 top_left_;
    vec2 bottom_left_;
    vec2 graph_size_;
};
