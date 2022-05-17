#include "analysis.h"


Analysis::Analysis(int windowSizeX, int windowSizeY) 
        : windowSizeX_(windowSizeX), windowSizeY_(windowSizeY) {
  top_left_ = vec2(0.1f * windowSizeX_, 0.1f * windowSizeY_);
  graph_size_ = vec2(0.8f * windowSizeX_);
  bottom_left_ = top_left_ + vec2(0, graph_size_.y);
}

void Analysis::drawErrorLineGraph(const VectorXf& ys) {
  if (ys.size() == 0) {
    return;
  }
  
  ci::gl::clear(ci::Color("black"));
  
  // draw graph background color
  color(Color::gray(0.2f));
  drawSolidRect(Rectf(top_left_, top_left_ + graph_size_));
  
  color( 1.0f, 0.5f, 0.25f ); // orange
  ci::gl::lineWidth(5);
  
  float spacingX = (float) graph_size_.x / ys.size();
  float maxY = ys.maxCoeff();
  float spacingY = graph_size_.y / ceil(maxY);
  vec2 lastPoint = vec2(top_left_.x, -spacingY * ys[0] + bottom_left_.y);
  
  for (int i = 0; i < (int) ys.size(); i++) {
    vec2 curr = vec2(spacingX * i, -spacingY * ys[i]) + bottom_left_;
    ci::gl::drawLine(lastPoint, curr);
    lastPoint = curr;
  }
}


