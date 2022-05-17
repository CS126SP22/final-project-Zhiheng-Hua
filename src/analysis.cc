#include "analysis.h"


Analysis::Analysis(int windowSizeX, int windowSizeY) 
        : windowSizeX_(windowSizeX), windowSizeY_(windowSizeY) {
  top_left_ = vec2(0.1f * windowSizeX_, 0.1f * windowSizeY_);
  graph_size_ = vec2(0.8f * windowSizeX_);
  bottom_left_ = top_left_ + vec2(0, graph_size_.y);
}

float Analysis::drawErrorLineGraph(const VectorXf& ys) {
  if (ys.size() == 0) {
    return 0.0f;
  }
  
  ci::gl::clear(ci::Color("black"));
  
  // draw graph background color
  color(Color::gray(0.2f));
  drawSolidRect(Rectf(top_left_, top_left_ + graph_size_));
  
  color( 1.0f, 0.5f, 0.25f ); // orange
  ci::gl::lineWidth(5);
  
  float spacingX = (float) graph_size_.x / ys.size();
  float maxY = ys.maxCoeff();
  float spacingY = graph_size_.y / maxY;
  vec2 lastPoint = vec2(top_left_.x, -spacingY * ys[0] + bottom_left_.y);
  
  for (int i = 0; i < (int) ys.size(); i++) {
    vec2 curr = vec2(spacingX * i, -spacingY * ys[i]) + bottom_left_;
    ci::gl::drawLine(lastPoint, curr);
    lastPoint = curr;
  }
  
  return maxY;
}

void Analysis::xAxisLabel(int x_lim, const string& label) {
  Font font = Font("Roboto", 20);
  ci::gl::drawString(label, top_left_ + vec2(graph_size_.x / 3, graph_size_.y), ColorT<float>("white"), font);
  ci::gl::drawString(to_string(x_lim), top_left_ + graph_size_ + vec2(-10, 0), ColorT<float>("white"), font);
}

void Analysis::yAxisLabel(float y_lim, const string& label) {
  Font font = Font("Roboto", 20);
  ci::gl::drawString(label, top_left_ + vec2(-50, graph_size_.y / 3), ColorT<float>("white"), font);
  ci::gl::drawString(to_string(y_lim), top_left_ + vec2(10, 10), ColorT<float>("white"), font);
}



