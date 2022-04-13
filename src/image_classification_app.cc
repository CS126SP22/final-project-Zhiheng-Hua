#include "image_classification_app.h"



ImageClassificationApp::ImageClassificationApp() {
  ci::app::setWindowSize(kWindowSizeX, kWindowSizeY);
}

void ImageClassificationApp::draw() {
  ci::Color background_color("black");
  ci::gl::clear(background_color);
}

void ImageClassificationApp::update() {
  
}


