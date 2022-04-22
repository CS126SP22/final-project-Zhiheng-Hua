#include "image_classification_app.h"


ImageClassificationApp::ImageClassificationApp() 
    : image_path_(""), message_(""), message_color_(ColorT<float>("white")), prediction_("")
{
  ci::app::setWindowSize(kWindowSizeX, kWindowSizeY);
}

void ImageClassificationApp::draw() {
  ci::gl::clear(ci::Color("black"));
  
  if (image_path_ != "") {
    //"data/CARS/AUDI/download.jpg"
    auto img = ci::loadImage( image_path_ );
    mTex = Texture2d::create( img );

    ci::gl::pushModelMatrix();
    ci::gl::translate( 0.5f * getWindowWidth(), 0.5f * getWindowHeight() );
    ci::gl::scale( 2.0f, 2.0f );
    ci::gl::translate( -0.5f * mTex->getWidth(), -0.5f * mTex->getHeight() );
    ci::gl::draw( mTex );
    ci::gl::popModelMatrix();
  }

  Font font = Font("Roboto", 30);
  ci::gl::drawString(prediction_, vec2(200, 20), ColorT<float>("white"), Font("Roboto", 40));
  ci::gl::drawString("press space to input image path", vec2(120, 520), ColorT<float>("white"), font);
  ci::gl::drawString("press return to make prediction", vec2(120, 540), ColorT<float>("white"), font);
  ci::gl::drawString(message_, vec2(120, 560), message_color_, font);
}

void ImageClassificationApp::fileDrop( ci::app::FileDropEvent event ) {
  const auto& file_vec = event.getFiles();
  
  if (file_vec.size() != 1) {
    message_ = "you can only drop one file";
    message_color_ = ColorT<float>("red");
    return;
  }

  image_path_ = file_vec[0].string();
  message_ = "image uploaded successfully";
  message_color_ = ColorT<float>("green");

  // TODO: make a Prediction
  prediction_ = "Prediction";
}
