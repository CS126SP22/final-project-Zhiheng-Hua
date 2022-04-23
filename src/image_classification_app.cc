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
  ci::gl::drawString("drag and drop model file to upload model", vec2(80, 520), ColorT<float>("white"), font);
  ci::gl::drawString("drag and drop image to make prediction", vec2(80, 540), ColorT<float>("white"), font);
  ci::gl::drawString(message_, vec2(80, 560), message_color_, font);
}

void ImageClassificationApp::fileDrop( ci::app::FileDropEvent event ) {
  const auto& file_vec = event.getFiles();
  
  if (file_vec.size() != 1) {
    message_ = "you can only drop one file";
    message_color_ = ColorT<float>("red");
    return;
  }
  
  auto file_path = file_vec[0];
  if (file_path.extension() == ".jpg" || file_path.extension() == ".png") {
    image_path_ = file_path.string();
    message_ = "image uploaded successfully";
    message_color_ = ColorT<float>("green");

    // make a Prediction
    MatrixXf* img_mat = Util::imageToMatrix(image_path_, cnn_.getImageWidth(), cnn_.getImageHeight());
    VectorXf probability = cnn_.predict( img_mat );
    prediction_ = cnn_.classifyImage(probability);
  } else if (file_path.extension() == ".mdl") {
    cnn_.readModel(file_path.string());
    message_ = "model uploaded successfully";
    message_color_ = ColorT<float>("green");
  } else {
    message_ = "cannot recognize file type";
    message_color_ = ColorT<float>("red");
  }
}
