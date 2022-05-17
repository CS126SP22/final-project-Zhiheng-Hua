#include "image_classification_app.h"


ImageClassificationApp::ImageClassificationApp() 
    : image_path_(""), message_(""), message_color_(ColorT<float>("white")), prediction_(""), 
    haveModel_(false), analysis_(kWindowSizeX, kWindowSizeY), showAnalysis_(false),
    kernel_size_(5), img_width_(256), img_height_(256), lw_(5), lh_(5), sw_(5), sh_(5),
    showConfig_(false), configCursorIndex_(0)
{
  ci::app::setWindowSize(kWindowSizeX, kWindowSizeY);
}

void ImageClassificationApp::draw() {
  ci::gl::clear(ci::Color("black"));

  Font font = Font("Roboto", 30);
  ci::gl::drawString(prediction_, vec2(200, 20), ColorT<float>("white"), Font("Roboto", 40));
  ci::gl::drawString("press c to set training configurations", vec2(80, 500), ColorT<float>("white"), font);
  ci::gl::drawString("drag and drop model file to upload model", vec2(80, 520), ColorT<float>("white"), font);
  ci::gl::drawString("drag and drop image to make prediction", vec2(80, 540), ColorT<float>("white"), font);
  ci::gl::drawString(message_, vec2(80, 560), message_color_, font);
  
  if (showAnalysis_) {
    float y_lim = analysis_.drawErrorLineGraph(training_errors_);
    analysis_.xAxisLabel(MAX_ITER, "iterations");
    analysis_.yAxisLabel(y_lim, "cost\nfunction\nerrors");
    return;
  }
  
  if (showConfig_) {
    drawConfigInterface();
    return;
  }
  
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
}

void ImageClassificationApp::fileDrop( ci::app::FileDropEvent event ) {
  showConfig_ = false;
  showAnalysis_ = false;
  const auto& file_vec = event.getFiles();
  
  if (file_vec.size() != 1) {
    message_ = "you can only drop one file";
    message_color_ = ColorT<float>("red");
    return;
  }
  
  auto file_path = file_vec[0];
  if (file_path.extension() == ".jpg" || file_path.extension() == ".png") {
    if (!haveModel_) {
      message_ = "please upload a model before prediction";
      message_color_ = ColorT<float>("red");
      return;
    }
    
    image_path_ = file_path.string();
    message_ = file_path.filename().string() + " uploaded successfully";
    message_color_ = ColorT<float>("green");

    // make a Prediction
    MatrixXf* img_mat = Util::imageToMatrix(image_path_, cnn_.getImageWidth(), cnn_.getImageHeight());
    VectorXf probability = cnn_.predict( img_mat );
    prediction_ = cnn_.classifyImage(probability);
  } else if (file_path.extension() == ".cnn") {
    cnn_.readModel(file_path.string());
    message_ = file_path.filename().string() + " uploaded successfully";
    message_color_ = ColorT<float>("green");
    haveModel_ = true;
  } else if (fs::is_directory(file_path)) {
    // train Model
    cnn_ = CNN(kernel_size_, file_path.string(), img_width_, img_height_, lw_, lh_, sw_, sh_);
    training_errors_ = cnn_.trainModel(MAX_ITER);
    
    // save model with timestamp filename
    time_t t = time(nullptr);   // get time now
    struct tm* now = localtime(&t);
    char buffer [80] = {0};
    strftime(buffer, 80,"%Y%m%d%H%M%S",now);
    cnn_.saveModel("model-" + std::string(buffer) + ".cnn");
    showAnalysis_ = true;
  } else {
    message_ = "cannot recognize file type";
    message_color_ = ColorT<float>("red");
  }
}

void ImageClassificationApp::keyDown(ci::app::KeyEvent event) {
  switch (event.getCode()) {
    case KeyEvent::KEY_ESCAPE:
      showConfig_ = false;
      break;
    case KeyEvent::KEY_c:
      showConfig_ = true;
      break; 
  }
  
  if (showConfig_) {
    switch (event.getCode()) {
      case KeyEvent::KEY_RIGHT:
        configCursorIndex_ = min(k_MAX_CONFIG_IDX, configCursorIndex_ + 1);
        break;
      case KeyEvent::KEY_LEFT:
        configCursorIndex_ = max(k_MIN_CONFIG_IDX, configCursorIndex_ - 1);
        break;
      case KeyEvent::KEY_UP:
        configIncrement();
        break;
      case KeyEvent::KEY_DOWN:
        configDecrement();
        break;
    }
  }
}

void ImageClassificationApp::drawConfigInterface() {
  ci::gl::clear(ci::Color("black"));
  color(Color::white());
  drawSolidRect(Rectf(configTopLeft, configTopLeft + configSize));

  color(Color("blue"));
  vec2 cursor_top_left = vec2(configCursorX, configOptionPos[configCursorIndex_].y) + configCursorSize;
  drawSolidRect(Rectf(cursor_top_left, cursor_top_left + configCursorSize));

  Font font = Font("Roboto", 30);
  
  ci::gl::drawString("Kernel Size\n" + to_string(kernel_size_), configOptionPos[0], ColorT<float>("black"), font);
  ci::gl::drawString("Image Width\n" + to_string(img_width_), configOptionPos[1], ColorT<float>("black"), font);
  ci::gl::drawString("Image Height\n" + to_string(img_height_), configOptionPos[2], ColorT<float>("black"), font);
  ci::gl::drawString("Max Pooling Layer Width\n" + to_string(lw_), configOptionPos[3], ColorT<float>("black"), font);
  ci::gl::drawString("Max Pooling Layer Height\n" + to_string(lh_), configOptionPos[4], ColorT<float>("black"), font);
  ci::gl::drawString("Max Pooling Stride Width\n" + to_string(sw_), configOptionPos[5], ColorT<float>("black"), font);
  ci::gl::drawString("Max Pooling Stride Height\n" + to_string(sh_), configOptionPos[6], ColorT<float>("black"), font);

  ci::gl::drawString("press -> to select next", vec2(80, 460), ColorT<float>("purple"), font);
  ci::gl::drawString("press <- to select previous", vec2(80, 480), ColorT<float>("purple"), font);
  ci::gl::drawString("press up to increase or down to decrease", vec2(80, 500), ColorT<float>("purple"), font);
  ci::gl::drawString("press esc to go back to main page", vec2(80, 520), ColorT<float>("purple"), font);
  ci::gl::drawString("drop image dataset folder to start training", vec2(60, 550), ColorT<float>("white"), font);
}

void ImageClassificationApp::configIncrement() {
  switch (configCursorIndex_) {
    case 0:
      ++kernel_size_;
      break;
    case 1:
      ++img_width_;
      break;
    case 2:
      ++img_height_;
      break;
    case 3:
      ++lw_;
      break;
    case 4:
      ++lh_;
      break;
    case 5:
      ++sw_;
      break;
    case 6:
      ++sh_;
      break;
  }
}

void ImageClassificationApp::configDecrement() {
  switch (configCursorIndex_) {
    case 0:
      --kernel_size_;
      break;
    case 1:
      --img_width_;
      break;
    case 2:
      --img_height_;
      break;
    case 3:
      --lw_;
      break;
    case 4:
      --lh_;
      break;
    case 5:
      --sw_;
      break;
    case 6:
      --sh_;
      break;
  }
}
