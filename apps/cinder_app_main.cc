#include "image_classification_app.h"


void prepareSettings(ImageClassificationApp::Settings* settings) {
  settings->setResizable(false);
}

// This line is a macro that expands into an "int main()" function.
CINDER_APP(ImageClassificationApp, ci::app::RendererGl, prepareSettings);
