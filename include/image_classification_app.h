#pragma once

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"



/**
 * An app for image classification
 */
class ImageClassificationApp : public ci::app::App {
  public:
    ImageClassificationApp();

    void draw() override;
    void update() override;

    // provided that you can see the entire UI on your screen.
    const int kWindowSizeX = 1000;
    const int kWindowSizeY = 800;
    const int kMargin = 100;

  private:
};

