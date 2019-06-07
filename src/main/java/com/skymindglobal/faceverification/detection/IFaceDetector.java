package com.skymindglobal.faceverification.detection;

import org.bytedeco.javacpp.opencv_core;

import java.util.List;

interface IFaceDetector {
    void setImageWidth(int width);
    int getImage_width();
    void setImageHeight(int height);
    int getImage_height();
    void setDetectionThreshold(double threshold);
    double getDetection_threshold();
    void detectFaces(opencv_core.Mat image);
    List<FaceLocalization> getFaceLocalization();
}

