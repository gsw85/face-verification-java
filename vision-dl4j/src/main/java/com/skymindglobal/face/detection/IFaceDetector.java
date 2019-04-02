package com.skymindglobal.face.detection;

import org.bytedeco.javacpp.opencv_core;

import java.util.List;

interface IFaceDetector {

    int image_width = 0;
    int image_height = 0;
    double detection_threshold = 0.0;

    public void setImageWidth(int width);
    public int getImage_width();
    public void setImageHeight(int height);
    public int getImage_height();
    public void setDetectionThreshold(double threshold);
    public double getDetection_threshold();

    public List<FaceLocalization> detectFaces(opencv_core.Mat image);
}

