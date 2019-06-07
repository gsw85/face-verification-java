package com.skymindglobal.faceverification.detection;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_dnn;

import java.util.List;

import static org.bytedeco.javacpp.opencv_dnn.blobFromImage;
import static org.bytedeco.javacpp.opencv_dnn.readNetFromCaffe;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

public class FaceDetector implements IFaceDetector {
    public static final String OPENCV_DL_FACEDETECTOR = "OPENCV_DL_FACEDETECTOR";
    public static final String OPENIMAJ_FKE_FACEDETECTOR = "OPENIMAJ_FKE_FACEDETECTOR";
    private opencv_dnn.Net model;
    private int image_width;
    private int image_height;
    private double detection_threshold;

    public FaceDetector() {
    }

    @Override
    public void setImageWidth(int width) {
        this.image_width = width;
    }

    @Override
    public int getImage_width() {
        return this.image_width;
    }

    @Override
    public void setImageHeight(int height) {
        this.image_height = height;
    }

    @Override
    public int getImage_height() {
        return this.image_height;
    }

    @Override
    public void setDetectionThreshold(double detection_threshold) {
        this.detection_threshold = detection_threshold;
    }

    @Override
    public double getDetection_threshold() {
        return this.detection_threshold;
    }

    @Override
    public void detectFaces(opencv_core.Mat image) {

    }

    @Override
    public List<FaceLocalization> getFaceLocalization() {
        return null;
    }


}
