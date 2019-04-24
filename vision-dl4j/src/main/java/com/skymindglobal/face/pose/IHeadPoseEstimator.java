package com.skymindglobal.face.pose;

import com.skymindglobal.face.detection.FaceLocalization;
import org.bytedeco.javacpp.opencv_core;

import java.util.List;

interface IHeadPoseEstimator {
    public void estimate(List<FaceLocalization> faceLocalizations, opencv_core.Mat image);
}
