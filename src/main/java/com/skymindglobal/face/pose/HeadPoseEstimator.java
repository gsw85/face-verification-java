package com.skymindglobal.face.pose;

import com.skymindglobal.face.detection.FaceLocalization;
import org.bytedeco.javacpp.opencv_core;

import java.util.List;

public class HeadPoseEstimator implements IHeadPoseEstimator {
    public static final String OPENCV_HEAD_POSE_ESTIMATOR = "OPENCV_HEAD_POSE_ESTIMATOR";
    public static final String KERAS_MODEL = "KERAS_MODEL";

    public void estimate(List<FaceLocalization> faceLocalizations, opencv_core.Mat cloneCopy) {
    }
}
