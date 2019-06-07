package com.skymindglobal.faceverification.identification;

import com.skymindglobal.faceverification.detection.FaceLocalization;
import org.bytedeco.javacpp.opencv_core;

import java.io.IOException;
import java.util.List;

public class FaceIdentifier implements IFaceIdentifier{
    public static final String CUSTOM_VGG16 = "CUSTOM_VGG16";
    public static final String FEATURE_DISTANCE_VGG16_PREBUILT = "FEATURE_DISTANCE_VGG16_PREBUILT";
    public static final String FEATURE_DISTANCE_FACENET_PREBUILT = "FEATURE_DISTANCE_FACENET_PREBUILT";
    public static final String ZHZD = "ZHZD";

    public List<List<Prediction>> identify(List<FaceLocalization> faceLocalizations, opencv_core.Mat image) throws IOException {
        return null;
    }
}
