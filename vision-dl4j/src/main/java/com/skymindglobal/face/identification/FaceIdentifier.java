package com.skymindglobal.face.identification;

import com.skymindglobal.face.detection.FaceLocalization;
import org.bytedeco.javacpp.opencv_core;

import java.io.IOException;
import java.util.List;

public class FaceIdentifier implements IFaceIdentifier{
    public static final String CUSTOM_VGG16 = "CUSTOM_VGG16";
    public static final String FACENETNN4SMALL2 = "FACENETNN4SMALL2";
    public static final String ZHZD = "ZHZD";

    public List<List<Prediction>> identify(List<FaceLocalization> faceLocalizations, opencv_core.Mat image) throws IOException {
        return null;
    }
}
