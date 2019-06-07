package com.skymindglobal.faceverification.identification;

import com.skymindglobal.faceverification.detection.FaceLocalization;
import org.bytedeco.javacpp.opencv_core;

import java.io.IOException;
import java.util.List;

interface IFaceIdentifier {
    List<List<Prediction>> identify(List<FaceLocalization> faceLocalizations, opencv_core.Mat image) throws IOException;
}
