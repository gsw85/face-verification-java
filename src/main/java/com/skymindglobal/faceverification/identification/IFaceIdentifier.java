package com.skymindglobal.faceverification.identification;

import com.skymindglobal.faceverification.detection.FaceLocalization;
import org.bytedeco.opencv.opencv_core.Mat;

import java.io.IOException;
import java.util.List;

interface IFaceIdentifier {
    List<List<Prediction>> identify(List<FaceLocalization> faceLocalizations, Mat image) throws IOException;
}
