package com.skymindglobal.faceverification.pose;

import com.skymindglobal.faceverification.detection.FaceLocalization;
import org.bytedeco.opencv.opencv_core.Mat;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;
import java.util.List;

public class KerasModel_HeadPoseEstimator extends HeadPoseEstimator {

    public KerasModel_HeadPoseEstimator() throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        String simpleMlp = new ClassPathResource("tf-keras-deep-head-pose/biwi_model.h5").getFile().getPath();
        ComputationGraph model = KerasModelImport.importKerasModelAndWeights(simpleMlp);
    }

    public void estimate(List<FaceLocalization> faceLocalizations, Mat image) {

    }
}
