package com.skymindglobal.faceverification.identification.feature;

import com.skymindglobal.faceverification.detection.FaceLocalization;
import com.skymindglobal.faceverification.identification.Prediction;
import org.bytedeco.javacpp.opencv_core;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

interface IFaceFeatureProvider {
    public INDArray getEmbeddings(INDArray arr);
    public ArrayList<LabelFeaturePair> setupAnchor(File classDict) throws IOException, ClassNotFoundException;
    public List<Prediction> predict(opencv_core.Mat image, FaceLocalization faceLocalization, int numPredictions, double threshold, int numSamples) throws IOException;
}
