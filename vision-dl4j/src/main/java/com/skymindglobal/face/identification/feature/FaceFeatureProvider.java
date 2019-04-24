package com.skymindglobal.face.identification.feature;

import com.skymindglobal.face.detection.FaceLocalization;
import com.skymindglobal.face.identification.Prediction;
import org.bytedeco.javacpp.opencv_core;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class FaceFeatureProvider implements IFaceFeatureProvider {

    @Override
    public INDArray getEmbeddings(INDArray arr) {
        return null;
    }

    @Override
    public ArrayList<LabelFeaturePair> setupAnchor(File classDict) throws IOException, ClassNotFoundException {
        return null;
    }

    public List<Prediction> predict(opencv_core.Mat image, FaceLocalization faceLocalization, int numPredictions, double threshold, int numSamples, int minSupport) throws IOException {
        return null;
    }

    public int decodeLabelID(INDArray encoded) {
        int topX;
        topX = Nd4j.argMax(encoded.getRow(0).dup(), 1).getInt(0, 0);
        encoded.getRow(0).dup().putScalar(0, topX, 0.0D);
        return topX;
    }
}
