package com.skymindglobal.face.identification;

import com.skymindglobal.face.detection.FaceLocalization;
import com.skymindglobal.face.identification.feature.FaceFeatureProvider;
import com.skymindglobal.face.identification.feature.LabelFeaturePair;
import com.skymindglobal.face.identification.feature.VGG16FeatureProvider;
import org.bytedeco.javacpp.opencv_core;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class DistanceFaceIdentifier extends FaceIdentifier {
    private static final Logger log = LoggerFactory.getLogger(DistanceFaceIdentifier.class);
    private final FaceFeatureProvider _FaceFeatureProvider;
    private final int numPredictions;
    private final double threshold;

    public DistanceFaceIdentifier(FaceFeatureProvider faceFeatureProvider, File classDict, int numPredictions, double threshold) throws IOException, ClassNotFoundException {
        this._FaceFeatureProvider = faceFeatureProvider;
        _FaceFeatureProvider.setupAnchor(classDict);
        this.numPredictions = numPredictions;
        this.threshold = threshold;
    }

    @Override
    public List<List<Prediction>> identify(List<FaceLocalization> faceLocalizations, opencv_core.Mat image) throws IOException {
        List<List<Prediction>> collection = new ArrayList<>();
        for (int i = 0; i<faceLocalizations.size(); i++) {

            int X = (int) faceLocalizations.get(i).getLeft_x();
            int Y = (int) faceLocalizations.get(i).getLeft_y();
            int Width = faceLocalizations.get(i).getValidWidth(image.size().width());
            int Height = faceLocalizations.get(i).getValidHeight(image.size().height());

            // Crop face, Resize and convert into INDArr
            opencv_core.Mat crop_image = new opencv_core.Mat(image, new opencv_core.Rect(X, Y, Width, Height));

            // predicts
            List<Prediction> predictions = _FaceFeatureProvider.predict(
                    crop_image, faceLocalizations.get(i), this.numPredictions, this.threshold);
            collection.add(predictions);
        }
        return collection;
    }
}
