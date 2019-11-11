package com.skymindglobal.faceverification.identification;

import com.skymindglobal.faceverification.detection.FaceLocalization;
import com.skymindglobal.faceverification.common.LabelManager;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public class VGG16FaceIdentifier extends FaceIdentifier {
    private static final Logger log = LoggerFactory.getLogger(VGG16FaceIdentifier.class);

    // labels - model's classes
    private static String[] labels;
    private ComputationGraph model = null;
    private static String modelFilename = new File(".").getAbsolutePath() + "/generated-models/vgg16.zip";
    private static String labelFilename = new File(".").getAbsolutePath() + "/generated-models/vgg16.lbl";
    public static final int VGG16_INPUT_WIDTH = 224;
    public static final int VGG16_INPUT_HEIGHT = 224;
    private int numPrediction;

    public VGG16FaceIdentifier(int numPrediction) throws IOException, ClassNotFoundException {
        this.numPrediction = numPrediction;
        this.labels = LabelManager.importLabels(labelFilename);
        if (new File(modelFilename).exists()) {
            log.info("Load model...");
            try {
                model = ModelSerializer.restoreComputationGraph(modelFilename);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            log.info("Model not found.");
        }
    }

    @Override
    public List<List<Prediction>> identify(List<FaceLocalization> faceLocalizations, Mat image) throws IOException {

        NativeImageLoader nativeImageLoader = new NativeImageLoader();
        List<List<Prediction>> collection = new ArrayList<>();
        for (int i = 0; i<faceLocalizations.size(); i++){

            int X = (int) faceLocalizations.get(i).getLeft_x();
            int Y = (int) faceLocalizations.get(i).getLeft_y();
            int Width = faceLocalizations.get(i).getValidWidth(image.size().width());
            int Height = faceLocalizations.get(i).getValidHeight(image.size().height());

            // Crop face, Resize and convert into INDArr
            Mat crop_image = new Mat(image, new Rect(X, Y, Width, Height));
            resize(crop_image, crop_image, new Size(VGG16_INPUT_WIDTH, VGG16_INPUT_HEIGHT));
            INDArray _image = nativeImageLoader.asMatrix(crop_image);

            // initiate vgg16 pre-processing
            VGG16ImagePreProcessor _VGG16ImagePreProcessor = new VGG16ImagePreProcessor();
            _VGG16ImagePreProcessor.transform(_image);

            // predicts
            INDArray[] output = model.output(false, _image);
            List<Prediction> predictions = decodePredictions(output[0], faceLocalizations.get(i));
            collection.add(predictions);
        }
        return collection;
    }

    private List<Prediction> decodePredictions(INDArray encodedPredictions, FaceLocalization faceLocalization) {
        List<Prediction> decodedPredictions = new ArrayList<>();
        int[] topX = new int[this.numPrediction];
        float[] topXProb = new float[this.numPrediction];

        int i = 0;
        for (INDArray currentBatch = encodedPredictions.getRow(0).dup(); i < this.numPrediction; ++i) {

            topX[i] = Nd4j.argMax(currentBatch, 1).getInt(0, 0);
            topXProb[i] = currentBatch.getFloat(0, topX[i]);
            currentBatch.putScalar(0, topX[i], 0.0D);
            decodedPredictions.add(new Prediction(labels[topX[i]],(topXProb[i] * 100.0F),faceLocalization));
        }
        return decodedPredictions;
    }
}