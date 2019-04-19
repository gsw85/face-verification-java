package com.skymindglobal.face.identification;

import com.skymindglobal.face.detection.FaceLocalization;
import com.skymindglobal.face.identification.feature.FaceFeatureProvider;
import com.skymindglobal.face.identification.feature.FaceNetNN4Small2FaceFeatureProvider;
import com.skymindglobal.face.identification.feature.LabelFeaturePair;
import com.skymindglobal.face.identification.training.facenet.dataHelpers.LFWCroppedDatasetIterator;
import org.bytedeco.javacpp.opencv_core;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

import static org.bytedeco.javacpp.opencv_imgproc.resize;

public class DistanceFaceIdentifier_Archived extends FaceIdentifier {
    private static final Logger log = LoggerFactory.getLogger(DistanceFaceIdentifier_Archived.class);
    private static final String EMBED_DISTANCE = "EMBED_DISTANCE";
    private static final String SIMILARITY_MODEL = "SIMILARITY_MODEL";
    private static final String NN_CLASSIFIER = "NN_CLASSIFIER";
    private static String prediction_mode = DistanceFaceIdentifier_Archived.EMBED_DISTANCE;

    private static final int FaceNetNN4Small2_HEIGHT = 96;
    private static final int FaceNetNN4Small2_WIDTH = 96;

    private static String similarityModelFile = new File(".").getAbsolutePath() + "/generated-models/same_or_not_softmaxtest1.zip";
    private static String classifierModelFile = new File(".").getAbsolutePath() + "/generated-models/lfw_classification_tryrun0.zip";

    private static ArrayList<LabelFeaturePair> labelFeaturePairList = new ArrayList<>();
    private final FaceFeatureProvider _FaceFeatureProvider;
    private ComputationGraph similarityModel;
    private ComputationGraph classifierModel;
    private List<String> labels;

    public DistanceFaceIdentifier_Archived(File classDict) throws IOException, ClassNotFoundException {

        _FaceFeatureProvider = new FaceNetNN4Small2FaceFeatureProvider();
        labelFeaturePairList = _FaceFeatureProvider.setupAnchor(classDict);

        if(new File(similarityModelFile).exists()){
            loadSimilarityModel();
        }

        if(new File(classifierModelFile).exists()){
            loadNNClassifierModel();
            loadLabels();
        }
    }

    private void loadLabels() throws IOException {
        LFWCroppedDatasetIterator _LFWCroppedDatasetIterator = new LFWCroppedDatasetIterator(
                new File("D:\\Public_Data\\face_recog\\lfw_train_96"),
                new File("D:\\Public_Data\\face_recog\\lfw_test_96"),
                472,
                465
        );
        labels = _LFWCroppedDatasetIterator.trainIterator().getLabels();
    }

    private void loadNNClassifierModel() throws IOException {
        log.info("Load classifier model...");
        classifierModel= ModelSerializer.restoreComputationGraph(classifierModelFile, true);
    }

    private void loadSimilarityModel() throws IOException {
        log.info("Load embedding model...");
        similarityModel= ModelSerializer.restoreComputationGraph(similarityModelFile, true);
    }

    @Override
    public List<List<Prediction>> identify(List<FaceLocalization> faceLocalizations, opencv_core.Mat image) throws IOException {

        NativeImageLoader nativeImageLoader = new NativeImageLoader();
        List<List<Prediction>> collection = new ArrayList<>();
        for (int i = 0; i<faceLocalizations.size(); i++) {

            int X = (int) faceLocalizations.get(i).getLeft_x();
            int Y = (int) faceLocalizations.get(i).getLeft_y();
            int Width = faceLocalizations.get(i).getValidWidth(image.size().width());
            int Height = faceLocalizations.get(i).getValidHeight(image.size().height());

            // Crop face, Resize and convert into INDArr
            opencv_core.Mat crop_image = new opencv_core.Mat(image, new opencv_core.Rect(X, Y, Width, Height));

//            // for debug purpose only
//            String generatedString = RandomStringUtils.random(10, true, true);
//            ImageIO.write(Java2DFrameUtils.toBufferedImage(crop_image), "jpg",
//                    new File("D:\\tmp\\"+generatedString+".jpg"));

            // predicts
            List<Prediction> predictions = null;
            switch(prediction_mode){
                case DistanceFaceIdentifier_Archived.EMBED_DISTANCE:
                    predictions = _FaceFeatureProvider.predict(crop_image, faceLocalizations.get(i),10, 1.0);
                    break;
//                case DistanceFaceIdentifier.SIMILARITY_MODEL:
//                    predictions = predictSimilarityModel(_image, faceLocalizations.get(i), 10, 0.5);
//                    break;
//                case DistanceFaceIdentifier.NN_CLASSIFIER:
//                    predictions = predictClassifierModel(_image, faceLocalizations.get(i), 10, 0.5);
//                    break;
                default:
                    break;
            }
            collection.add(predictions);
        }
        return collection;
    }

    private List<Prediction> predictClassifierModel(INDArray image, FaceLocalization faceLocalization, int numPredictions, double confidenceThreshold) throws IOException {
        INDArray embedding = _FaceFeatureProvider.getEmbeddings(image);
        INDArray[] result = classifierModel.output(embedding);
        List<Prediction> predictions = decodePredictions(result[0], numPredictions, faceLocalization);

        // sort and select top N
        predictions.sort(Comparator.comparing(Prediction::getScore));
        Collections.reverse(predictions);

        return predictions;
    }

    private List<Prediction> decodePredictions(INDArray encodedPredictions, int numPredicted, FaceLocalization faceLocalization) throws IOException {
        List<Prediction> decodedPredictions = new ArrayList<>();
        int[] topX = new int[numPredicted];
        float[] topXProb = new float[numPredicted];

        int i = 0;
        for (INDArray currentBatch = encodedPredictions.getRow(0).dup(); i < numPredicted; ++i) {

            topX[i] = Nd4j.argMax(currentBatch, 1).getInt(0, 0);
            topXProb[i] = currentBatch.getFloat(0, topX[i]);
            currentBatch.putScalar(0, topX[i], 0.0D);
            decodedPredictions.add(new Prediction(labels.get(topX[i]), (topXProb[i] * 100.0F), faceLocalization));
        }
        return decodedPredictions;
    }

    private List<Prediction> predictSimilarityModel(INDArray image, FaceLocalization faceLocalization, int numPredictions, double confidenceThreshold) {
        INDArray anchor = _FaceFeatureProvider.getEmbeddings(image);
        List<Prediction> predicted = new ArrayList<>();
        for (LabelFeaturePair i: labelFeaturePairList){
            double confidence = getSimilarityConfidence(anchor, i.getEmbedding());
            if(confidence > confidenceThreshold){
                predicted.add(new Prediction(i.getLabel(), confidence, faceLocalization));
            }
        }

        // aggregator - average comparison per class
        List<Prediction> summary = new ArrayList<>();
        final Map<String, List<Prediction>> map = predicted.stream().collect(Collectors.groupingBy(p -> p.getLabel()));
        for (final Map.Entry<String, List<Prediction>> entry : map.entrySet()) {
            final double min = entry.getValue().stream().mapToDouble(p -> p.getScore()).average().getAsDouble();
            summary.add(new Prediction(entry.getKey(), min, faceLocalization));
        }

        // sort and select top N
        summary.sort(Comparator.comparing(Prediction::getScore));
        Collections.reverse(summary);
        List<Prediction> result = new ArrayList();
        for(int i=0; i<numPredictions; i++){
            if(i<summary.size()){
                result.add(summary.get(i));
            }
        }
        return result;
    }

    private double getSimilarityConfidence(INDArray anchor, INDArray embedding) {
        INDArray feature = Nd4j.concat(1, anchor, embedding);
        INDArray[] result = similarityModel.output(feature);
        double negative = result[0].toDoubleVector()[0];
        double positive = result[0].toDoubleVector()[1];

        return positive/(positive+negative);
    }

    private static int decodeLabelID(INDArray encoded) {
        int topX;
        topX = Nd4j.argMax(encoded.getRow(0).dup(), 1).getInt(0, 0);
        encoded.getRow(0).dup().putScalar(0, topX, 0.0D);
        return topX;
    }
}
