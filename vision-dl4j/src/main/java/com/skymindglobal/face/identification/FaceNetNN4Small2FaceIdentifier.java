package com.skymindglobal.face.identification;

import com.skymindglobal.face.detection.FaceLocalization;
import org.bytedeco.javacpp.opencv_core;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.nd4j.linalg.ops.transforms.Transforms.cosineDistance;

public class FaceNetNN4Small2FaceIdentifier extends FaceIdentifier  {
    private static final Logger log = LoggerFactory.getLogger(FaceNetNN4Small2FaceIdentifier.class);

    private static final int FaceNetNN4Small2_HEIGHT = 96;
    private static final int FaceNetNN4Small2_WIDTH = 96;
    private static final int channels = 3;
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static String modelFilename = new File(".").getAbsolutePath() + "/generated-models/facenet_lfw_v2.zip";
    private static ArrayList<FaceNetEmbed> FaceNetEmbedList = new ArrayList<>();
    private static ComputationGraph snipped;

    public FaceNetNN4Small2FaceIdentifier(File classDict) throws IOException {

        ImageRecordReader recordReader = new ImageRecordReader(FaceNetNN4Small2_HEIGHT, FaceNetNN4Small2_WIDTH, channels, labelMaker);
        recordReader.initialize(new FileSplit(classDict));
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 1, 1, classDict.listFiles().length);
        List<String> labels = iter.getLabels();
        if (new File(modelFilename).exists()) {
            log.info("Load model...");
            ComputationGraph net = ModelSerializer.restoreComputationGraph(modelFilename, true);
            snipped = new TransferLearning.GraphBuilder(net)
                    .setFeatureExtractor("embeddings") // the L2Normalize vertex and layers below are frozen
                    .removeVertexAndConnections("lossLayer")
                    .setOutputs("embeddings")
                    .build();

            while (iter.hasNext()) {
                DataSet Ds = iter.next();
                INDArray embedding = getEmbeddings(Ds.getFeatures());
                String label = labels.get(decodeLabelID(Ds.getLabels()));
                FaceNetEmbedList.add(new FaceNetEmbed(label, embedding));
            }
        }
    }

    private INDArray getEmbeddings(INDArray arr) {
        return snipped.feedForward(arr, false).get("embeddings");
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
            resize(crop_image, crop_image, new opencv_core.Size(FaceNetNN4Small2_WIDTH, FaceNetNN4Small2_HEIGHT));
            INDArray _image = nativeImageLoader.asMatrix(crop_image);

            // predicts
            List<Prediction> predictions = predict(_image, faceLocalizations.get(i),3, 0.5);
            collection.add(predictions);
        }
        return collection;
    }

    private List<Prediction> predict(INDArray image,FaceLocalization faceLocalizations, int numPredictions, double cosineDistanceThreshold) {
        INDArray anchor = getEmbeddings(image);
        List<Prediction> predicted = new ArrayList<>();
        for (FaceNetEmbed i:FaceNetEmbedList){
            double cosineDistance = cosineDistance(anchor, i.getEmbedding());
            if(cosineDistance<cosineDistanceThreshold){
                predicted.add(new Prediction(i.getLabel(), cosineDistance, faceLocalizations));
            }
        }

        final Map<String, List<Prediction>> map = predicted.stream().collect(Collectors.groupingBy(p -> p.getLabel()));
        List<Prediction> result = new ArrayList<>();
        for (final Map.Entry<String, List<Prediction>> entry : map.entrySet()) {
            final double average = entry.getValue().stream()
                    .mapToDouble(p -> p.getScore()).min().getAsDouble();
            result.add(new Prediction(entry.getKey(), average, faceLocalizations));
        }

        result.sort(Comparator.comparing(Prediction::getScore));
        return result;
    }

    private static int decodeLabelID(INDArray encoded) {
        int topX;
        topX = Nd4j.argMax(encoded.getRow(0).dup(), 1).getInt(0, 0);
        encoded.getRow(0).dup().putScalar(0, topX, 0.0D);
        return topX;
    }
}
