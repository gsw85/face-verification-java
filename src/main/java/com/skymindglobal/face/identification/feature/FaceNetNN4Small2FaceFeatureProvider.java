package com.skymindglobal.face.identification.feature;

import com.skymindglobal.face.detection.FaceLocalization;
import com.skymindglobal.face.identification.Prediction;
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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.nd4j.linalg.ops.transforms.Transforms.euclideanDistance;

public class FaceNetNN4Small2FaceFeatureProvider extends FaceFeatureProvider {
    private static String embeddingModelFilename = new File(".").getAbsolutePath() + "/generated-models/embedding.zip";
    private static final int FaceNetNN4Small2_HEIGHT = 96;
    private static final int FaceNetNN4Small2_WIDTH = 96;
    private static final int channels = 3;
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private ComputationGraph model;
    private static ArrayList<LabelFeaturePair> labelFeaturePairList = new ArrayList<>();

    public FaceNetNN4Small2FaceFeatureProvider() throws IOException {
        if (new File(embeddingModelFilename).exists()) {
            ComputationGraph net = ModelSerializer.restoreComputationGraph(embeddingModelFilename, true);
            model = new TransferLearning.GraphBuilder(net)
                    .setFeatureExtractor("embeddings") // the L2Normalize vertex and layers below are frozen
                    .removeVertexAndConnections("lossLayer")
                    .setOutputs("embeddings")
                    .build();
        }
    }

    public ArrayList<LabelFeaturePair> setupAnchor(File dictionary) throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(FaceNetNN4Small2_HEIGHT, FaceNetNN4Small2_WIDTH, channels, labelMaker);
        recordReader.initialize(new FileSplit(dictionary));
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 1, 1, dictionary.listFiles().length);
        List<String> labels = iter.getLabels();
        while (iter.hasNext()) {
            DataSet Ds = iter.next();
            INDArray embedding = getEmbeddings(Ds.getFeatures());
            String label = labels.get(decodeLabelID(Ds.getLabels()));
            labelFeaturePairList.add(new LabelFeaturePair(label, embedding));
        }
        return labelFeaturePairList;
    }

    public INDArray getEmbeddings(INDArray arr) {
        return model.feedForward(arr, false).get("embeddings");
    }

    public List<Prediction> predict(opencv_core.Mat image, FaceLocalization faceLocalization, int numPredictions, double threshold) throws IOException {
        NativeImageLoader nativeImageLoader = new NativeImageLoader();
        resize(image, image, new opencv_core.Size(FaceNetNN4Small2_WIDTH, FaceNetNN4Small2_HEIGHT));
        INDArray _image = nativeImageLoader.asMatrix(image);
        INDArray anchor = getEmbeddings(_image);
        List<Prediction> predicted = new ArrayList<>();
        for (LabelFeaturePair i: labelFeaturePairList){
            INDArray embed = i.getEmbedding();
            double distance = euclideanDistance(anchor, embed);
            predicted.add(new Prediction(i.getLabel(), distance, faceLocalization));
        }

        // aggregator - average comparison per class
        List<Prediction> summary = new ArrayList<>();
        final Map<String, List<Prediction>> map = predicted.stream().collect(Collectors.groupingBy(p -> p.getLabel()));
        for (final Map.Entry<String, List<Prediction>> entry : map.entrySet()) {
            final double average = entry.getValue().stream()
                    .mapToDouble(p -> p.getScore()).average().getAsDouble();
//
//            final double min = entry.getValue().stream()
//                    .mapToDouble(p -> p.getScore()).min().getAsDouble();
//            final double topNAvg = entry.getValue().stream()
//                    .mapToDouble(p -> p.getScore()).sorted().limit(3).average().getAsDouble();
            // median
//            MedianFinder _MedianFinder = new MedianFinder();
//            entry.getValue().stream().forEach(p -> _MedianFinder.addNum(p.getScore()));
            if(average < threshold) {
                summary.add(new Prediction(entry.getKey(), average, faceLocalization));
            }
        }

        // sort and select top N
        summary.sort(Comparator.comparing(Prediction::getScore));
        List<Prediction> result = new ArrayList();
        for(int i=0; i<numPredictions; i++){
            if(i<summary.size()) {
                result.add(summary.get(i));
            }
        }
        return result;
    }
}
