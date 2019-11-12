package com.skymindglobal.faceverification.identification.feature;

import com.skymindglobal.faceverification.detection.FaceLocalization;
import com.skymindglobal.faceverification.identification.Prediction;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.nd4j.linalg.ops.transforms.Transforms.cosineSim;

/**
 * generates embedding based on pre-build model, reference https://github.com/nyoki-mtl/keras-facenet
 */
public class KerasFaceNetFeatureProvider extends FaceFeatureProvider {
    private static final Logger log = LoggerFactory.getLogger(KerasFaceNetFeatureProvider.class);
    private ComputationGraph model;
    private static ArrayList<LabelFeaturePair> labelFeaturePairList = new ArrayList<>();

    public KerasFaceNetFeatureProvider() throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
//        String simpleMlp = new ClassPathResource("keras/facenet/facenet_keras_weights.h5").getFile().getPath();
        String simpleMlp = new ClassPathResource("keras/facenet/facenet_keras.h5").getFile().getPath();

        model = KerasModelImport.importKerasModelAndWeights(simpleMlp);
    }

    public ArrayList<LabelFeaturePair> setupAnchor(File dictionary) throws IOException, ClassNotFoundException {
//        ImageRecordReader recordReader = new ImageRecordReader(224, 224, 3, new ParentPathLabelGenerator());
//        recordReader.initialize(new FileSplit(dictionary));
//        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 1, 1, dictionary.listFiles().length);
//        List<String> labels = iter.getLabels();
//        generateEmbeddings(iter, labels);
//        return labelFeaturePairList;
        return null;
    }

    private void generateEmbeddings(RecordReaderDataSetIterator iter, List<String> labels) {
//        while (iter.hasNext()) {
//            DataSet Ds = iter.next();
//            INDArray embedding = this.getEmbeddings(Ds.getFeatures());
//            String label = labels.get(decodeLabelID(Ds.getLabels()));
//            labelFeaturePairList.add(new LabelFeaturePair(label, embedding));
//        }
    }

    public INDArray getEmbeddings(INDArray arr) {
//        VGG16ImagePreProcessor _VGG16ImagePreProcessor = new VGG16ImagePreProcessor();
//        _VGG16ImagePreProcessor.preProcess(arr);
        return null; //model.feedForward(arr, false).get("fc8");
    }

    public static IntStream reverseOrderStream(IntStream intStream) {
//        int[] tempArray = intStream.toArray();
//        return IntStream.range(1, tempArray.length + 1).boxed()
//                .mapToInt(i -> tempArray[tempArray.length - i]);
        return null;
    }

    public List<Prediction> predict(Mat image, FaceLocalization faceLocalization, int numPredictions, double threshold, int numSamples) throws IOException {
//        NativeImageLoader nativeImageLoader = new NativeImageLoader();
//        resize(image, image, new Size(224, 224));
//        INDArray _image = nativeImageLoader.asMatrix(image);
//        INDArray anchor = getEmbeddings(_image);
//        List<Prediction> predicted = new ArrayList<>();
//        for (LabelFeaturePair i: labelFeaturePairList){
//            INDArray embed = i.getEmbedding();
//            double distance = cosineSim(anchor, embed);
//            predicted.add(new Prediction(i.getLabel(), distance, faceLocalization));
//        }
//
//        // aggregator - average comparison per class
//        List<Prediction> summary = new ArrayList<>();
//        final Map<String, List<Prediction>> map = predicted.stream().collect(Collectors.groupingBy(p -> p.getLabel()));
//        for (final Map.Entry<String, List<Prediction>> entry : map.entrySet()) {
////            final double max = entry.getValue().stream()
////                    .mapToDouble(p -> p.getScore()).max().getAsDouble();
//
//            double topNAvg = reverseOrderStream(entry
//                    .getValue()
//                    .stream()
//                    .mapToInt(p -> (int) (p.getScore() * 10000))
//                    .sorted()
//            )
//                    .limit(numSamples)
//                    .mapToDouble(num -> (double) num / 10000)
//                    .average()
//                    .getAsDouble();
//            if(topNAvg >= threshold) {
//                summary.add(new Prediction(entry.getKey(), topNAvg, faceLocalization));
//            }
//        }
//
//        // sort and select top N
//        summary.sort(Comparator.comparing(Prediction::getScore));
//        Collections.reverse(summary);
//
//        List<Prediction> result = new ArrayList();
//        for(int i=0; i<numPredictions; i++){
//            if(i<summary.size()) {
//                result.add(summary.get(i));
//            }
//        }
//        return result;
        return null;
    }
}
