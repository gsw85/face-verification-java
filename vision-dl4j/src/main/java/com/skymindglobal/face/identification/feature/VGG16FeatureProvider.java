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
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.nd4j.linalg.ops.transforms.Transforms.cosineSim;
import static org.nd4j.linalg.ops.transforms.Transforms.euclideanDistance;

public class VGG16FeatureProvider extends FaceFeatureProvider {
    private static final Logger log = LoggerFactory.getLogger(VGG16FeatureProvider.class);

    private static final int channels = 3;
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private ComputationGraph model;
    private static ArrayList<LabelFeaturePair> labelFeaturePairList = new ArrayList<>();
    private static final String random_uuid = "c5cc2a8e-d630-42a6-91d8-54c3df95e05e";

    public VGG16FeatureProvider() throws IOException {
        model = (ComputationGraph) VGG16.builder().build().initPretrained(PretrainedType.VGGFACE);
        log.info(model.summary());
    }

    public ArrayList<LabelFeaturePair> setupAnchor(File dictionary) throws IOException, ClassNotFoundException {
        ImageRecordReader recordReader = new ImageRecordReader(224, 224, channels, labelMaker);
        recordReader.initialize(new FileSplit(dictionary));
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 1, 1, dictionary.listFiles().length);
        List<String> labels = iter.getLabels();

//        File file = new File(new File(".").getAbsolutePath() + "/generated-models/" + random_uuid + ".bin");
//        if(file.exists()){
//            FileInputStream _FileInputStream = new FileInputStream(file);
//            ObjectInputStream _ObjectInputStream = new ObjectInputStream(_FileInputStream);
//            labelFeaturePairList = (ArrayList<LabelFeaturePair>) _ObjectInputStream.readObject();
//            _ObjectInputStream.close();
//        }
//        else{
            generateEmbeddings(iter, labels);

//            FileOutputStream _FileOutputStream = new FileOutputStream(file);
//            ObjectOutputStream _ObjectOutputStream = new ObjectOutputStream(_FileOutputStream);
//            _ObjectOutputStream.writeObject(labelFeaturePairList);
//            _ObjectOutputStream.close();
//        }

        return labelFeaturePairList;
    }

    private void generateEmbeddings(RecordReaderDataSetIterator iter, List<String> labels) {
        while (iter.hasNext()) {
            DataSet Ds = iter.next();
            INDArray embedding = this.getEmbeddings(Ds.getFeatures());
            String label = labels.get(decodeLabelID(Ds.getLabels()));
            labelFeaturePairList.add(new LabelFeaturePair(label, embedding));
        }
    }

    public INDArray getEmbeddings(INDArray arr) {
        VGG16ImagePreProcessor _VGG16ImagePreProcessor = new VGG16ImagePreProcessor();
        _VGG16ImagePreProcessor.preProcess(arr);
        return model.feedForward(arr, false).get("fc8");
    }

    public List<Prediction> predict(opencv_core.Mat image, FaceLocalization faceLocalization, int numPredictions, double threshold) throws IOException {
        NativeImageLoader nativeImageLoader = new NativeImageLoader();
        resize(image, image, new opencv_core.Size(224, 224));
        INDArray _image = nativeImageLoader.asMatrix(image);
        INDArray anchor = getEmbeddings(_image);
        List<Prediction> predicted = new ArrayList<>();
        for (LabelFeaturePair i: labelFeaturePairList){
            INDArray embed = i.getEmbedding();
            double distance = cosineSim(anchor, embed);
            predicted.add(new Prediction(i.getLabel(), distance, faceLocalization));
        }

        // aggregator - average comparison per class
        List<Prediction> summary = new ArrayList<>();
        final Map<String, List<Prediction>> map = predicted.stream().collect(Collectors.groupingBy(p -> p.getLabel()));
        for (final Map.Entry<String, List<Prediction>> entry : map.entrySet()) {
//            final double max = entry.getValue().stream()
//                    .mapToDouble(p -> p.getScore()).max().getAsDouble();
            final double topNAvg = entry.getValue().stream()
                    .mapToDouble(p -> p.getScore()).sorted().limit(3).average().getAsDouble();
            if(topNAvg > threshold) {
                summary.add(new Prediction(entry.getKey(), topNAvg, faceLocalization));
            }
        }

        // sort and select top N
        summary.sort(Comparator.comparing(Prediction::getScore));
        Collections.reverse(summary);

        List<Prediction> result = new ArrayList();
        for(int i=0; i<numPredictions; i++){
            if(i<summary.size()) {
                result.add(summary.get(i));
            }
        }
        return result;
    }
}
