package com.skymindglobal.face.identification.feature;

import com.skymindglobal.face.detection.FaceLocalization;
import com.skymindglobal.face.identification.Prediction;
import com.skymindglobal.face.toolkit.SkilClient;
import org.bytedeco.javacpp.opencv_core;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class FaceNetFeatureProvider extends FaceFeatureProvider {
    private static ArrayList<LabelFeaturePair> labelFeaturePairList = new ArrayList<>();
    private static final Logger log = LoggerFactory.getLogger(FaceNetFeatureProvider.class);
    private final SkilClient _SkilClient;

    public FaceNetFeatureProvider(){
        _SkilClient = new SkilClient("admin", "adminpassword", "localhost", "9008");
    }

    public ArrayList<LabelFeaturePair> setupAnchor(File dictionary) throws IOException, ClassNotFoundException {
        ImageRecordReader recordReader = new ImageRecordReader(224, 224, 3, new ParentPathLabelGenerator());
        recordReader.initialize(new FileSplit(dictionary));
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 1, 1, dictionary.listFiles().length);
        List<String> labels = iter.getLabels();
        generateEmbeddings(iter, labels);
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
        String result = _SkilClient.classify(
                "http://localhost:9008/endpoints/facenet/model/facenetfacevgg/default/",
                arr
        );

        log.info(result);
        return null;
    }

    public List<Prediction> predict(opencv_core.Mat image, FaceLocalization faceLocalization, int numPredictions, double threshold, int numSamples, int minSupport) throws IOException {

        return null;
    }
}
