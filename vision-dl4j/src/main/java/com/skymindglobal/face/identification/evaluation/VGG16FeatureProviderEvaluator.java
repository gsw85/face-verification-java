package com.skymindglobal.face.identification.evaluation;

import com.skymindglobal.face.identification.feature.VGG16FeatureProvider;
import com.skymindglobal.face.toolkit.CSVUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.metadata.RecordMetaDataImageURI;
import org.datavec.api.records.metadata.RecordMetaDataURI;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

import static org.nd4j.linalg.ops.transforms.Transforms.*;
import static org.nd4j.linalg.ops.transforms.Transforms.euclideanDistance;

public class VGG16FeatureProviderEvaluator {
    private static final Logger log = LoggerFactory.getLogger(VGG16FeatureProviderEvaluator.class);

    private static VGG16FeatureProvider _VGG16FeatureProvider;
    private static int VGG16_HEIGHT = 224;
    private static int VGG16_WIDTH = 224;
    private static int CHANNEL = 3;

    private static String CSV_FILE_NAME = new File(".").getAbsolutePath() + "/generated-models/vgg16_embedding_evaluation.csv";
    private static List<String> labels;
    private static int batchSize = 64;

    public static void main(String[] args) throws IOException {
        _VGG16FeatureProvider = new VGG16FeatureProvider();

        File dir = new File("D:\\Public_Data\\face_recog\\vgg16\\train_08_topN");

        ImageRecordReader recordReaderSource = new ImageRecordReader(VGG16_HEIGHT, VGG16_WIDTH, CHANNEL, new ParentPathLabelGenerator());
        recordReaderSource.initialize(new FileSplit(dir));

        RecordReaderDataSetIterator source = new RecordReaderDataSetIterator(recordReaderSource, batchSize, 1, dir.listFiles().length);

        labels = source.getLabels();
        ImageRecordReader recordReaderTarget = new ImageRecordReader(VGG16_HEIGHT, VGG16_WIDTH, CHANNEL, new ParentPathLabelGenerator());
        recordReaderTarget.initialize(new FileSplit(dir));
        RecordReaderDataSetIterator target = new RecordReaderDataSetIterator(recordReaderTarget, batchSize, 1, dir.listFiles().length);

        FileWriter writer = new FileWriter(CSV_FILE_NAME);
        CSVUtils.writeLine(
                writer,
                Arrays.asList(
                        "source_id",
                        "source_name",
                        "source_uri",
                        "target_id",
                        "target_name",
                        "target_uri",
                        "same_source_target",
                        "cosine_sim",
                        "euclidean_distance"
                )
        );

        source.setCollectMetaData(true);
        target.setCollectMetaData(true);

        source.reset();
        while (source.hasNext()){
            DataSet sourceDs = source.next();
            INDArray source_embedding = _VGG16FeatureProvider.getEmbeddings(sourceDs.getFeatures());
            target.reset();
            while (target.hasNext()) {
                DataSet targetDs = target.next();
                INDArray target_embedding = _VGG16FeatureProvider.getEmbeddings(targetDs.getFeatures());
                for(int i=0; i< source_embedding.shape()[0]; i++){
                    for(int j=0; j< target_embedding.shape()[0]; j++) {
                        RecordMetaDataURI sourceMetadata = (RecordMetaDataURI) sourceDs.getExampleMetaData().get(i);
                        RecordMetaDataURI targetMetadata = (RecordMetaDataURI) targetDs.getExampleMetaData().get(j);
                        INDArray sourceArr = source_embedding.getRow(i);
                        INDArray targetArr = target_embedding.getRow(j);
                        INDArray sourceLblArr = sourceDs.getLabels().getRow(i);
                        INDArray targetLblArr = targetDs.getLabels().getRow(j);

                        int sameSourceTarget = 0;
                        if (sourceMetadata.getURI().toString().equals(targetMetadata.getURI().toString())){
                            sameSourceTarget = 1;
                        }

                        CSVUtils.writeLine(
                                writer,
                                Arrays.asList(
                                        String.valueOf(decodeLabelID(sourceLblArr)),
                                        labels.get(decodeLabelID(sourceLblArr)),
                                        sourceMetadata.getURI().toString(),
                                        String.valueOf(decodeLabelID(targetLblArr)),
                                        labels.get(decodeLabelID(targetLblArr)),
                                        targetMetadata.getURI().toString(),
                                        String.valueOf(sameSourceTarget),
                                        String.valueOf(cosineSim(sourceArr, targetArr)),
                                        String.valueOf(euclideanDistance(sourceArr, targetArr))
                                )
                        );
                    }
                }
            }
        }
        writer.flush();
        writer.close();
    }

    private static int decodeLabelID(INDArray encoded) {
        int topX;
        topX = Nd4j.argMax(encoded.getRow(0).dup(), 1).getInt(0, 0);
        encoded.getRow(0).dup().putScalar(0, topX, 0.0D);
        return topX;
    }
}
