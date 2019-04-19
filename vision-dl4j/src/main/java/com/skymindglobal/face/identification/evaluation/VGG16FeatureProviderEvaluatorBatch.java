package com.skymindglobal.face.identification.evaluation;

import com.skymindglobal.face.identification.feature.VGG16FeatureProvider;
import com.skymindglobal.face.toolkit.CSVUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.metadata.RecordMetaDataURI;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.nd4j.linalg.ops.transforms.Transforms.cosineSim;
import static org.nd4j.linalg.ops.transforms.Transforms.euclideanDistance;

public class VGG16FeatureProviderEvaluatorBatch {
    private static final Logger log = LoggerFactory.getLogger(VGG16FeatureProviderEvaluatorBatch.class);

    private static VGG16FeatureProvider _VGG16FeatureProvider;
    private static int VGG16_HEIGHT = 224;
    private static int VGG16_WIDTH = 224;
    private static int CHANNEL = 3;

    private static String CSV_FILE_NAME = new File(".").getAbsolutePath() + "/generated-models/vgg16_embedding_evaluation.csv";
    private static List<String> labels;

    public static void main(String[] args) throws IOException {
        _VGG16FeatureProvider = new VGG16FeatureProvider();

        File dir = new File("D:\\Public_Data\\face_recog\\vgg16\\train_08_topN");

        ImageRecordReader recordReaderSource = new ImageRecordReader(VGG16_HEIGHT, VGG16_WIDTH, CHANNEL, new ParentPathLabelGenerator());
        recordReaderSource.initialize(new FileSplit(dir));
        RecordReaderDataSetIterator source = new RecordReaderDataSetIterator(recordReaderSource, 1, 1, dir.listFiles().length);

        labels = source.getLabels();
        ImageRecordReader recordReaderTarget = new ImageRecordReader(VGG16_HEIGHT, VGG16_WIDTH, CHANNEL, new ParentPathLabelGenerator());
        recordReaderTarget.initialize(new FileSplit(dir));
        RecordReaderDataSetIterator target = new RecordReaderDataSetIterator(recordReaderTarget, 1, 1, dir.listFiles().length);

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
        int sourceCount = 0;
        while (source.hasNext()){
            DataSet sourceDs = source.next();
            RecordMetaDataURI sourceMetadata = (RecordMetaDataURI) sourceDs.getExampleMetaData().get(0);
            INDArray source_embedding = _VGG16FeatureProvider.getEmbeddings(sourceDs.getFeatures());
            target.reset();
            int targetCount = 0;
            while (target.hasNext()) {
                DataSet targetDs = target.next();
                RecordMetaDataURI targetMetadata = (RecordMetaDataURI) targetDs.getExampleMetaData().get(0);
                INDArray target_embedding = _VGG16FeatureProvider.getEmbeddings(targetDs.getFeatures());

                int sameSourceTarget = 0;
                if(sourceCount == targetCount){
                    sameSourceTarget = 1;
                }
                CSVUtils.writeLine(
                        writer,
                        Arrays.asList(
                                String.valueOf(decodeLabelID(sourceDs.getLabels())),
                                labels.get(decodeLabelID(sourceDs.getLabels())),
                                sourceMetadata.getURI().toString(),
                                String.valueOf(decodeLabelID(targetDs.getLabels())),
                                labels.get(decodeLabelID(targetDs.getLabels())),
                                targetMetadata.getURI().toString(),
                                String.valueOf(sameSourceTarget),
                                String.valueOf(cosineSim(source_embedding, target_embedding)),
                                String.valueOf(euclideanDistance(source_embedding, target_embedding))
                        )
                );
                log.info(sourceCount + ", "+ targetCount);
                targetCount++;
            }
            sourceCount++;
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
