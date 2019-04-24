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

public class VGG16FeatureProviderBatchEvaluator {
    private static final Logger log = LoggerFactory.getLogger(VGG16FeatureProviderBatchEvaluator.class);

    private static VGG16FeatureProvider _VGG16FeatureProvider;
    private static int VGG16_HEIGHT = 224;
    private static int VGG16_WIDTH = 224;
    private static int CHANNEL = 3;
    private static List<String> labels;
    private static int batchSize = 4382;

    public static void main(String[] args) throws IOException {
        _VGG16FeatureProvider = new VGG16FeatureProvider();

        File dir = new File("D:\\Public_Data\\face_recog\\vgg16\\train_08_topN");
        ImageRecordReader recordReaderSource = new ImageRecordReader(VGG16_HEIGHT, VGG16_WIDTH, CHANNEL, new ParentPathLabelGenerator());
        recordReaderSource.initialize(new FileSplit(dir));
        RecordReaderDataSetIterator source = new RecordReaderDataSetIterator(recordReaderSource, batchSize, 1, dir.listFiles().length);
        labels = source.getLabels();

        source.setCollectMetaData(true);
        source.reset();
        int counter = 0;
        while (source.hasNext()){
            FileWriter writer = new FileWriter(new File(".").getAbsolutePath() +
                    "/generated-models/result_" + counter + ".csv");
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

            // load batch data
            DataSet sourceDs = source.next();
            INDArray source_embedding = _VGG16FeatureProvider.getEmbeddings(sourceDs.getFeatures());
            for(int i=0; i< source_embedding.shape()[0]; i++){
                for(int j=0; j< source_embedding.shape()[0]; j++) {
                    RecordMetaDataURI sourceMetadata = (RecordMetaDataURI) sourceDs.getExampleMetaData().get(i);
                    RecordMetaDataURI targetMetadata = (RecordMetaDataURI) sourceDs.getExampleMetaData().get(j);
                    INDArray sourceFeatures = source_embedding.getRow(i);
                    INDArray targetFeatures = source_embedding.getRow(j);
                    INDArray sourceLblArr = sourceDs.getLabels().getRow(i);
                    INDArray targetLblArr = sourceDs.getLabels().getRow(j);

                    int sameSourceTarget = 0;
                    if (sourceMetadata.getURI().toString().equals(targetMetadata.getURI().toString())) {
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
                                    String.valueOf(cosineSim(sourceFeatures, targetFeatures)),
                                    String.valueOf(euclideanDistance(sourceFeatures, targetFeatures))
                            )
                    );
                }
            }
            writer.flush();
            writer.close();
            counter++;
        }
    }

    private static int decodeLabelID(INDArray encoded) {
        int topX;
        topX = Nd4j.argMax(encoded.getRow(0).dup(), 1).getInt(0, 0);
        encoded.getRow(0).dup().putScalar(0, topX, 0.0D);
        return topX;
    }
}
