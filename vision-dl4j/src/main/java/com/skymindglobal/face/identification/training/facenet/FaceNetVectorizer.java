package com.skymindglobal.face.identification.training.facenet;

import com.skymindglobal.face.toolkit.CSVUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
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
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import static org.nd4j.linalg.ops.transforms.Transforms.*;

public class FaceNetVectorizer {
    private static final Logger log = LoggerFactory.getLogger(FaceNetVectorizer.class);

    private static String modelFilename = new File(".").getAbsolutePath() + "/generated-models/facenet_embeddingfacenet_embedding.zip";
    private static String CSV_FILE_NAME = new File(".").getAbsolutePath() + "/generated-models/aligned_distance.csv";
    private static ComputationGraph net;

    public static void main(String[] args) throws IOException {

        File dir = new File("D:\\Public_Data\\face_recog\\lfw_test_align_96");
        ImageRecordReader recordReaderSource = new ImageRecordReader(96, 96, 3, new ParentPathLabelGenerator());
        recordReaderSource.initialize(new FileSplit(dir));
        RecordReaderDataSetIterator source = new RecordReaderDataSetIterator(recordReaderSource, 1, 1, dir.listFiles().length);

        ImageRecordReader recordReaderTarget = new ImageRecordReader(96, 96, 3, new ParentPathLabelGenerator());
        recordReaderTarget.initialize(new FileSplit(dir));
        RecordReaderDataSetIterator target = new RecordReaderDataSetIterator(recordReaderTarget, 1, 1, dir.listFiles().length);


        if (new File(modelFilename).exists()) {
            log.info("Load model...");
            net = ModelSerializer.restoreComputationGraph(modelFilename, true);
        }

        ComputationGraph snipped = new TransferLearning.GraphBuilder(net)
                .setFeatureExtractor("embeddings") // the L2Normalize vertex and layers below are frozen
                .removeVertexAndConnections("lossLayer")
                .setOutputs("embeddings")
                .build();

        FileWriter writer = new FileWriter(CSV_FILE_NAME);
        CSVUtils.writeLine(writer, Arrays.asList("source","target", "sameSourceTarget", "cosineSim", "cosineDistance", "hammingDistance", "jaccardDistance", "manhattanDistance", "euclideanDistance"));

        source.reset();
        int sourceCount = 0;
        while (source.hasNext()){
            DataSet sourceDs = source.next();
            INDArray source_embedding = snipped.feedForward(sourceDs.getFeatures(), false).get("embeddings");
            target.reset();
            int targetCount = 0;
            while (target.hasNext()) {
                DataSet targetDs = target.next();
                INDArray target_embedding = snipped.feedForward(targetDs.getFeatures(), false).get("embeddings");

                int sameSourceTarget = 0;
                if(sourceCount == targetCount){
                    sameSourceTarget = 1;
                }
                CSVUtils.writeLine(
                        writer,
                        Arrays.asList(
                                String.valueOf(decodeLabelID(sourceDs.getLabels())),
                                String.valueOf(decodeLabelID(targetDs.getLabels())),
                                String.valueOf(sameSourceTarget),
                                String.valueOf(cosineSim(source_embedding, target_embedding)),
                                String.valueOf(cosineDistance(source_embedding, target_embedding)),
                                String.valueOf(hammingDistance(source_embedding, target_embedding)),
                                String.valueOf(jaccardDistance(source_embedding, target_embedding)),
                                String.valueOf(manhattanDistance(source_embedding, target_embedding)),
                                String.valueOf(euclideanDistance(source_embedding, target_embedding))
                        )
                );
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
