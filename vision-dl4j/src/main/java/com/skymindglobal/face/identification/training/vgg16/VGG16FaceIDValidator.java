package com.skymindglobal.face.identification.training.vgg16;

import com.skymindglobal.face.identification.Prediction;
import com.skymindglobal.face.identification.training.vgg16.dataHelpers.VGG16DatasetIterator;
import com.skymindglobal.face.toolkit.CSVUtils;
import com.skymindglobal.face.toolkit.LabelManager;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.management.AttributeList;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.jcodec.common.Assert.assertTrue;

public class VGG16FaceIDValidator {
    private static final Logger log = LoggerFactory.getLogger(VGG16FaceIDValidator.class);
    private static final String EVALUATE = "EVALUATE";
    private static final String EXPORT_CSV = "EXPORT_CSV";
    private static final String TRAINING_UI = "TRAINING_UI";

    private static String unique_id = "vgg16_faceid_v13";
    private static String modelFilename = new File(".").getAbsolutePath() + "/generated-models/" + unique_id + ".zip";
    private static String labelFilename = new File(".").getAbsolutePath() + "/generated-models/" + unique_id + ".lbl";
    private static String csvFilename = new File(".").getAbsolutePath() + "/generated-models/" + unique_id + ".csv";
    private static String trainingUIStoragePath = new File(".").getAbsolutePath() + "/.trainingUI/" + unique_id;
    private static ComputationGraph model;
    private static String[] labels;
    private static String mode = VGG16FaceIDValidator.EVALUATE;

    public static void main(String[] args) throws Exception {
        
        log.info("Load data...");
        VGG16DatasetIterator _VGG16DatasetIterator = new VGG16DatasetIterator(
                new File("D:\\Public_Data\\face_recog\\lfw_custom_train_cropped"),
                new File("D:\\Public_Data\\face_recog\\lfw_custom_test_cropped"),
                1,
                1 // get all samples
        );

        RecordReaderDataSetIterator trainIter = _VGG16DatasetIterator.trainIterator();
        trainIter.setPreProcessor( new VGG16ImagePreProcessor());

        RecordReaderDataSetIterator testIter = _VGG16DatasetIterator.testIterator();
        testIter.setPreProcessor( new VGG16ImagePreProcessor());

        labels = LabelManager.importLabels(labelFilename);
        if(new File(modelFilename).exists()){
            log.info("Load model...");
            model = ModelSerializer.restoreComputationGraph(modelFilename);
        }

        switch (mode){
            case VGG16FaceIDValidator.EXPORT_CSV:
                writeCSV(csvFilename, testIter);
                break;
            case VGG16FaceIDValidator.TRAINING_UI:
                UIServer server = UIServer.getInstance();
                StatsStorage storage = new FileStatsStorage(
                        new File(trainingUIStoragePath)
                );
                server.attach(storage);
                break;
            case VGG16FaceIDValidator.EVALUATE:
                log.info(model.evaluate(trainIter).stats(true, true) + "\n");
                log.info(model.evaluate(testIter).stats(true, true) + "\n");
                break;
            default:
                break;
        }

//        displayEach(testIter, model);
//        Evaluation eval = new Evaluation(labels.length);
//        DataSet data = testIter.next();
//        INDArray[] output = model.output(data.getFeatures());
//        for (INDArray i: output){
//            log.info(i.toString());
//        }
//        eval.eval(data.getLabels(), output[0]);
//        System.out.println(eval.confusionToString());

    }

    private static void writeData(FileWriter writer, RecordReaderDataSetIterator testIter, ComputationGraph model) throws IOException {
        CSVUtils.writeLine(writer,Arrays.asList("label","predicted"));
        log.info("Collecting...");
        while (testIter.hasNext()) {
            DataSet record = testIter.next();
            INDArray[] output = model.output(record.getFeatures());
//            System.out.println(decodeLabelID(record.getLabels()) + "," +predictionsToID(decodePredictions(output[0], 1)));
            CSVUtils.writeLine(
                    writer,
                    Arrays.asList(
                            String.valueOf(decodeLabelID(record.getLabels())),
                            String.valueOf(predictionsToID(decodePredictions(output[0], 1).get(0).getLabel()))
                    )
            );
        }
    }

    private static void writeCSV(String CSV_FILE_NAME, RecordReaderDataSetIterator testIter) throws IOException {
        FileWriter writer = new FileWriter(CSV_FILE_NAME);
        writeData(writer, testIter, model);
        writer.flush();
        writer.close();
    }

    public static String convertToCSV(String[] data) {
        return Stream.of(data)
                .map(VGG16FaceIDValidator::escapeSpecialCharacters)
                .collect(Collectors.joining(","));
    }

    public static String escapeSpecialCharacters(String data) {
        String escapedData = data.replaceAll("\\R", " ");
        if (data.contains(",") || data.contains("\"") || data.contains("'")) {
            data = data.replace("\"", "\"\"");
            escapedData = "\"" + data + "\"";
        }
        return escapedData;
    }

    private static void displayEach(RecordReaderDataSetIterator testIter, ComputationGraph model) throws IOException {
        while (testIter.hasNext()) {
            DataSet record = testIter.next();
            INDArray[] output = model.output(record.getFeatures());
            log.info(
                    decodeLabel(record.getLabels()) + " predicted as " +
                            predictionsToString(decodePredictions(output[0], 1))
            );
        }
    }

    private static String decodeLabel(INDArray encoded) {
        int topX = Nd4j.argMax(encoded, 1).getInt(0, 0);
        return labels[topX];
    }

    private static int decodeLabelID(INDArray encoded) {
        int topX;
        topX = Nd4j.argMax(encoded.getRow(0).dup(), 1).getInt(0, 0);
        encoded.getRow(0).dup().putScalar(0, topX, 0.0D);
        return topX;
    }

    private static String predictionsToString(List<Prediction> predictions) {
        StringBuilder builder = new StringBuilder();
        for (Prediction prediction : predictions) {
            builder.append(prediction.toString());
//            builder.append('\n');
        }
        return builder.toString();
    }

    private static int predictionsToID(String predicted_label) {
        int x=0;
        for (String i: labels) {
            if (i.equals(predicted_label)) {
                break;
            }
            x++;
        }
        return x;
    }

    private static List<Prediction> decodePredictions(INDArray encodedPredictions, int numPredicted) throws IOException {
        List<Prediction> decodedPredictions = new ArrayList<>();
        int[] topX = new int[numPredicted];
        float[] topXProb = new float[numPredicted];

        int i = 0;
        for (INDArray currentBatch = encodedPredictions.getRow(0).dup(); i < numPredicted; ++i) {

            topX[i] = Nd4j.argMax(currentBatch, 1).getInt(0, 0);
            topXProb[i] = currentBatch.getFloat(0, topX[i]);
            currentBatch.putScalar(0, topX[i], 0.0D);
            decodedPredictions.add(new Prediction(labels[topX[i]], (topXProb[i] * 100.0F)));
        }
        return decodedPredictions;
    }


}
