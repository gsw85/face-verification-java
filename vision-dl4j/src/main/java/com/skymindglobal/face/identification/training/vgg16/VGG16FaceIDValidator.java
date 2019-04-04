package com.skymindglobal.face.identification.training.vgg16;

import com.skymindglobal.face.identification.training.vgg16.dataHelpers.VGG16DatasetIterator;
import com.skymindglobal.face.toolkit.LabelManager;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Arrays;
import java.util.List;

public class VGG16FaceIDValidator {
    private static final Logger log = LoggerFactory.getLogger(VGG16FaceIDValidator.class);

    private static String unique_id = "vgg16_faceid_v1";
    private static String modelFilename = new File(".").getAbsolutePath() + "/generated-models/" + unique_id + ".zip";
    private static String labelFilename = new File(".").getAbsolutePath() + "/generated-models/" + unique_id + ".lbl";
    private static String trainingUIStoragePath = new File(".").getAbsolutePath() + "/.trainingUI/" + unique_id;
    private static int nClasses;
    private static ComputationGraph model;
    private static String[] labels;

    public static void main(String[] args) throws Exception {
        
        log.info("Load data...");
        VGG16DatasetIterator _VGG16DatasetIterator = new VGG16DatasetIterator(
                new File("D:\\Public_Data\\face_recog\\lfw_custom_train_cropped"),
                new File("D:\\Public_Data\\face_recog\\lfw_custom_test_cropped"),
                0,
                1 // get all samples
        );
        nClasses = _VGG16DatasetIterator.getNumClass();
        RecordReaderDataSetIterator testIter = _VGG16DatasetIterator.testIterator();
        testIter.setPreProcessor( new VGG16ImagePreProcessor());

        labels = LabelManager.importLabels(labelFilename);
        if(new File(modelFilename).exists()){
            log.info("Load model...");
            model = ModelSerializer.restoreComputationGraph(modelFilename);
        }
    }
}
