package com.skymindglobal.face.identification.training.facenet;

import com.skymindglobal.face.identification.training.facenet.dataHelpers.LFWCroppedDatasetIterator;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class LFWClassifier {

    private static String project = "lfw_classification_tryrun";
    private static String modelFilename = new File(".").getAbsolutePath() + "/generated-models/"+project+".zip";
    private static String embeddingFilename = new File(".").getAbsolutePath() + "/generated-models/embedding.zip";

    private static final Logger log = LoggerFactory.getLogger(LFWClassifier.class);
    private static boolean TRAINING_MODE = true;
    private static ComputationGraph snipped;
    private static int modelCheckpointInterval = 20;
    private static String trainingUIStoragePath = new File(".").getAbsolutePath() + "/.trainingUI/" + project;
    private static int epoch = 200;
    private static double learningRate = 0.1;
    private static int nClasses;
    private static int batchSize = 256;
    private static double dropOut = 1.0;
    private static double l2 = 0.0;

    public static void main(String[] args) throws IOException {
        Nd4j.getMemoryManager().setAutoGcWindow(10000);

        loadEmbeddingNet();
        LFWCroppedDatasetIterator _LFWCroppedDatasetIterator = new LFWCroppedDatasetIterator(
                new File("D:\\Public_Data\\face_recog\\lfw_train_96"),
                new File("D:\\Public_Data\\face_recog\\lfw_test_96"),
                472,
                465
        );

        RecordReaderDataSetIterator trainIter = _LFWCroppedDatasetIterator.trainIterator();
        RecordReaderDataSetIterator testIter = _LFWCroppedDatasetIterator.testIterator();

        nClasses = _LFWCroppedDatasetIterator.getNumClass();

        if (new File(modelFilename).exists() && TRAINING_MODE) {
            log.info("Load model...");
            ComputationGraph net = ModelSerializer.restoreComputationGraph(modelFilename, true);
            log.info("Continue Training...");
            trainModel(trainIter,testIter, net);
        }
        else
        {
            ComputationGraphConfiguration  conf = getNetworkConfiguration();
            ComputationGraph net = new ComputationGraph(conf);
            net.init();
            System.out.println(net.summary());
            trainModel(trainIter,testIter, net);
        }
    }
    private static INDArray getEmbeddings(INDArray arr) {
        return snipped.feedForward(arr, false).get("embeddings");
    }

    private static void loadEmbeddingNet() throws IOException {
        if (new File(embeddingFilename).exists()) {
            log.info("Load embedding model...");
            ComputationGraph net = ModelSerializer.restoreComputationGraph(embeddingFilename, false);
            snipped = new TransferLearning.GraphBuilder(net)
                    .setFeatureExtractor("embeddings") // the L2Normalize vertex and layers below are frozen
                    .removeVertexAndConnections("lossLayer")
                    .setOutputs("embeddings")
                    .build();

        }
    }

    private static void trainModel(RecordReaderDataSetIterator trainIter,RecordReaderDataSetIterator testIter, ComputationGraph net) throws IOException {
        setupTrainingUI(net);
        for (int i = 0; i < epoch; i++) {
            trainIter.reset();
            DataSet train = trainIter.next();
            train.shuffle();
            List<DataSet> record = train.batchBy(batchSize);
            for (DataSet x : record) {

                // train
                DataSet dummy_ds = new DataSet();
                dummy_ds.setFeatures(getEmbeddings(x.getFeatures()));
                dummy_ds.setLabels(x.getLabels());
                net.fit(dummy_ds);

                if((i+1)% evalInterval() ==0) {
                    //eval train
                    Evaluation eval = new Evaluation(nClasses);
                    INDArray[] output = net.output(dummy_ds.getFeatures());
                    eval.eval(dummy_ds.getLabels(), output[0]);
                    log.info("Train | Accuracy: " + eval.accuracy());
                }
            }
            {
                if((i+1)% evalInterval() ==0) {
                    // eval test
                    testIter.reset();
                    DataSet ds = testIter.next();
                    DataSet dummy_ds = new DataSet();
                    dummy_ds.setFeatures(getEmbeddings(ds.getFeatures()));
                    dummy_ds.setLabels(ds.getLabels());

                    Evaluation eval = new Evaluation(nClasses);
                    INDArray[] output = net.output(dummy_ds.getFeatures());
                    eval.eval(dummy_ds.getLabels(), output[0]);
                    log.info("Test  | Accuracy: " + eval.accuracy());
                }
            }

            if((i+1)%modelCheckpointInterval==0){
                ModelSerializer.writeModel(net, modelFilename, true);
                log.info("Checkpoint: Model saved!");
            }
            log.info("*** Completed epoch {} ***", i);
        }
    }

    private static int evalInterval() {
        return 10;
    }

    private static void setupTrainingUI(ComputationGraph net) {
        UIServer server = UIServer.getInstance();
        StatsStorage storage = new FileStatsStorage(
                new File(trainingUIStoragePath)
        );

        server.attach(storage);
        net.setListeners(
                new StatsListener(storage, 1),
                new ScoreIterationListener(1)
        );
    }

    private static ComputationGraphConfiguration getNetworkConfiguration() {



        ComputationGraphConfiguration conf =new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .l2(l2)
                .dropOut(dropOut)
                .graphBuilder()
                .addInputs("input")
                .addLayer("l1", new DenseLayer.Builder().nIn(128).nOut(128).activation(Activation.IDENTITY).build(), "input")
                .addLayer("b1", new BatchNormalization.Builder().nIn(128).nOut(128).build(),"l1")
                .addLayer("l2", new DenseLayer.Builder().nIn(128).nOut(64).activation(Activation.LEAKYRELU).build(), "b1")
                .addLayer("l3", new DenseLayer.Builder().nIn(64).nOut(64).activation(Activation.IDENTITY).build(), "l2")
                .addLayer("b2", new BatchNormalization.Builder().nIn(64).nOut(64).build(),"l3")
                .addLayer("l4", new DenseLayer.Builder().nIn(64).nOut(32).activation(Activation.LEAKYRELU).build(), "b2")
                .addLayer("l5", new DenseLayer.Builder().nIn(32).nOut(32).activation(Activation.IDENTITY).build(), "l4")
                .addLayer("b3", new BatchNormalization.Builder().nIn(32).nOut(32).build(),"l5")
                .addLayer("l6", new DenseLayer.Builder().nIn(32).nOut(16).activation(Activation.LEAKYRELU).build(), "b3")
                .addLayer("l7", new DenseLayer.Builder().nIn(16).nOut(16).activation(Activation.IDENTITY).build(), "l6")
                .addLayer("b4", new BatchNormalization.Builder().nIn(16).nOut(16).build(),"l7")
                .addLayer("l8", new DenseLayer.Builder().nIn(16).nOut(8).activation(Activation.LEAKYRELU).build(), "b4")
                .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(8).nOut(nClasses).activation(Activation.SOFTMAX).build(), "l8")
                .setOutputs("output")
                .pretrain(false)
                .backprop(true)
                .build();
        return conf;
    }
}
