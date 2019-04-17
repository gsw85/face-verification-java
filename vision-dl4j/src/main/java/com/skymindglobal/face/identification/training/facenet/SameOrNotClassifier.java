package com.skymindglobal.face.identification.training.facenet;

import com.skymindglobal.face.identification.training.facenet.dataHelpers.LFWCroppedDatasetIterator;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.jetbrains.annotations.NotNull;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class SameOrNotClassifier {

    private static String project = "same_or_not";
    private static String modelFilename = new File(".").getAbsolutePath() + "/generated-models/"+project+".zip";
    private static String embeddingFilename = new File(".").getAbsolutePath() + "/generated-models/embedding.zip";

    private static final Logger log = LoggerFactory.getLogger(SameOrNotClassifier.class);
    private static boolean TRAINING_MODE = true;
    private static ComputationGraph snipped;
    private static int modelCheckpointInterval = 1;
    private static String trainingUIStoragePath = new File(".").getAbsolutePath() + "/.trainingUI/" + project;
    private static int epoch = 20;
    private static double learningRate = 0.01;

    public static void main(String[] args) throws IOException {

        loadEmbeddingNet();
        LFWCroppedDatasetIterator _LFWCroppedDatasetIterator = new LFWCroppedDatasetIterator(
                new File("D:\\Public_Data\\face_recog\\lfw_custom_train_cropped"),
                new File("D:\\Public_Data\\face_recog\\lfw_custom_test_cropped"),
                128,
                349
        );

        RecordReaderDataSetIterator trainIter = _LFWCroppedDatasetIterator.trainIterator();
        RecordReaderDataSetIterator testIter = _LFWCroppedDatasetIterator.testIterator();

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
            log.info("Load model...");
            ComputationGraph net = ModelSerializer.restoreComputationGraph(embeddingFilename, false);
            snipped = new TransferLearning.GraphBuilder(net)
                    .setFeatureExtractor("embeddings") // the L2Normalize vertex and layers below are frozen
                    .removeVertexAndConnections("lossLayer")
                    .setOutputs("embeddings")
                    .build();

        }
    }

    private static void trainModel(RecordReaderDataSetIterator trainIter,RecordReaderDataSetIterator testIter, ComputationGraph net) throws IOException {
//        net.setListeners(new ScoreIterationListener(1));
        setupTrainingUI(net);
        for (int i = 0; i < epoch; i++) {
            trainIter.reset();
            while (trainIter.hasNext()) {
                {
                    log.info("TRAIN");
                    DataSet ds = trainIter.next();
                    ds.shuffle();
                    List<DataSet> pairList = ds.batchBy(2);
                    DataSet MasterDataset = SimilarityModelDataset(pairList);
                    log.info(String.valueOf(MasterDataset.labelCounts()));
                    net.fit(MasterDataset);
//                log.info("trained.");
//                log.info("predicted as (ffw) " +net.feedForward(dummy_ds.getFeatures(), false));
//                log.info("predicted as " +net.output(MasterDataset.getFeatures())[0]);

                    // eval train
                    Evaluation eval = new Evaluation(2);
                    INDArray[] output = net.output(MasterDataset.getFeatures());
                    eval.eval(MasterDataset.getLabels(), output[0]);
                    log.info(eval.stats(false, false));
                }

                {
                    log.info("TEST");
                    // eval test
                    testIter.reset();
                    DataSet ds = testIter.next();
                    ds.shuffle();
                    List<DataSet> pairList = ds.batchBy(2);
                    DataSet MasterDataset = SimilarityModelDataset(pairList);
                    log.info(String.valueOf(MasterDataset.labelCounts()));
                    Evaluation evalTest = new Evaluation(2);
                    INDArray[] output = net.output(MasterDataset.getFeatures());
                    evalTest.eval(MasterDataset.getLabels(), output[0]);
                    log.info(evalTest.stats(false, false));
                }

            }
            if((i+1)%modelCheckpointInterval==0){
                ModelSerializer.writeModel(net, modelFilename, true);
                log.info("Checkpoint: Model saved!");
            }
            log.info("*** Completed epoch {} ***", i);
        }
    }

    private static void setupTrainingUI(ComputationGraph net) {
        UIServer server = UIServer.getInstance();
        StatsStorage storage = new FileStatsStorage(
                new File(trainingUIStoragePath)
        );

        server.attach(storage);
        net.setListeners(
                new StatsListener(storage, 1)
        );
    }

    @NotNull
    private static DataSet SimilarityModelDataset(List<DataSet> pairList) {
        DataSet MasterDataset = new DataSet();
        for (DataSet x: pairList){
            INDArray a_feature = getEmbeddings(x.get(0).getFeatures());
            int a_label = decodeLabelID(x.get(0).getLabels());
            INDArray b_feature = getEmbeddings(x.get(1).getFeatures());
            int b_label = decodeLabelID(x.get(1).getLabels());

            INDArray feature = Nd4j.concat(1, a_feature, b_feature);

            double[] arr;
            if(a_label==b_label){
                arr = new double[]{0.0, 1.0};
//                log.info("samples matched");
            }
            else{
                arr = new double[]{1.0, 0.0};
//                log.info("samples not matched");
            }
            double[] flat = ArrayUtil.flattenDoubleArray(arr);
            int[] shape = new int[]{1,2};
            INDArray label = Nd4j.create(flat, shape,'c');

            DataSet dummy_ds = new DataSet();
            dummy_ds.setFeatures(feature);
            dummy_ds.setLabels(label);

            ArrayList<DataSet> data = new ArrayList<>();
            data.add(MasterDataset);
            data.add(dummy_ds);
            MasterDataset = DataSet.merge(data);
        }
        return MasterDataset;
    }


    private static int decodeLabelID(INDArray encoded) {
        int topX;
        topX = Nd4j.argMax(encoded.getRow(0).dup(), 1).getInt(0, 0);
        encoded.getRow(0).dup().putScalar(0, topX, 0.0D);
        return topX;
    }

    private static ComputationGraphConfiguration getNetworkConfiguration() {

        ComputationGraphConfiguration conf =new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(learningRate,0.9))
                .weightInit(WeightInit.XAVIER)
                .l2(0.001)
                .dropOut(0.5)
                .graphBuilder()
                .addInputs("input")
                .addLayer("l1", new DenseLayer.Builder().nIn(128*2).nOut(64).activation(Activation.IDENTITY).build(), "input")
                .addLayer("b1", new BatchNormalization.Builder().nIn(64).nOut(64).build(),"l1")
                .addLayer("l2", new DenseLayer.Builder().nIn(64).nOut(64).activation(Activation.RELU).build(), "b1")
                .addLayer("l3", new DenseLayer.Builder().nIn(64).nOut(32).activation(Activation.IDENTITY).build(), "l2")
                .addLayer("b2", new BatchNormalization.Builder().nIn(32).nOut(32).build(),"l3")
                .addLayer("l4", new DenseLayer.Builder().nIn(32).nOut(32).activation(Activation.RELU).build(), "b2")
                .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(32).nOut(2).activation(Activation.SOFTMAX).build(), "l4")
                .setOutputs("output")
                .pretrain(false)
                .backprop(true)
                .build();
        return conf;
    }
}
