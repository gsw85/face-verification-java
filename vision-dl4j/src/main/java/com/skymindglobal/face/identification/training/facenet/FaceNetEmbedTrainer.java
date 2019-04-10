package com.skymindglobal.face.identification.training.facenet;

import org.datavec.image.loader.LFWLoader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.FaceNetNN4Small2;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class FaceNetEmbedTrainer {
    private static final Logger log = LoggerFactory.getLogger(FaceNetEmbedTrainer.class);

    private static String unique_id = "lfw_v0";
    private static String modelFilename = new File(".").getAbsolutePath() + "/generated-models/facenet_" + unique_id + ".zip";

    private static int nEpochs = 1;
    private static int batchSize = 48; // depending on your hardware, you will want to increase or decrease
    private static int numExamples = LFWLoader.NUM_IMAGES;
    private static double splitTrainTest = 1.0;

    private static int seed = 123;
    private static int[] inputShape = new int[] {3, 96, 96};
    private static int outputNum = LFWLoader.NUM_LABELS;
    private static IUpdater updater = new Adam(0.1D, 0.9D, 0.999D, 0.01D);
    private static Activation transferFunction = Activation.RELU;
    private static int embeddingSize = 128;
    private static CacheMode cacheMode = CacheMode.NONE;
    private static WorkspaceMode workspaceMode = WorkspaceMode.NONE;
    private static ConvolutionLayer.AlgoMode algoMode = ConvolutionLayer.AlgoMode.NO_WORKSPACE;
    private static boolean TRAINING_MODE = true;
    private static int modelCheckpointInterval = 1;
    private static String trainingUIStoragePath = new File(".").getAbsolutePath() + "/.trainingUI/" + unique_id;

    public static void main(String[] args) throws IOException {

        log.info(modelFilename);
        int[] inputWHC = new int[]{inputShape[2], inputShape[1], inputShape[0]};
        LFWDataSetIterator iter = new LFWDataSetIterator(
                batchSize,
                numExamples,
                inputWHC,
                outputNum,
                false,
                true,
                splitTrainTest,
                new Random(seed)
        );

        if (new File(modelFilename).exists() && TRAINING_MODE) {
            log.info("Load model...");
            ComputationGraph net = ModelSerializer.restoreComputationGraph(modelFilename, true);
            log.info("Continue Training...");
            trainModel(iter, net);
        }
        else
        {
            ComputationGraph net = getFaceNetNN4Small2();
            System.out.println(net.summary());
            trainModel(iter, net);
        }

        log.info("Execution completed.");
    }

    private static void trainModel(LFWDataSetIterator iter, ComputationGraph net) throws IOException {

        UIServer server = UIServer.getInstance();
        StatsStorage storage = new FileStatsStorage(
                new File(trainingUIStoragePath)
        );

        server.attach(storage);
        net.setListeners(
                new ScoreIterationListener(1),
                new StatsListener(storage, 1)
        );

        for (int i = 0; i < nEpochs; i++) {
            iter.reset();
            while (iter.hasNext()) {
                net.fit(iter);
            }
            Evaluation eval = net.evaluate(iter);
            System.out.println("Accuracy: " + eval.accuracy() + " | Precision: " +eval.precision() + " | Recall: " + eval.recall());
            if((i+1)%modelCheckpointInterval==0){
                ModelSerializer.writeModel(net, modelFilename, true);
                log.info("Checkpoint: Model saved!");
            }
            log.info("*** Completed epoch {} ***", i);
        }
    }

    private static ComputationGraph getFaceNetNN4Small2() {
        return new FaceNetNN4Small2(
                seed,
                inputShape,
                outputNum,
                updater,
                transferFunction,
                cacheMode,
                workspaceMode,
                algoMode,
                embeddingSize
        ).init();
    }


}
