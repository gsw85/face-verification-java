package com.skymindglobal.face.identification.training.facenet;

import org.datavec.image.loader.LFWLoader;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.zoo.model.FaceNetNN4Small2;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.Random;

public class FaceNetFaceIDTrainer {
    private static final Logger log = LoggerFactory.getLogger(FaceNetFaceIDTrainer.class);

    private static int nEpochs = 5;
    private static int batchSize = 48; // depending on your hardware, you will want to increase or decrease
    private static int numExamples = LFWLoader.NUM_IMAGES;
    private static double splitTrainTest = 1.0;

    private static int seed = 123;
    private static int[] inputShape = new int[] {3, 96, 96};
    private static int outputNum = LFWLoader.NUM_LABELS;
    private static IUpdater updater = new Adam(0.1D, 0.9D, 0.999D, 0.01D);
    private static Activation transferFunction = Activation.RELU;
    private static int embeddingSize=128;
    private static CacheMode cacheMode = CacheMode.NONE;
    private static WorkspaceMode workspaceMode = WorkspaceMode.NONE;
    private static ConvolutionLayer.AlgoMode algoMode = ConvolutionLayer.AlgoMode.NO_WORKSPACE;

    public static void main(String[] args) {

        ComputationGraph net = getFaceNetNN4Small2();
        System.out.println(net.summary());

        net.setListeners(new ScoreIterationListener(1));

        int[] inputWHC = new int[]{inputShape[2], inputShape[1], inputShape[0]};
        LFWDataSetIterator iter = new LFWDataSetIterator(batchSize, numExamples, inputWHC, outputNum, false, true, splitTrainTest, new Random(seed));

        for (int i = 0; i < nEpochs; i++) {
            iter.reset();
            while (iter.hasNext()) {
                net.fit(iter);
            }
            Evaluation eval = net.evaluate(iter);
            System.out.println("Accuracy: " + eval.accuracy() + " | Precision: " +eval.precision() + " | Recall: " + eval.recall());
            log.info("*** Completed epoch {} ***", i);
        }

        ComputationGraph snipped = new TransferLearning.GraphBuilder(net)
                .setFeatureExtractor("embeddings") // the L2Normalize vertex and layers below are frozen
                .removeVertexAndConnections("lossLayer")
                .setOutputs("embeddings")
                .build();

        // grab a single example to test feed forward
        DataSet ds = iter.next();
        // when you forward a batch of examples ("faces") through the graph, you'll get a compressed representation as a result
        Map<String, INDArray> embedding = snipped.feedForward(ds.getFeatures(), false);
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
