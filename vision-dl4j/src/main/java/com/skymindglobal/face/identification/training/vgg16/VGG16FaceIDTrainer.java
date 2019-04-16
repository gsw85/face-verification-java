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

public class VGG16FaceIDTrainer {
    private static final Logger log = LoggerFactory.getLogger(VGG16FaceIDTrainer.class);
    // parameters for the training phase
    private static int trainBatchSize = 64;
    private static int nEpochs = 40;
    private static double learningRate = 1e-3;
    private static int nClasses = 0;
    private static List<String> labels;
    private static int seed = 123;
    private static int saveModelEpochInterval = 1;
    private static boolean TRAINING_MODE = true;
    private static String unique_id = "vgg16_faceid_v14";
    private static String modelFilename = new File(".").getAbsolutePath() + "/generated-models/" + unique_id + ".zip";
    private static String labelFilename = new File(".").getAbsolutePath() + "/generated-models/" + unique_id + ".lbl";
    private static String trainingUIStoragePath = new File(".").getAbsolutePath() + "/.trainingUI/" + unique_id;

    public static void main(String[] args) throws Exception {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);

        ComputationGraph model;
        // Directory for Custom train and test datasets
        log.info("Load data...");
        VGG16DatasetIterator _VGG16DatasetIterator = new VGG16DatasetIterator(
                new File("D:\\Public_Data\\face_recog\\lfw_custom_train_cropped"),
                new File("D:\\Public_Data\\face_recog\\lfw_custom_test_cropped"),
                trainBatchSize,
                1 // get all samples
        );
        nClasses = _VGG16DatasetIterator.getNumClass();
        RecordReaderDataSetIterator trainIter = _VGG16DatasetIterator.trainIterator();
        trainIter.setPreProcessor( new VGG16ImagePreProcessor());

        RecordReaderDataSetIterator testIter = _VGG16DatasetIterator.testIterator();
        testIter.setPreProcessor( new VGG16ImagePreProcessor());

        // Print Labels
        labels = trainIter.getLabels();
        LabelManager.exportLabels(labelFilename, labels);
        System.out.println(Arrays.toString(labels.toArray()));

        if (new File(modelFilename).exists() && TRAINING_MODE) {
            log.info("Loading model from "+ modelFilename);
            model = ModelSerializer.restoreComputationGraph(modelFilename, true);
            log.info("Continue Training...");
            trainModel(trainIter,testIter, model);
        }
        else if(new File(modelFilename).exists()){
            log.info("Load model...");
            model = ModelSerializer.restoreComputationGraph(modelFilename);
        }
        else{
            log.info("Build model...");
            ComputationGraph pretrained = (ComputationGraph) VGG16.builder().build().initPretrained(PretrainedType.VGGFACE);
            log.info(pretrained.summary());
            // Transfer Learning steps - Model Configurations.
            FineTuneConfiguration fineTuneConf = getFineTuneConfiguration();
            // Transfer Learning steps - Modify prebuilt model's architecture for current scenario
            model = buildComputationGraph(pretrained, fineTuneConf);
            trainModel(trainIter,testIter, model);
        }
    }

    private static void trainModel(RecordReaderDataSetIterator trainIter,RecordReaderDataSetIterator testIter, ComputationGraph model) throws IOException {
        log.info("Train model...");
        UIServer server = UIServer.getInstance();
        StatsStorage storage = new FileStatsStorage(
                new File(trainingUIStoragePath)
        );

        server.attach(storage);
        model.setListeners(
                new ScoreIterationListener(1),
                new StatsListener(storage, 1)
        );

        for (int i = 0; i < nEpochs; i++) {
            trainIter.reset();
            while (trainIter.hasNext()) {
                model.fit(trainIter);
            }
            log.info("*** Completed epoch {} ***", i);
            if((i+1)%saveModelEpochInterval==0){
                ModelSerializer.writeModel(model, modelFilename, true);
                log.info("Checkpoint: Model saved!");
            }
            log.info(model.evaluate(trainIter).stats(true, false) + "\n");
            log.info(model.evaluate(testIter).stats(true, false) + "\n");

        }
    }

    private static ComputationGraph buildComputationGraph(ComputationGraph pretrained, FineTuneConfiguration fineTuneConf) {
        //Construct a new model with the intended architecture and print summary
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(2622).nOut(nClasses)
                                .weightInit(WeightInit.DISTRIBUTION)
                                //This weight init dist gave better results than Xavier
                                .dist(new NormalDistribution(0,0.2*(2.0/(4096+nClasses))))
                                .activation(Activation.SOFTMAX).build(),
                        "fc8")
                .setOutputs("predictions")
                .build();

        log.info(vgg16Transfer.summary());
        return vgg16Transfer;
    }

    private static FineTuneConfiguration getFineTuneConfiguration() {

        FineTuneConfiguration _FineTuneConfiguration = new FineTuneConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Nesterovs.Builder().learningRate(learningRate).momentum(Nesterovs.DEFAULT_NESTEROV_MOMENTUM).build())
                .l2(0.001)
                .activation(Activation.IDENTITY)
                .build();

        return _FineTuneConfiguration;
    }
}
