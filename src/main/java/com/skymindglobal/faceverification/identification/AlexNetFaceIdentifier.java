package com.skymindglobal.faceverification.identification;

import com.skymindglobal.faceverification.detection.FaceLocalization;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public class AlexNetFaceIdentifier extends FaceIdentifier {
    private static final Logger log = LoggerFactory.getLogger(AlexNetFaceIdentifier.class);

    // labels - model's classes
    private static List<String> labels;
    private MultiLayerNetwork model = null;
    private static String modelFilename;
    public static final int WIDTH = 100;
    public static final int HEIGHT = 100;
    private int numPrediction;
    private ImagePreProcessingScaler _DataNormalization;
    private int channels = 3;
    private int numLabels;
    private int seed= 42;
    private Random rng= new Random(seed);
    private int maxPathsPerLabel= 18;

    public AlexNetFaceIdentifier(int numPrediction) throws IOException, ClassNotFoundException {
        this.numPrediction = numPrediction;
        modelFilename = new ClassPathResource("ZHZD/face.bin").getFile().getAbsolutePath();
        loadCNTrainingEnvironmentConfiguration();
        if (new File(modelFilename).exists()) {
            log.info("Load model...");
            try {
                model = ModelSerializer.restoreMultiLayerNetwork(modelFilename);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            log.info("Model not found.");
        }
    }

    private void loadCNTrainingEnvironmentConfiguration() throws IOException {
        ParentPathLabelGenerator labelMaker= new ParentPathLabelGenerator();
        File _file = new ClassPathResource("ZHZD/fz/train").getFile();
        InputSplit trainData= getData(_file.getAbsolutePath(),labelMaker);
        ImageRecordReader recordReader = new ImageRecordReader(HEIGHT, WIDTH, channels, labelMaker);
        recordReader.initialize(trainData, null);
        int batchSize= 115; // load all train for normalization stats
        RecordReaderDataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);

        labels = dataIter.getLabels();

        _DataNormalization = new ImagePreProcessingScaler(0, 1);
        _DataNormalization.fit(dataIter);
    }

    private InputSplit getData(String path, ParentPathLabelGenerator labelMaker){
        File mainPath= new File(path);
        FileSplit fileSplit= new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        int numExamples= Math.toIntExact(fileSplit.length());
        numLabels = fileSplit.getRootDir().listFiles().length;
        BalancedPathFilter pathFilter= new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, maxPathsPerLabel);
        InputSplit[] inputSplit= fileSplit.sample(pathFilter, 1);
        return inputSplit[0];
    }

    @Override
    public List<List<Prediction>> identify(List<FaceLocalization> faceLocalizations, Mat image) throws IOException {

        NativeImageLoader nativeImageLoader = new NativeImageLoader();
        List<List<Prediction>> collection = new ArrayList<>();
        for (int i = 0; i<faceLocalizations.size(); i++){

            int X = (int) faceLocalizations.get(i).getLeft_x();
            int Y = (int) faceLocalizations.get(i).getLeft_y();
            int Width = faceLocalizations.get(i).getValidWidth(image.size().width());
            int Height = faceLocalizations.get(i).getValidHeight(image.size().height());

            // Crop face, Resize and convert into INDArr
            Mat crop_image = new Mat(image, new Rect(X, Y, Width, Height));
            resize(crop_image, crop_image, new Size(HEIGHT, WIDTH));
            INDArray _image = nativeImageLoader.asMatrix(crop_image);

            _DataNormalization.transform(_image);

            // predicts
            INDArray output = model.output(_image);
            List<Prediction> predictions = decodePredictions(output, faceLocalizations.get(i));
            collection.add(predictions);
        }
        return collection;
    }

    private List<Prediction> decodePredictions(INDArray encodedPredictions, FaceLocalization faceLocalization) {
        List<Prediction> decodedPredictions = new ArrayList<>();
        int[] topX = new int[this.numPrediction];
        float[] topXProb = new float[this.numPrediction];

        log.info(String.valueOf(encodedPredictions));
        int i = 0;
        for (INDArray currentBatch = encodedPredictions.getRow(0).dup(); i < encodedPredictions.size(0); ++i) {

            topX[i] = Nd4j.argMax(currentBatch, 1).getInt(0, 0);
            topXProb[i] = currentBatch.getFloat(0, topX[i]);
            currentBatch.putScalar(0, topX[i], 0.0D);
            decodedPredictions.add(new Prediction(labels.get(topX[i]),(topXProb[i] * 100.0F),faceLocalization));
        }

        log.info(String.valueOf(decodedPredictions.size()));
        return decodedPredictions;
    }
}