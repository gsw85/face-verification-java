package com.skymindglobal.face.identification;

import com.skymindglobal.face.detection.FaceLocalization;
import org.bytedeco.javacpp.opencv_core;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.opencv_dnn.blobFromImage;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

public class CustomVGG16FaceIdentifier extends FaceIdentifier {
    private static final Logger log = LoggerFactory.getLogger(CustomVGG16FaceIdentifier.class);

    // labels - model's classes
    private static String[] labels = new String[]{"Individual A", "Individual B", "Individual C"};
    private ComputationGraph model = null;

    public CustomVGG16FaceIdentifier() {
        String modelFilename = null;
        try {
            modelFilename = new ClassPathResource("vgg16/VGG16_TLDetectorActors.zip").getFile().getAbsolutePath();
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (new File(modelFilename).exists()) {
            log.info("Load model...");
            try {
                model = ModelSerializer.restoreComputationGraph(modelFilename);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            log.info("Model not found.");
        }
    }

    @Override
    public List<List<Prediction>> identify(List<FaceLocalization> faceLocalizations, opencv_core.Mat image) throws IOException {

        NativeImageLoader nativeImageLoader = new NativeImageLoader();
        List<List<Prediction>> collection = new ArrayList<>();
        for (int i = 0; i<faceLocalizations.size(); i++){

            int X = (int) faceLocalizations.get(i).getLeft_x();
            int Y = (int) faceLocalizations.get(i).getLeft_y();
//            int Width = (int) faceLocalizations.get(i).getWidth(image.size().width()) -1;
//            int Height = (int) faceLocalizations.get(i).getHeight(image.size().height()) -1;
            int Width = (int) (faceLocalizations.get(i).getRight_x() - faceLocalizations.get(i).getLeft_x());
            int Height = (int) (faceLocalizations.get(i).getRight_y() - faceLocalizations.get(i).getLeft_y());

            if (X < 0){
                X = 0;
            }

            if (Y < 0){
                Y = 0;
            }

            if (( Y + Height)> image.size().height()){
                Height = image.size().height() - Y;
            }
            if (( X + Width)> image.size().width()){
                Width = image.size().width() - X;
            }

            log.info(String.valueOf(X));
            log.info(String.valueOf(Y));
            log.info(String.valueOf(Width));
            log.info(String.valueOf(Height));
            opencv_core.Mat crop_image = new opencv_core.Mat(image, new opencv_core.Rect(X, Y, Width, Height));
            resize(crop_image, crop_image, new opencv_core.Size(244, 244));
            INDArray _image = nativeImageLoader.asMatrix(crop_image);
            VGG16ImagePreProcessor _VGG16ImagePreProcessor = new VGG16ImagePreProcessor();
            _VGG16ImagePreProcessor.transform(_image);
            INDArray[] output = model.output(false, _image);
            List<Prediction> predictions = decodePredictions(output[0], 1, faceLocalizations.get(i));
            collection.add(predictions);
        }
        return collection;
    }

    private List<Prediction> decodePredictions(INDArray encodedPredictions, int numPredicted, FaceLocalization faceLocalization) {
        List<Prediction> decodedPredictions = new ArrayList<>();
        int[] topX = new int[numPredicted];
        float[] topXProb = new float[numPredicted];

        int i = 0;
        for (INDArray currentBatch = encodedPredictions.getRow(0).dup(); i < numPredicted; ++i) {

            topX[i] = Nd4j.argMax(currentBatch, 1).getInt(0, 0);
            topXProb[i] = currentBatch.getFloat(0, topX[i]);
            currentBatch.putScalar(0, topX[i], 0.0D);
            decodedPredictions.add(
                    new Prediction(
                            labels[topX[i]],
                            (topXProb[i] * 100.0F),
                            faceLocalization
                    )
            );
        }
        return decodedPredictions;
    }
}