package com.skymindglobal.faceverification;

import com.skymindglobal.faceverification.detection.FaceDetector;
import com.skymindglobal.faceverification.detection.FaceLocalization;
import com.skymindglobal.faceverification.detection.OpenCV_DeepLearningFaceDetector;
import com.skymindglobal.faceverification.detection.OpenIMAJ_FKEFaceDetector;
import com.skymindglobal.faceverification.identification.*;
import com.skymindglobal.faceverification.identification.feature.VGG16FeatureProvider;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.List;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

public class FaceIDVideo {
    private static final OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
    private static final int WIDTH = 480;//1920;
    private static final int HEIGHT = 360;//1080;

    public static void main(String[] args) throws IOException, ClassNotFoundException, CanvasFrame.Exception {
        FaceDetector FaceDetector = getFaceDetector(com.skymindglobal.faceverification.detection.FaceDetector.OPENCV_DL_FACEDETECTOR);
        FaceIdentifier FaceIdentifier = getFaceIdentifier(com.skymindglobal.faceverification.identification.FaceIdentifier.ZHZD);

        String videoPath = "C:\\Users\\PK Chuah\\Videos\\Captures\\sample.mp4";
        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(videoPath);
        grabber.setFormat("mp4");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        grabber.start();

        CanvasFrame mainframe = new CanvasFrame(
                "FaceLocalization Identification",
//                0,
//                null,
                CanvasFrame.getDefaultGamma() / 2.2
        );

        mainframe.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        mainframe.setCanvasSize(WIDTH, HEIGHT);
        mainframe.setLocationRelativeTo(null);
        mainframe.setVisible(true);

        while (true) {
            while (mainframe.isVisible()) {
                Frame frame = grabber.grabImage();
                opencv_core.Mat image = converter.convert(frame);

                Mat cloneCopy = new Mat();

                // face detection
                image.copyTo(cloneCopy);
                FaceDetector.detectFaces(cloneCopy);
                List<FaceLocalization> faceLocalizations = FaceDetector.getFaceLocalization();
                annotateFaces(faceLocalizations, image);

                // face identification
                image.copyTo(cloneCopy);
                List<List<Prediction>> faceIdentities = FaceIdentifier.identify(faceLocalizations, cloneCopy);
                labelIndividual(faceIdentities, image);

                mainframe.showImage(converter.convert(image));

                try {
                    Thread.sleep(50);
                } catch (InterruptedException ex) {
                    System.out.println(ex.getMessage());
                }
            }
        }
    }

    private static void labelIndividual(List<List<Prediction>> faceIdentities, Mat image) {
        for (List<Prediction> i: faceIdentities){
            for(int j=0; j<i.size(); j++)
            {
                putText(
                        image,
                        i.get(j).toString(),
                        new Point(
                                (int)i.get(j).getFaceLocalization().getLeft_x(),
                                (int)i.get(j).getFaceLocalization().getLeft_y() + j*13
                        ),
                        FONT_HERSHEY_PLAIN,
                        0.7,
                        Scalar.YELLOW
                );
            }
        }
    }

    private static FaceIdentifier getFaceIdentifier(String faceIdentifier) throws IOException, ClassNotFoundException {
        switch (faceIdentifier){
            case FaceIdentifier.CUSTOM_VGG16:
                return new VGG16FaceIdentifier(3);
            case FaceIdentifier.FEATURE_DISTANCE_VGG16:
                File dictionary = new ClassPathResource("vgg16_faces_224").getFile();
                return new DistanceFaceIdentifier(new VGG16FeatureProvider(), dictionary,3, 0.8, 3, 3);
            case FaceIdentifier.ZHZD:
                return new AlexNetFaceIdentifier(5);
            default:
                return null;
        }
    }

    private static FaceDetector getFaceDetector(String faceDetector) {
        switch (faceDetector){
            case FaceDetector.OPENCV_DL_FACEDETECTOR:
                return new OpenCV_DeepLearningFaceDetector(300, 300, 0.6);
            case FaceDetector.OPENIMAJ_FKE_FACEDETECTOR:
                return new OpenIMAJ_FKEFaceDetector( 0.6);
            default:
                return null;
        }
    }

    private static void annotateFaces(List<FaceLocalization> faceLocalizations, Mat image) {
        for (FaceLocalization i : faceLocalizations){
            rectangle(image,new Rect(new Point((int) i.getLeft_x(),(int) i.getLeft_y()), new Point((int) i.getRight_x(),(int) i.getRight_y())), new Scalar(255, 0, 0, 0));
        };
    }
}