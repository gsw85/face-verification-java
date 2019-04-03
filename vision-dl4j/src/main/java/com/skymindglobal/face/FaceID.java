package com.skymindglobal.face;

import com.skymindglobal.face.detection.FaceDetector;
import com.skymindglobal.face.detection.FaceLocalization;
import com.skymindglobal.face.detection.OpenCVDeepLearningFaceDetector;
import com.skymindglobal.face.identification.CustomVGG16FaceIdentifier;
import com.skymindglobal.face.identification.FaceIdentifier;
import com.skymindglobal.face.identification.Prediction;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.javacpp.opencv_videoio.CAP_PROP_FRAME_WIDTH;
import org.bytedeco.javacpp.opencv_videoio.VideoCapture;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.IOException;
import java.util.List;

public class FaceID {
    private static final OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
    private static final int WIDTH = 480;//1920;
    private static final int HEIGHT = 360;//1080;

    public static void main(String[] args) throws IOException, ClassNotFoundException, CanvasFrame.Exception {
        FaceDetector FaceDetector = getFaceDetector(com.skymindglobal.face.detection.FaceDetector.OPENCV_DL_FACEDETECTOR);
        FaceIdentifier FaceIdentifier = getFaceIdentifier(com.skymindglobal.face.identification.FaceIdentifier.CUSTOM_VGG16);

        VideoCapture capture = new VideoCapture();
        capture.set(CAP_PROP_FRAME_WIDTH, WIDTH);
        capture.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);

        if (!capture.open(0)) {
            System.out.println("Can not open the cam !!!");
        }

        Mat image = new Mat();
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
            while (capture.read(image) && mainframe.isVisible()) {

                Mat cloneCopy = new Mat();

                image.copyTo(cloneCopy);
                List<FaceLocalization> faceLocalizations = FaceDetector.detectFaces(cloneCopy);
                annotateFaces(faceLocalizations, image);

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
                        new opencv_core.Point(
                                (int)i.get(j).getFaceLocalization().getLeft_x(),
                                (int)i.get(j).getFaceLocalization().getLeft_y() + j*13
                        ),
                        FONT_HERSHEY_PLAIN,
                        0.7,
                        opencv_core.Scalar.YELLOW
                );
            }
        }
    }

    private static FaceIdentifier getFaceIdentifier(String faceIdentifier) throws IOException, ClassNotFoundException {
        switch (faceIdentifier){
            case FaceIdentifier.CUSTOM_VGG16:
                return new CustomVGG16FaceIdentifier(3);
            default:
                return null;
        }
    }

    private static FaceDetector getFaceDetector(String faceDetector) {
        switch (faceDetector){
            case FaceDetector.OPENCV_DL_FACEDETECTOR:
                return new OpenCVDeepLearningFaceDetector(300, 300, 0.6);
            default:
                return null;
        }
    }

    private static void annotateFaces(List<FaceLocalization> faceLocalizations, Mat image) {
        for (FaceLocalization i : faceLocalizations){
            rectangle(image,new opencv_core.Rect(new opencv_core.Point((int) i.getLeft_x(),(int) i.getLeft_y()), new opencv_core.Point((int) i.getRight_x(),(int) i.getRight_y())), new opencv_core.Scalar(255, 0, 0, 0));
        };
    }
}