package com.skymindglobal.face;

import com.skymindglobal.face.detection.OpenIMAJ_FKEFaceDetector;
import com.skymindglobal.face.detection.FaceDetector;
import com.skymindglobal.face.detection.FaceLocalization;
import com.skymindglobal.face.detection.OpenCV_DeepLearningFaceDetector;
import com.skymindglobal.face.identification.*;
import com.skymindglobal.face.identification.feature.VGG16FeatureProvider;
import com.skymindglobal.face.pose.*;
import com.skymindglobal.face.pose.HeadPoseEstimator;
import com.skymindglobal.face.pose.KerasModel_HeadPoseEstimator;
import com.skymindglobal.face.pose.OpenCV_HeadPoseEstimator;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;

import static com.skymindglobal.face.pose.HeadPoseEstimator.KERAS_MODEL;
import static com.skymindglobal.face.pose.HeadPoseEstimator.OPENCV_HEAD_POSE_ESTIMATOR;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.javacpp.opencv_videoio.CAP_PROP_FRAME_WIDTH;
import org.bytedeco.javacpp.opencv_videoio.VideoCapture;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.io.ClassPathResource;
import org.openimaj.image.processing.face.detection.keypoints.FacialKeypoint;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class FaceID {
    private static final OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
    private static final int WIDTH = 1920;
    private static final int HEIGHT = 1080;

    public static void main(String[] args) throws IOException, ClassNotFoundException, CanvasFrame.Exception, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        FaceDetector FaceDetector = getFaceDetector(com.skymindglobal.face.detection.FaceDetector.OPENCV_DL_FACEDETECTOR);
        FaceIdentifier FaceIdentifier = getFaceIdentifier(com.skymindglobal.face.identification.FaceIdentifier.FEATURE_DISTANCE);
        HeadPoseEstimator HeadPoseEstimator = getHeadPoseEstimator(OPENCV_HEAD_POSE_ESTIMATOR);

        VideoCapture capture = new VideoCapture();
        capture.set(CAP_PROP_FRAME_WIDTH, WIDTH);
        capture.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);

        if (!capture.open(0)) {
            System.out.println("Can not open the cam !!!");
        }

        Mat image = new Mat();
        CanvasFrame mainframe = new CanvasFrame(
                "FaceLocalization Identification",
                0,
                null,
                CanvasFrame.getDefaultGamma() / 2.2
        );

        mainframe.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        mainframe.setCanvasSize(WIDTH, HEIGHT);
        mainframe.setLocationRelativeTo(null);
        mainframe.setVisible(true);

        while (true) {
            while (capture.read(image) && mainframe.isVisible()) {

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

//                // head pose estimation
//                image.copyTo(cloneCopy);
//                HeadPoseEstimator.estimate(null, cloneCopy);

                mainframe.showImage(converter.convert(image));

                try {
                    Thread.sleep(50); //50
                } catch (InterruptedException ex) {
                    System.out.println(ex.getMessage());
                }
            }
        }
    }

    private static HeadPoseEstimator getHeadPoseEstimator(String HeadPoseEstimator) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        switch(HeadPoseEstimator){
            case OPENCV_HEAD_POSE_ESTIMATOR:
                // not stable
                return new OpenCV_HeadPoseEstimator();
            case KERAS_MODEL:
                // not stable
                return new KerasModel_HeadPoseEstimator();
            default:
                return null;
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
                return new VGG16FaceIdentifier(3);
            case FaceIdentifier.FEATURE_DISTANCE:
                File dictionary = new ClassPathResource("Office-Faces").getFile();
                return new DistanceFaceIdentifier(
                        new VGG16FeatureProvider(), dictionary, 1, 0.78, 3, 3);
            case FaceIdentifier.ZHZD:
                return new AlexNetFaceIdentifier(5);
            default:
                return null;
        }
    }

    private static FaceDetector getFaceDetector(String faceDetector) {
        switch (faceDetector){
            case FaceDetector.OPENCV_DL_FACEDETECTOR:
                return new OpenCV_DeepLearningFaceDetector(300, 300, 0.8);
            case FaceDetector.OPENIMAJ_FKE_FACEDETECTOR:
                return new OpenIMAJ_FKEFaceDetector( 1.0);
            default:
                return null;
        }
    }

    private static void annotateFaces(List<FaceLocalization> faceLocalizations, Mat image) {
        for (FaceLocalization i : faceLocalizations){
            rectangle(image,new opencv_core.Rect(new opencv_core.Point((int) i.getLeft_x(),(int) i.getLeft_y()), new opencv_core.Point((int) i.getRight_x(),(int) i.getRight_y())), new opencv_core.Scalar(255, 0, 0, 0));

            if(i.getKeyPoints() != null){
                for (FacialKeypoint x : i.getKeyPoints()){
                    circle(
                            image,
                            new Point(
                                    (int)(x.position.x+i.getLeft_x()),
                                    (int)(x.position.y+i.getLeft_y())
                            ),
                            2,
                            new opencv_core.Scalar(255, 0, 0, 0),
                            1,
                            8,
                            0
                    );
                }
            }
        };
    }
}