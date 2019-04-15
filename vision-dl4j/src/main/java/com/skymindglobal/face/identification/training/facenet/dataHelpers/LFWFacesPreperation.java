package com.skymindglobal.face.identification.training.facenet.dataHelpers;

import com.skymindglobal.face.detection.FaceLocalization;
import com.skymindglobal.face.detection.OpenCVDeepLearningFaceDetector;
import org.apache.commons.io.FileUtils;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.Java2DFrameUtils;
import org.slf4j.Logger;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

public class LFWFacesPreperation {
    private static String lfwSource = "D:\\Public_Data\\lfw\\lfw";
    private static String imageSourceTrainCropped = "D:\\Public_Data\\lfw\\lfw_opencv_face";;

    private static int OUTPUT_IMAGE_WIDTH = 96;
    private static int OUTPUT_IMAGE_HEIGHT = 96;
    private static int OPENCV_DL_FACEDETECTOR_WIDTH = 300;
    private static int OPENCV_DL_FACEDETECTOR_HEIGHT = 300;
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(LFWFacesPreperation.class);

    public static void main(String[] args) throws IOException {
        cropFaces(lfwSource, imageSourceTrainCropped);
    }

    private static void cropFaces(String source, String destination) throws IOException {
        File imageSourceDir = new File(source);
        listFilesForFolder(imageSourceDir, destination);
    }

    public static void listFilesForFolder(final File folder, String imageSourceCropped) throws IOException {
        for (final File fileEntry : folder.listFiles()) {
            if (fileEntry.isDirectory()) {
                listFilesForFolder(fileEntry, imageSourceCropped);
            } else {
                String target = imageSourceCropped + "\\" + folder.getName() + '\\' + fileEntry.getName();
                detectFacesAndSave(fileEntry.getAbsolutePath(), target);
            }
        }
    }

    public static void detectFacesAndSave(String source, String target) throws IOException {
        OpenCVDeepLearningFaceDetector _OpenCVDeepLearningFaceDetector = new OpenCVDeepLearningFaceDetector(300, 300, 0.6);
        opencv_core.Mat image = imread(source);

        resize(image, image, new opencv_core.Size(OPENCV_DL_FACEDETECTOR_WIDTH, OPENCV_DL_FACEDETECTOR_HEIGHT));
        List<FaceLocalization> faceLocalizations = _OpenCVDeepLearningFaceDetector.detectFaces(image);

        for (FaceLocalization i : faceLocalizations) {
            int X = (int) i.getLeft_x();
            int Y = (int) i.getLeft_y();
            int Width = i.getValidWidth(OPENCV_DL_FACEDETECTOR_WIDTH);
            int Height = i.getValidHeight(OPENCV_DL_FACEDETECTOR_HEIGHT);

            log.info(X+", "+ Y+", "+Width+", "+Height);
            if(Width>0 && Height>0){
                opencv_core.Mat crop_image = new opencv_core.Mat(image, new opencv_core.Rect(X, Y, Width, Height));
                resize(crop_image, crop_image, new opencv_core.Size(OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT));
                File targetFile = new File(target);
                targetFile.getParentFile().mkdirs();
                ImageIO.write(Java2DFrameUtils.toBufferedImage(crop_image), "jpg", targetFile);
            }
            else
            {
                log.info("Skipping: "+source);
            }
        };
    }
}
