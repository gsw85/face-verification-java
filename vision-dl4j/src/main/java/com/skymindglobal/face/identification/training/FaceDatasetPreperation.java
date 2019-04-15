package com.skymindglobal.face.identification.training;

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
import static org.bytedeco.javacpp.opencv_imgproc.logPolar;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

public class FaceDatasetPreperation {
    private static String lfwSource = "D:\\Public_Data\\lfw\\lfw";

    private static String imageSourceTrain = "D:\\Public_Data\\face_recog\\lfw_custom_train";
    private static String imageSourceTest = "D:\\Public_Data\\face_recog\\lfw_custom_test";
    private static String imageSourceTrainCropped = "D:\\Public_Data\\face_recog\\lfw_train_96";
    private static String imageSourceTestCropped = "D:\\Public_Data\\face_recog\\lfw_test_96";

    private static int trainPerc = 50;
    private static int numClass = 50;
    private static int minSamples = 20;
    private static int maxSamples = 30;
    private static int OUTPUT_IMAGE_WIDTH = 96;
    private static int OUTPUT_IMAGE_HEIGHT = 96;
    private static int OPENCV_DL_FACEDETECTOR_WIDTH = 300;
    private static int OPENCV_DL_FACEDETECTOR_HEIGHT = 300;
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FaceDatasetPreperation.class);

    public static void main(String[] args) throws IOException {
        /**
         * Random sample 50 classes from lfw: D:\Public_Data\lfw\lfw => D:\Public_Data\face_recog_lfw50\lfw50
         * Detect and crop faces
         * Resize image to 244: vgg16 input
         *
         **/
//        dataSampling(minSamples, maxSamples);
        processFaces(imageSourceTrain, imageSourceTrainCropped);
        processFaces(imageSourceTest, imageSourceTestCropped);

    }

    private static void dataSampling(int minSamples, int maxSamples) {
        File lfwSourceDir = new File(lfwSource);
        int i=0;
        for (final File fileEntry : lfwSourceDir.listFiles()) {
            if (fileEntry.isDirectory()){
                if(fileEntry.listFiles().length>=minSamples && fileEntry.listFiles().length<=maxSamples) {
                    try {
                        randomAssignImages(fileEntry);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    i++;
                }
            }
            if(i>=numClass){
                break;
            }
        }
    }

    private static void randomAssignImages(File fileEntry) throws IOException {
        for (File i: fileEntry.listFiles()){
            Random rand = new Random();
            int n = rand.nextInt(100);
            if (n > trainPerc) {
                FileUtils.copyFile(i, new File(imageSourceTest + "\\" + fileEntry.getName() + "\\" + i.getName()));
            } else {
                FileUtils.copyFile(i, new File(imageSourceTrain + "\\" + fileEntry.getName() + "\\" + i.getName()));
            }
        }
    }

    private static void processFaces(String source, String destination) throws IOException {
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

        // detect faces
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
