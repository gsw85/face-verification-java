package com.skymindglobal.face.identification.training.vgg16.dataHelpers;

import com.skymindglobal.face.detection.FaceLocalization;
import com.skymindglobal.face.detection.OpenCVDeepLearningFaceDetector;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.Java2DFrameUtils;
import org.opencv.imgcodecs.Imgcodecs;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.util.List;

import static org.bytedeco.javacpp.opencv_imgcodecs.cvSaveImage;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

public class VGG16DatasetPreperation {

    private static String imageSource = "D:\\Public_Data\\face_recog_lfw50\\lfw50";
    private static String imageFace = "D:\\Public_Data\\face_recog_lfw50\\lfw50_faces";

    public static void main(String[] args) throws IOException {
        /**
         * Random sample 50 classes from lfw: D:\Public_Data\lfw\lfw => D:\Public_Data\face_recog_lfw50\lfw50
         * Detect and crop faces
         * Resize image to 244: vgg16 input
         *
         **/
        cropFaces();
    }

    private static void cropFaces() throws IOException {
        File imageSourceDir = new File(imageSource);
        listFilesForFolder(imageSourceDir);
    }

    public static void listFilesForFolder(final File folder) throws IOException {
        for (final File fileEntry : folder.listFiles()) {
            if (fileEntry.isDirectory()) {
                listFilesForFolder(fileEntry);
            } else {
                String target = imageFace + "\\" + folder.getName() + '\\' + fileEntry.getName();
                detectFacesAndSave(fileEntry.getAbsolutePath(), target);
            }
        }
    }

    public static void detectFacesAndSave(String source, String target) throws IOException {
        System.out.println(source + "-->" + target);
        OpenCVDeepLearningFaceDetector _OpenCVDeepLearningFaceDetector = new OpenCVDeepLearningFaceDetector(300, 300, 0.6);
        opencv_core.Mat image = imread(source);

        resize(image, image, new opencv_core.Size(300, 300));
        List<FaceLocalization> faceLocalizations = _OpenCVDeepLearningFaceDetector.detectFaces(image);

        for (FaceLocalization i : faceLocalizations) {
            int X = (int) i.getLeft_x();
            int Y = (int) i.getLeft_y();
            int Width = (int) i.getValidWidth(300);
            int Height = (int) i.getValidHeight(300);

            opencv_core.Mat crop_image = new opencv_core.Mat(image, new opencv_core.Rect(X, Y, Width, Height));
            resize(crop_image, crop_image, new opencv_core.Size(244, 244));
            File targetFile = new File(target);
            targetFile.getParentFile().mkdirs();
            ImageIO.write(Java2DFrameUtils.toBufferedImage(crop_image), "jpg", targetFile);

        };
    }
}
