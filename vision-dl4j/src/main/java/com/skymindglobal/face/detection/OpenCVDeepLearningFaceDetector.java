package com.skymindglobal.face.detection;

import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_dnn;
import org.bytedeco.javacv.Java2DFrameUtils;
import org.nd4j.linalg.io.ClassPathResource;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.opencv_core.CV_32F;
import static org.bytedeco.javacpp.opencv_dnn.blobFromImage;
import static org.bytedeco.javacpp.opencv_dnn.readNetFromCaffe;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

public class OpenCVDeepLearningFaceDetector extends FaceDetector {

    private opencv_dnn.Net model;

    public OpenCVDeepLearningFaceDetector(int imageWidth, int imageHeight, double detectionThreshold) {
        this.setImageHeight(imageHeight);
        this.setImageWidth(imageWidth);
        this.setDetectionThreshold(detectionThreshold);
        setModel();
    }


    private void setModel() {
        String PROTO_FILE = null;
        String CAFFE_MODEL_FILE = null;
        try {
            PROTO_FILE = new ClassPathResource("OpenCVDeepLearningFaceDetector/deploy.prototxt").getFile().getAbsolutePath();
            CAFFE_MODEL_FILE = new ClassPathResource("OpenCVDeepLearningFaceDetector/res10_300x300_ssd_iter_140000.caffemodel").getFile().getAbsolutePath();
        } catch (IOException e) {
            e.printStackTrace();
        }
        this.model = readNetFromCaffe(PROTO_FILE, CAFFE_MODEL_FILE);
    }

    @Override
    public List<FaceLocalization> detectFaces(opencv_core.Mat image) {
        int ori_height = image.size().height();
        int ori_width = image.size().width();

        // resize the image to match the input size of the model
        resize(image, image, new opencv_core.Size(this.getImage_width(), this.getImage_height()));

        // create a 4-dimensional blob from image with NCHW (Number of images in the batch -for training only-, Channel, Height, Width) dimensions order,
        // for more detailes read the official docs at https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#gabd0e76da3c6ad15c08b01ef21ad55dd8
        opencv_core.Mat blob = blobFromImage(image, 1.0,
                new opencv_core.Size(this.getImage_width(), this.getImage_height()),
                new opencv_core.Scalar(104.0, 177.0, 123.0, 0), false, false, CV_32F);

        // set the input to network model
        model.setInput(blob);
        // feed forward the input to the netwrok to get the output matrix
        opencv_core.Mat output = model.forward();
        // extract a 2d matrix for 4d output matrix with form of (number of detections x 7)
        opencv_core.Mat ne = new opencv_core.Mat(new opencv_core.Size(output.size(3), output.size(2)), CV_32F, output.ptr(0, 0));
        // create indexer to access elements of the matric
        FloatIndexer srcIndexer = ne.createIndexer();

        List<FaceLocalization> faceLocalizations = new ArrayList();
        for (int i = 0; i < output.size(3); i++) {//iterate to extract elements
            float confidence = srcIndexer.get(i, 2);
            float f1 = srcIndexer.get(i, 3);
            float f2 = srcIndexer.get(i, 4);
            float f3 = srcIndexer.get(i, 5);
            float f4 = srcIndexer.get(i, 6);
            if (confidence > this.getDetection_threshold()) {
                //top left point's x
                float tx = f1 * ori_width;
                //top left point's y
                float ty = f2 * ori_height;
                //bottom right point's x
                float bx = f3 * ori_width;
                //bottom right point's y
                float by = f4 * ori_height;
                faceLocalizations.add(new FaceLocalization(tx, ty, bx, by));
            }
        }
        return faceLocalizations;
    }
}
