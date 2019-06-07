package com.skymindglobal.faceverification.pose;

import com.skymindglobal.faceverification.detection.FaceLocalization;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;

import java.io.IOException;
import java.util.List;

public class OpenCV_HeadPoseEstimator extends HeadPoseEstimator {
//    private final CascadeClassifier eyeCascade;
//    private final CascadeClassifier faceCascade;
//    private final CascadeClassifier profileCascade;
    OpenCVFrameConverter.ToMat toJavacv_convertor = new OpenCVFrameConverter.ToMat();
    OpenCVFrameConverter.ToOrgOpenCvCoreMat toOpencv_convertor = new OpenCVFrameConverter.ToOrgOpenCvCoreMat();

    public OpenCV_HeadPoseEstimator() throws IOException {
//        faceCascade = new CascadeClassifier(
//                new ClassPathResource("haarcascades/haarcascade_frontalface_alt.xml").getFile().getAbsolutePath());
//        eyeCascade = new CascadeClassifier(
//                new ClassPathResource("haarcascades/haarcascade_eye.xml").getFile().getAbsolutePath());
//        profileCascade = new CascadeClassifier(
//                new ClassPathResource("haarcascades/haarcascade_profileface.xml").getFile().getAbsolutePath());
    }

    public void estimate(List<FaceLocalization> faceLocalizations, Mat image) {
//        Rect faceRect = new Rect();
//        Rect eyeRect = new Rect();
//        Rect faceProfile = new Rect();
//
//        Mat mat = new Mat(image.address());
//
//        faceCascade.detectMultiScale(mat,faceRect);
//        eyeCascade.detectMultiScale(mat,eyeRect);
//        profileCascade.detectMultiScale(mat,faceProfile);

//        HPE(mat);
    }

//    public void HPE(Mat matOfImg){
//
//        int MaxCount= 10;
//        TermCriteria termCriteria = new TermCriteria(TermCriteria.EPS | TermCriteria.COUNT,20,0.3);
//
//        List<MatOfPoint2f> matOfPointList = new ArrayList<MatOfPoint2f>();
//        Size subPixWinSize = new Size(10,10);
//        Size winSize = new Size(21,21);
//
//        Imgproc.goodFeaturesToTrack(matOfImg, (MatOfPoint) matOfPointList,MaxCount,0.01,10, null,3,true,0.04);
//        Imgproc.cornerSubPix(matOfImg, (MatOfPoint2f) matOfPointList,subPixWinSize,new Size(-1,-1),termCriteria);
//
//    }

}
