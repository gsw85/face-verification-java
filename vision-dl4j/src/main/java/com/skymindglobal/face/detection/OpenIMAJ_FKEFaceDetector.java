package com.skymindglobal.face.detection;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.Java2DFrameUtils;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.face.alignment.AffineAligner;
import org.openimaj.image.processing.face.alignment.FaceAligner;
import org.openimaj.image.processing.face.detection.keypoints.FKEFaceDetector;
import org.openimaj.image.processing.face.detection.keypoints.FacialKeypoint;
import org.openimaj.image.processing.face.detection.keypoints.KEDetectedFace;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

public class OpenIMAJ_FKEFaceDetector extends FaceDetector {
    private List<KEDetectedFace> detectedFaces;
    private BufferedImage frame;

    public OpenIMAJ_FKEFaceDetector(double detectionThreshold) {
        this.setDetectionThreshold(detectionThreshold);
    }

    @Override
    public void detectFaces(opencv_core.Mat image) {
        frame = Java2DFrameUtils.toBufferedImage(image);
        FImage img = ImageUtilities.createFImage(frame);
        FKEFaceDetector detector = new FKEFaceDetector();
        detectedFaces = detector.detectFaces(img);
    }

    @Override
    public List<FaceLocalization> getFaceLocalization() {
        List<FaceLocalization> faceLocalizations = new ArrayList();
        for (KEDetectedFace i: detectedFaces){
            if (i.getConfidence() > this.getDetection_threshold()) {
                //top left point's x
                float tx = i.getBounds().x;
                //top left point's y
                float ty = i.getBounds().y;
                //bottom right point's x
                float bx = i.getBounds().x + i.getBounds().width;
                //bottom right point's y
                float by = i.getBounds().y + i.getBounds().height;
                faceLocalizations.add(new FaceLocalization(tx, ty, bx, by));
            }
        }
        return faceLocalizations;
    }

    public List<BufferedImage> getAlignedFacePatches(){
        FaceAligner<KEDetectedFace> aligner = new AffineAligner();

        List<BufferedImage> patches = new ArrayList<>();
        for(KEDetectedFace i : detectedFaces){
            FImage alignedFace = aligner.align(i);
            patches.add(
                    ImageUtilities.createBufferedImage(
                        alignedFace,
                        frame.getSubimage(
                                (int)i.getBounds().x,
                                (int)i.getBounds().y,
                                (int)i.getBounds().width,
                                (int)i.getBounds().height
                            )
                    )
            );
        }
        return patches;
    }

}
