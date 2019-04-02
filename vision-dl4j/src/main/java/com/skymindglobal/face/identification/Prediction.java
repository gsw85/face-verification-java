package com.skymindglobal.face.identification;

import com.skymindglobal.face.detection.FaceLocalization;

public class Prediction {

    private String label;
    private double percentage;
    private FaceLocalization faceLocalization;

    public Prediction(String label, double percentage, FaceLocalization faceLocalization) {
        this.label = label;
        this.percentage = percentage;
        this.faceLocalization = faceLocalization;
    }

    public FaceLocalization getFaceLocalization(){
        return this.faceLocalization;
    }

    public String toString() {
        return String.format("%s: %.2f ", this.label, this.percentage);
    }
}