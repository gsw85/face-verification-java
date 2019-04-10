package com.skymindglobal.face.identification;

import org.nd4j.linalg.api.ndarray.INDArray;

public class FaceNetEmbed {
    private final String label;
    private final INDArray embedding;

    public FaceNetEmbed(String label, INDArray embedding) {
        this.label = label;
        this.embedding = embedding;
    }

    public INDArray getEmbedding() {
        return this.embedding;
    }

    public String getLabel() {
        return this.label;
    }
}
