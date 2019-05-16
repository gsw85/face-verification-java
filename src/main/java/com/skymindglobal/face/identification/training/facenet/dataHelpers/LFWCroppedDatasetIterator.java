package com.skymindglobal.face.identification.training.facenet.dataHelpers;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;

public class LFWCroppedDatasetIterator {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(LFWCroppedDatasetIterator.class);
    private static final int height = 96;
    private static final int width = 96;
    private static final int channels = 3;

    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static File trainSource,testSource;
    private static int trainBatchSize;
    private static int testBatchSize;

    public LFWCroppedDatasetIterator(File train, File test, int trainBatchSize, int testBatchSize) {
        this.trainSource = train;
        this.testSource = test;
        this.trainBatchSize = trainBatchSize;
        this.testBatchSize = testBatchSize;
    }

    public static RecordReaderDataSetIterator trainIterator() throws IOException {
        return makeIterator(trainSource, trainBatchSize);
    }

    public static RecordReaderDataSetIterator testIterator() throws IOException {
        return makeIterator(testSource, testBatchSize);
    }

    private static RecordReaderDataSetIterator makeIterator(File file, int batchSize) throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(new FileSplit(file));
        return new RecordReaderDataSetIterator(recordReader, batchSize, 1, file.listFiles().length);
    }

    public int getNumClass() {
        return trainSource.listFiles().length;
    }
}
