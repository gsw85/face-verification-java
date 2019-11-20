package com.skymindglobal.faceverification_training.identification.feature.FaceNetNN4Small2FaceFeatureProvider;

import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import java.io.File;
import java.io.IOException;
import java.util.Map;

public class ComputeL2 {
    public static void main(String[] args) throws IOException {
//        ComputationGraph model = ModelSerializer.restoreComputationGraph("C:\\Users\\choowilson\\Downloads\\FaceNetNN4Small2_embedding_cus2.zip");
        ComputationGraph model = ModelSerializer.restoreComputationGraph("C:\\Users\\choowilson\\Desktop\\RamokConfigNWeights.zip",false);
        ComputationGraph genEmbd_model = new TransferLearning.GraphBuilder(model)
                .setFeatureExtractor("encodings") // the L2Normalize vertex and layers below are frozen
                .removeVertexAndConnections("lossLayer")
                .setOutputs("encodings")
                .build();
        genEmbd_model.init();
        System.out.println(genEmbd_model.summary());

        //        Get Embeddings  of ImageA
        INDArray testimage = read("C:\\Users\\choowilson\\lfw-Wilson\\lfw_train_cropped224\\Angelina_Jolie\\Angelina_Jolie_0019.jpg");

        Map<String, INDArray> output  = genEmbd_model.feedForward(normalize(testimage),false);
        GraphVertex embeddings = genEmbd_model.getVertex("encodings");
        INDArray dense = output.get("dense");
        embeddings.setInputs(dense);
        INDArray embeddingValues = embeddings.doForward(false, LayerWorkspaceMgr.builder().defaultNoWorkspace().build());
        System.out.println("dense =                 " + dense);
        System.out.println("encodingsValues =                 " + embeddingValues);

//        Get Embeddings  of ImageB
//        INDArray testimageB = loader.asMatrix("C:\\Users\\choowilson\\Desktop\\gitrepos\\nov11\\faceverification-java\\src\\main\\resources\\vgg16_faces_224\\Evonne\\IMG_20190422_223921.jpg");
//        INDArray testimageB = loader.asMatrix("C:\\Users\\choowilson\\Desktop\\gitrepos\\KlevisRamok\\ComputerVision\\FaceRecognition\\src\\main\\resources\\images\\Ariel_Sharon\\Ariel_Sharon_0006.jpg");
//
//        Map<String, INDArray> output2  = genEmbd_model.feedForward(testimageB,false);
//        GraphVertex embeddings2 = genEmbd_model.getVertex("embeddings");
//        INDArray dense2 = output2.get("dense");
//        embeddings2.setInputs(dense);
//        INDArray embeddingValues2 = embeddings.doForward(false, LayerWorkspaceMgr.builder().defaultNoWorkspace().build());
//        System.out.println("dense =                 " + dense2);
//        System.out.println("encodingsValues =                 " + embeddingValues2);
//        double L2Distance = embeddingValues.distance2(embeddingValues2);
//        System.out.println(L2Distance);

    }
    private static final NativeImageLoader LOADER = new NativeImageLoader(96, 96, 3);

    private static INDArray transpose(INDArray indArray1) {
        INDArray one = Nd4j.create(new int[]{1, 96, 96});
        one.assign(indArray1.get(NDArrayIndex.point(0), NDArrayIndex.point(2)));
        INDArray two = Nd4j.create(new int[]{1, 96, 96});
        two.assign(indArray1.get(NDArrayIndex.point(0), NDArrayIndex.point(1)));
        INDArray three = Nd4j.create(new int[]{1, 96, 96});
        three.assign(indArray1.get(NDArrayIndex.point(0), NDArrayIndex.point(0)));
        return Nd4j.concat(0, one, two, three).reshape(new int[]{1, 3, 96, 96});
    }
    private static INDArray read(String pathname) throws IOException {
        Mat imreadimg = opencv_imgcodecs.imread(new File(pathname).getAbsolutePath(), 1);
        INDArray indArray = LOADER.asMatrix(imreadimg);
        return transpose(indArray);
    }
    private static INDArray normalize(INDArray read) {
        return read.div(255.0);
    }
}
