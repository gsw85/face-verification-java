# Vision

## FaceID
Kindly Execute `/vision-dl4j/com/skymindglobal/face/FaceID` for realtime inferencing.

### Face Detection
- FaceDetector.OPENCV_DL_FACEDETECTOR (default)
  - OpenCV with prebuilt caffe face detection model
  - [Configuration](https://github.com/skymindglobal/Vision/blob/master/vision-dl4j/src/main/java/com/skymindglobal/face/FaceID.java#L145):
    - Model inputs: Width and Height
    - Threshold
  - Resources:
    - `OpenCVDeepLearningFaceDetector\res10_300x300_ssd_iter_140000.caffemodel`
    - `OpenCVDeepLearningFaceDetector\deploy.prototxt`
- FaceDetector.OPENIMAJ_FKE_FACEDETECTOR
  - OpenIMAJ FKEFaceDetector
  - Configuration
    - Threshold

### Face Identification
- FaceIdentifier.FEATURE_DISTANCE_VGG16
  - Identification based on cosine similarity between images in dictionary and image from webcam.
  - [Configuration](https://github.com/skymindglobal/Vision/blob/master/vision-dl4j/src/main/java/com/skymindglobal/face/FaceID.java#L133)
    - Embeddings Provider (default: `VGG16FeatureProvider`)
    - Dictionary (default: resources `\vgg16_faces_224`)
      Directory of detection targets 
    - numPredicts: 1
      Number of predictions to be display.
    - detectionThreshold: 0.78
    - numSamples: 3
      Average of top 3 per class.
    - minSupport (deprecated)
      Minimum samples in dictionary per class.
  - Resources:
    `\vgg16_faces_224`
        `person A`
            `face1.jpg`
            `face2.jpg`
        `person B`
            `face1.jpg`
            `face2.jpg`
    
### Dataset Preparation
- Using FaceIdentifier.FEATURE_DISTANCE
May invoke [FaceDatasetPreperation](https://github.com/skymindglobal/Vision/blob/master/vision-dl4j/src/main/java/com/skymindglobal/face/identification/training/FaceDatasetPreperation.java) to generate detection targets face images and load into `\vgg16_faces_224` resource directory.
