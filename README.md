# Face Identification Prototypes (DL4J, OpenIMEJ, JavaCV and more)

## FaceID
Kindly Execute [FaceID](https://github.com/skymindglobal/faceverification-java/blob/master/src/main/java/com/skymindglobal/faceverification/FaceID.java) for realtime inferencing.

### Face Detection
- `FaceDetector.OPENCV_DL_FACEDETECTOR` (default): OpenCV with prebuilt caffe face detection model
  - [Configuration](https://github.com/skymindglobal/faceverification-java/blob/master/src/main/java/com/skymindglobal/faceverification/FaceID.java#L151):
    - `imageWidth:300`
    - `imageHeight:300`
    - `detectionThreshold:0.8`
  - Resources:
    - `OpenCVDeepLearningFaceDetector\res10_300x300_ssd_iter_140000.caffemodel`
    - `OpenCVDeepLearningFaceDetector\deploy.prototxt`
- `FaceDetector.OPENIMAJ_FKE_FACEDETECTOR`: OpenIMAJ's FKEFaceDetector
  - Configuration
    - `detectionThreshold:1.0`

### Face Identification
- `FaceIdentifier.FEATURE_DISTANCE_VGG16_PREBUILT`: Identification by highest cosine similarity between webcam image and target embeddings (prebuilt VGG16 model with VGGFACE dataset, layer `fc8` as features).
  - [Configuration](https://github.com/skymindglobal/faceverification-java/blob/master/src/main/java/com/skymindglobal/faceverification/FaceID.java#L134-L136)
    - `FaceFeatureProvider:VGG16FeatureProvider`
    - `dictDir: resources \vgg16_faces_224` detection target faces.
    - `numPredicts:1` number of predictions
    - `detectionThreshold:0.78`
    - `numSamples:3` average of top 3 per class
  - Resources:
    - `\vgg16_faces_224`
      - `person A`
        - `face1.jpg`
        - `face2.jpg`
      - `person B`
        - `face1.jpg`
        - `face2.jpg`
- `FaceIdentifier.CUSTOM_VGG16`: Identification by inference self trained model (may refer [training steps](https://github.com/skymindglobal/faceverification-java/tree/master/src/main/java/com/skymindglobal/faceverification_training/identification/VGG16FaceIdentifier/VGG16Classifier.java))
  - Configuration
    - `numPrediction:3`
- `FaceIdentifier.FEATURE_DISTANCE_FACENET_PREBUILT` (not stable): Identification by highest cosine similarity between webcam image and target embeddings (prebuilt [InceptionResNetv1](https://github.com/davidsandberg/facenet) model  deployed on SKIL).
- `FaceIdentifier.ZHZD`: Identification by inference model trained by zhzd@skymind.cc, mainly for testing purposes.
### Dataset Preparation
- Using FaceIdentifier.FEATURE_DISTANCE: kindly invoke [VGG16FaceDatasetPreperation.java](https://github.com/skymindglobal/faceverification-java/blob/master/src/main/java/com/skymindglobal/faceverification/VGG16FaceDatasetPreperation.java) to extract detection targets face images and load into `\vgg16_faces_224` resource directory.
