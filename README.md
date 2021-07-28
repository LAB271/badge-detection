# Badge Detection

# To run:
git clone
cd to directory and in terminal execute:
export PYTHONPATH="${PYTHONPATH}:/`pwd`"
python3 app/badge-detector/main.py

# Change Log / Process Flow / Updates
The person detection model will be doing a lot of work in areas with a significant amount of people - it needs to be very lightweight & fast, yet still reliable - that's why we chose the ssd model provided by Linzaer. It comes in a couple versions, we're still testing out which one to choose, so far the lightest one (version-RFB-320) seems to be working just fine.

When selecting the badge detection model, we focused more on the accuracy, since the model can theoretically run parallel to the rest of the program. So we chose a pre-trained version of the resnet50, faster-rcnn, which was indeed available to achieve near perfect accuracy (90-98% confidence). However, after building the [firstAttempt.py](https://github.com/nojuskybartas/badge-detection/blob/main/firstAttempt.py) approach to the problem, it was already very clear that this model is way too big, complicated and most importantly - slow, for the given problem.

That's when we started looking for a new approach and the mobilenet-v3 seems to be a great improvement in general performance. It is much less accurate, with quite a few false positives and negatives, however, the inference time is almost 30 times faster. We're still looking how to improve it's accuracy, but for now this is the best option.

[firstAttempt.py](https://github.com/nojuskybartas/badge-detection/blob/main/firstAttempt.py) uses a pandas dataframe to store data, and is so far the fastest approach to the problem (smallest inference time)

[secondAttempt.py](https://github.com/nojuskybartas/badge-detection/blob/main/secondAttempt.py) uses OOP, and is more reliable, and more scalable, however performs a little slower than the first approach. 

## Update (23 July 2021):
- The program now supports self-destructing Person objects - meaning memory usage is now minimized, and unsurprisingly, a macbook laptop can now run the program and check significant amounts (tested on 15-20) of people in real time.
- A single person will only be checked a certain amount of times, and will be assigned a bagde value of either True or False.
- There are now 3 colour codes - light blue for persons being checked, green for persons that have a badge, and red for persons that do not have a badge.
- person bounding boxes are now dynamic, calculated using the ratio between the human head and body - this means that a cutout picture of a person more in the back will cover the same amount of body mass as it would if the person was more in the front.
- I attempted to introduce multi-threading however that was unsuccessful. 

# TODO:
- investigate multithreading
- create a lifetime for the Person objects, to guarantee minimal memory usage
- improve the accuracy of the detection model
- build the classification model
- build the interface
- optimization, mainly for the stream input, as reading the image is so far the heaviest task
- rewrite the ipython training code into regular python syntax

## Update (27 July 2021):
- the Camera is now an object, and therefore can be easily created with all its parameters. The camera also has a parameter for recording and interface. 
- introduced an fps engine
- For dev purposes the camera now stores each detected badge
- general bug fixes and cleanup

## Update (28 July 2021):
- set up the Google Cloud Console environment
- collected more data
- retrained the mobilenet on the new data, which drastically improved the model's performance (it's now equally as precise as the resnet50), and converted the FP32 weights to FP16, which resulted in no quality loss, but halfed the amount of computation required to check the badges. The new model weights are saved under mobilenet_v2
- TODO: investigate weight conversion to uint8

## [ultra-lightweight face detection model](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
This model is a lightweight facedetection model designed for edge computing devices.
- In terms of model size, the default FP32 precision (.pth) file size is **1.04~1.1MB**, and the inference framework int8 quantization size is about **300KB**.
- In terms of the calculation amount of the model, the input resolution of 320x240 is about **90~109 MFlops**.
- There are two versions of the model, version-slim (network backbone simplification,slightly faster) and version-RFB (with the modified RFB module, higher precision).
- Widerface training pre-training model with different input resolutions of 320x240 and 640x480 is provided to better work in different application scenarios.
- Support for onnx export for ease of migration and inference.
- [Provide NCNN C++ inference code](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/ncnn).
- [Provide MNN C++ inference code](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/MNN), [MNN Python inference code](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/MNN/python), [FP32/INT8 quantized models](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/MNN/model).
- [Provide Caffe model](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/caffe/model) and [onnx2caffe conversion code](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/caffe).
- [Caffe python inference code](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/caffe/ultra_face_caffe_inference.py) and [OpencvDNN inference code](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/caffe/ultra_face_opencvdnn_inference.py).


### Accuracy, speed, model size comparison
The training set is the VOC format data set generated by using the cleaned widerface labels provided by [Retinaface](https://github.com/deepinsight/insightface/blob/master/RetinaFace/README.md)  in conjunction with the widerface data set (PS: the following test results were obtained by myself, and the results may be partially inconsistent).
#### Widerface test
- Test accuracy in the WIDER FACE val set (single-scale input resolution: **320*240 or scaling by the maximum side length of 320**)

Model|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
libfacedetection v1（caffe）|0.65 |0.5       |0.233
libfacedetection v2（caffe）|0.714 |0.585       |0.306
Retinaface-Mobilenet-0.25 (Mxnet)   |0.745|0.553|0.232
version-slim|0.77     |0.671       |0.395
version-RFB|**0.787**     |**0.698**       |**0.438**


- Test accuracy in the WIDER FACE val set (single-scale input resolution: **VGA 640*480 or scaling by the maximum side length of 640** )

Model|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
libfacedetection v1（caffe）|0.741 |0.683       |0.421
libfacedetection v2（caffe）|0.773 |0.718       |0.485
Retinaface-Mobilenet-0.25 (Mxnet)   |**0.879**|0.807|0.481
version-slim|0.853     |0.819       |0.539
version-RFB|0.855     |**0.822**       |**0.579**

 
### Third-party related projects
 - [NNCase C++ inference code](https://github.com/kendryte/nncase/tree/master/examples/fast_facedetect)
 - [UltraFaceDotNet (C#)](https://github.com/takuya-takeuchi/UltraFaceDotNet)
 - [faceDetect-ios](https://github.com/Ian778/faceDetect-ios)
 - [Android-FaceDetection-UltraNet-MNN](https://github.com/jackweiwang/Android-FaceDetection-UltraNet-MNN)
 - [Ultra-Tensorflow-Model-Converter](https://github.com/jason9075/Ultra-Light-Fast-Generic-Face-Detector_Tensorflow-Model-Converter)
  
###  Reference
- [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd)
- [libfacedetection](https://github.com/ShiqiYu/libfacedetection/)
- [RFBNet](https://github.com/ruinmessi/RFBNet)
- [RFSong-779](https://github.com/songwsx/RFSong-779)
- [Retinaface](https://github.com/deepinsight/insightface/blob/master/RetinaFace/README.md)
