# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers
OpenVINO Toolkit Documentations has a list of Supported Framework Layers for DL Inference. Custom layers are layers that are not included in the list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

The process behind converting custom layers involves two necessary custom layer extensions:
- Custom Layer Extractor: Responsible for identifying the custom layer operation and extracting the parameters for each instance of the custom layer. The layer parameters are stored per instance and used by the layer operation before finally appearing in the output IR. Typically the input layer parameters are unchanged, which is the case covered by this tutorial
- Custom Layer Operation : Responsible for specifying the attributes that are supported by the custom layer and computing the output shape for each instance of the custom layer from its parameters.

Some of the potential reasons for handling custom layers are...

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...
  ### Model size

|                   | SSD Inception V2 | SSDLite Mobilenet| SSD MobileNet V2 | Person-detection-retail-0013 |
| ----------------- | ---------------- | ---------------- | ---------------- |------------------------------|
| Before Conversion | 98 MB            | 8.6 MB           | 33 MB            | N/A                          |
| After Conversion  | 95 MB            | 17.2 MB          | 31 MB            | 1.4 MB                       |

### Inference Time

|                   | SSD Inception V2 | SSDLite Mobilenet| SSD MobileNet V2  | Person-detection-retail-0013 |
| ----------------- | ---------------- | ---------------- | ----------------- |------------------------------|
| Before Conversion | 42 ms            | 27  ms           |  31 ms            |  N/A                         |
| After Conversion  | 158 ms           | 31 ms            |  68 ms            |  44ms                        |

### FPS

|                   | SSD Inception V2 | SSDLite Mobilenet| SSD MobileNet V2  | Person-detection-retail-0013 |
| ----------------- | ---------------- | ---------------- | ----------------- |------------------------------|
| FPS               | 4.50             | 10.71            |  7.62             |  9.30                         |

## Assess Model Use Cases

Some of the potential use cases of the people counter app are : Retail Analysis, security systems, Queue Management and Space management applications.

Each of these use cases would be useful because it can help save lives in case of pandemics with social distancing enforcement, give businesses insights on how to provide better customer satisfaction and can help companies generate profits through identifying the type of customers and align themselves to their preferences. It can also help architects in case of Airports and Stations to handle crowds and peak times effectively

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:
 - *Lighting -*  Perhaps no other aspects of a computer vision model has consistently caused delays, false detections and detection failures than lighting. In an ideal scenario, lighting would maximize the contrast on features of interest, and would in turn make it easier for a model to detect separate instances of object, in our case person. Since most of the use cases of a people counter application rely on a static CCTV camera, it is critical to have a proper lighting in the area it aims to cover or it may cause false detections or no-detection at all.
- *Model Accuracy -* The model needs to be highly accurate if deployed in a mass scale as it may cause false detections or no-detection which can produce misleading data, and in case of retail applications may cause the company or business to lose money. 
- *Image Size/Focal Length -* It is critical to have a sharp and high resolution image as an input to our model to make it easy for it to perform segmentation easily and keep the features of interest detectable.

## Model Research
In investigating potential people counter models, I tried each of the following three models:

- Model 1: SSD Inception V2
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments
     *Download the model tar file*
    ```
    wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
    ```

    *Extract the tar file*
    ```
    tar -xvf ssd_inception_v2_coco_2018_01_28.tar.gz
    ```
    *To convert is using Model Optimizer*
    ```
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
    ```
  
  - The model was insufficient for the app because it had pretty high latency in making predictions ~155 ms. It made accurate predictions but due to a very huge tradeoff in inference time, the model could not be used
  - I tried to improve the model for the app by reducing the precision of weights, however this had a very huge impact on the accuracy
  
- Model 2:  SSD Lite Mobilenet V2
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments:
    *Download the model tar file*
    ```
    wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
    ```

    *Extract the tar file*
    ```
    tar -xvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
    ```
    *To convert is using Model Optimizer*
    ```
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
    ```
  - The model was insufficient for the app because the model was not  appropriate in terms of detecting the people facing backwards althought the inference speed was very fast inference , around 31 ms. Also, it is sometimes impossible to draw the selection boxes correctly 
  - I tried to improve the model for the app by ignoring these intermediate calculation errors

- Model 3: SSD Mobilenet V2
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments
    *Download the model tar file*
    ```
    wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    ```

    *Extract the tar file*
    ```
    tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    ```
    *To convert is using Model Optimizer*
    ```
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
    ```
  - The model was insufficient for the app because the model lost some accuracy but the performance is much better. The bounding boxes were all over the place and hence did not give accurate results on count
  - I tried to improve the model for the app by reducing the probablity threshold but it landed in getting false positives a lot.

## Model Used
As having explained above the issues I faced with some models so I ended up using  a model  from the OpenVino Model zoo: [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)

That model is in fact based on the MobileNet model, the MobileNet model performed well for me considering latency and size apart of few inference errors. This model has fixed that error.

 - Navigate to the directory containing the Model Downloader:

   ```cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader```
 - Download Model
 
   ````sudo ./downloader.py --name person-detection-retail-0013 --precisions FP16 -o /home/workspace````