# Dependency
OpenCV

# Usage
This repo demonstrates how to use a custom Tensorflow model in OpenCV. 
OpenCV requires a specific pbtxt file to get the model load properly, and to generate this file, we need the frozen graph of the model and the pipeline configuration for the model.
I get the respective files (frozen_inference_graph.pb and ssd_mobilenet_v1_coco.config) from this [project](https://github.com/victordibia/handtracking)
To generate frozen_inference_graph.pbtxt, run tf_text_graph_ssd.py --input frozen_inference_graph.pb --output frozen_inference_graph.pbtxt --config ssd_mobilenet_v1_coco.config
To test using OpenCV to load and do inference, run the test.py file