"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np
from random import randint

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def draw_bounding_boxes(frame, result,prob_threshold,width,height):
    '''
    Draw bounding boxes onto the frame.
    '''
    person_in_frame = 0
    PERSON_CLASS = 1
    for obj in result[0][0]:
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * width)
            ymin = int(obj[4] * height)
            xmax = int(obj[5] * width)
            ymax = int(obj[6] * height)
            class_id = int(obj[1])
            color = (0, 0, 255)
            if class_id == PERSON_CLASS:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                person_in_frame += 1
    return frame, person_in_frame

def infer_on_stream(args,client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    frame_count = 0
    frame_time = 0
    
    duration_prev = 0
    total_count = 0
    time_thresh = 0    
    person_count_in_each_frame = 0
    last_count = 0
    previous_last_count = 0
    request_id = 0
    
    font_scale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Flag for the input image
    single_image_mode = False
    
    # Initialise the class
    infer_network = Network()
    
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    model = args.model
    device = args.device
    CPU_EXTENSION = args.cpu_extension

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model, device, CPU_EXTENSION)
    infer_network_input_shape = infer_network.get_input_shape()
   
    
    # Check if the input is a webcam
    if args.input == 'CAM':
        input_Type = 0
        
    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_Type = args.input

    # Checks for video file
    else:
        input_Type = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    ### TODO: Handle the input stream ###
    input_stream = cv2.VideoCapture(input_Type)
    if input_Type:
        input_stream.open(args.input)
    if not input_stream.isOpened():
        log.error("ERROR! Unable to open video source")
    
    # Grab the shape of the input 
    width = int(input_stream.get(3))
    height = int(input_stream.get(4))
    
    if not single_image_mode:
        # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
        # on Mac, and `0x00000021` on Linux
        # 100x100 to match desired resizing
        out = cv2.VideoWriter('output_video.mp4', 0x00000021, 30, (width,height))
    else:
        out = None
   
    ### TODO: Loop until stream is over ###
    while input_stream.isOpened():
        ### TODO: Read from the video capture ###
        flag,frame = input_stream.read()
        if not flag:
            break
        frame_count += 1
        t = time.time()
        key_pressed = cv2.waitKey(60)
        
        ### TODO: Pre-process the image as needed ### n c h w
        preProcessed_frame = cv2.resize(frame, (infer_network_input_shape[3], infer_network_input_shape[2]))
        preProcessed_frame = preProcessed_frame.transpose((2,0,1))
        preProcessed_frame = preProcessed_frame.reshape(1, *preProcessed_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        inferencing_start = time.time()
        total_time_spent = None
        infer_network.exec_net(request_id, preProcessed_frame)

        ### TODO: Wait for the result ###
        if infer_network.wait(request_id) == 0:
            
            detection_time = time.time() - inferencing_start
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            frame, current_count = draw_bounding_boxes(frame,result,prob_threshold,width,height)
            inference_time_message = "Inference time: {:.3f}ms".format(detection_time * 1000)
            cv2.putText(frame, inference_time_message, (25, 25),cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 10, 250),1)
    
            ### TODO: Extract any desired stats from the results ###
            if current_count == last_count:
                time_thresh += 1
                if time_thresh >= 10:
                    person_count_in_each_frame = last_count
                    if time_thresh == 10 and last_count > previous_last_count:
                        total_count += last_count - previous_last_count
                    elif time_thresh == 10 and last_count < previous_last_count:
                        total_time_spent = int((duration_prev / 10.0) * 1000) # in ms
            else:
                previous_last_count = last_count
                last_count = current_count
                if time_thresh >= 10:
                    duration_prev = time_thresh
                    time_thresh = 0
                else:
                    time_thresh = duration_prev + time_thresh
            
            frame_time += time.time() - t
            fps = frame_count / float(frame_time)
            fps_label = "FPS : {:.2f}".format(fps)
            cv2.putText(frame, fps_label,(25,100),font, font_scale,(0, 0, 255), 1)
            
            current_count_label = "People in Frame : {:.2f}".format(current_count)
            cv2.putText(frame, current_count_label,(25,50),font, font_scale,(0, 255, 0), 1) 
            
            total_count_label = "Total People Detected : {:.2f}".format(total_count)
            cv2.putText(frame, total_count_label,(25,75),font, font_scale,(0, 255, 0), 1)
         
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            client.publish("person", json.dumps({"count": current_count,"total": total_count}))
            if total_time_spent is not None:
                client.publish("person/duration",json.dumps({"duration": total_time_spent}))
            
        ### TODO: Send the frame to the FFMPEG server ###
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        # Break if escape key pressed
        if key_pressed == 27:
            break

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            frame = cv2.resize(frame, (1980, 1080))
            cv2.imwrite('output_image.jpg', frame)
        else:
            out.write(frame)
        
    # Release the capture and destroy any OpenCV windows
    if not single_image_mode:
        out.release()
    input_stream.release()
    cv2.destroyAllWindows()
        
    ### TODO: Disconnect from MQTT
    client.disconnect()


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args,client)


if __name__ == '__main__':
    main()