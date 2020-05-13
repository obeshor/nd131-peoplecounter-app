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
import time
import socket
import json
import cv2
import os
import sys
import numpy as np
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



def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    model = args.model
    
    DEVICE = args.device
    CPU_EXTENSION = args.cpu_extension
    
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model, CPU_EXTENSION, DEVICE)
    network_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    # Checks for live feed
    if args.input.lower() == "cam" or args.input.lower() == "camera":
        args.input = 0
    elif os.path.splitext(args.input)[1] not in [".mp4", ".jpg", ".jpeg",
                                                 ".png"]:  # assume that we accept only mp4, jpg, jpeg, png
        print("Your input file is not supported! please provide a valid file.")
        return

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    width = int(cap.get(3))
    height = int(cap.get(4))

    in_shape = network_shape['image_tensor']

    #iniatilize variables
    
    duration_prev = 0
    counter_total = 0
    duration = 0
    request_id=0
    report = 0
    counter = 0
    counter_prev = 0
    
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        if key_pressed == 27:
            break

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (in_shape[3], in_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
  

        ### TODO: Start asynchronous inference for specified request ###
        net_input = {'image_tensor': p_frame,'image_info': p_frame.shape[1:]}
        duration_report = None
        infer_network.exec_net(net_input, request_id)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:

            ### TODO: Get the results of the inference request ###
            net_output = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
         
            pointer = 0
            probs = net_output[0, 0, :, 2]
            for i, confidence in enumerate(probs):
                if confidence > prob_threshold:
                    pointer += 1
                    box = net_output[0, 0, i, 3:]
                    position1 = (int(box[0] * width), int(box[1] * height))
                    position2 = (int(box[2] * width), int(box[3] * height))
                    frame = cv2.rectangle(frame, position1, position2, (255, 0, 255), 2)
        
            if pointer != counter:
                counter_prev = counter
                counter = pointer
                if duration >= 3:
                    duration_prev = duration
                    duration = 0
                else:
                    duration += duration_prev 
                    duration_prev = 0  # unknown, not needed in this case
            else:
                duration += 1
                if duration >= 3:
                    report = counter
                    if duration == 3 and counter > counter_prev:
                        counter_total += counter - counter_prev
                    elif duration == 3 and counter < counter_prev:
                        duration_report = int((duration_prev / 10.0) * 1000)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            client.publish('person',
                           payload=json.dumps({
                               'count': report, 'total': counter_total}),
                           qos=0, retain=False)
            if duration_report is not None:
                client.publish('person/duration',
                               payload=json.dumps({'duration': duration_report}),
                               qos=0, retain=False)
 

        ### TODO: Send the frame to the FFMPEG server ###
        #  Resize the frame
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        ### Write an output image if `single_image_mode` ###
        if os.path.splitext(args.input)[1] in [".jpg", ".jpeg", ".png"]:
                cv2.imwrite("outputed" + os.path.splitext(args.input)[1], frame)


    cap.release()
    cv2.destroyAllWindows()
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
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()