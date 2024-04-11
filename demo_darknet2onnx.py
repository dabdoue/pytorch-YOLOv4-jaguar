from datetime import datetime
import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime

from tool.utils import *
from tool.darknet2onnx import *
from PIL import Image

def main(cfg_file, namesfile, weight_file, image_path, batch_size):
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    onnx_name=f"{os.path.splitext(weight_file)[0]}_{current_time}.onnx"
    if batch_size <= 0:
        
        onnx_path_demo = transform_to_onnx(cfg_file, weight_file, batch_size)
    else:
        # Transform to onnx as specified batch size
        # transform_to_onnx(cfg_file, weight_file, batch_size)
        # Transform to onnx as demo
        onnx_path_demo = transform_to_onnx(cfg_file, weight_file, 1, onnx_file_name=onnx_name)

    session = onnxruntime.InferenceSession(onnx_path_demo, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider'])
    # session = onnx.load(onnx_path)
    print("The model expects input shape: ", session.get_inputs()[0].shape)

    image_src = cv2.imread(image_path)
    detect(session, image_src, namesfile)
    onnx_model = onnx.load(onnx_path_demo)
    # adding metadata
    
    meta = onnx_model.metadata_props.add()
    meta.key = "revision"
    meta.value = current_time
    meta.key = "cfg_path"
    meta.value = cfg_file
    meta.key = "copyright"
    meta.value = "Copyright Jaguar Software Development © 2024"
    meta.key = "training_data_path"
    if "resize" in weight_file:
        meta.value = "/usr/local/yolo_training_constants/baseline-resized_train.txt"
    else:
        meta.value = "/usr/local/yolo_training_constants/baseline_train.txt"

    print(onnx_path_demo)
    onnx.save(onnx_model, onnx_path_demo)



def detect(session, image_src, namesfile):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # # Input
    img_in = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("resized_img.png", img_in)

    transposed_img = []
    for channel in range(len(img_in[0][0])):
        transposed_channel = []
        for i in range(len(img_in)):
            transposed_row = []
            for j in range(len(img_in[0])):
                transposed_row.append(img_in[i][j][channel])
            transposed_channel.append(transposed_row)
        transposed_img.append(transposed_channel)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    # reshaped_image = np.array(transposed_img).reshape((IN_IMAGE_H, IN_IMAGE_W, 3)).astype(np.uint8)
    # cv2.imwrite("output_image.png", reshaped_image)

    img_in = np.expand_dims(img_in, axis=0).astype(np.float32)
    

    # # Save the image to a file
    
    img_in /= 255.0
    # print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name
    input_img = np.array([image_src]).astype(np.float32)
    print(input_img.shape)
    inf_time = time.time()
    outputs = session.run(None, {input_name: img_in})
    
    # outputs = session.run(None, {input_name: img_in})
    print("inf time: " + str(time.time() - inf_time))
    
    # session1 = onnxruntime.InferenceSession("postprocess.onnx")
    # print(session1.get_inputs()[0].name)
    # print("The model expects input shape: ", session1.get_inputs()[0].shape)
    # print("The model expects input shape: ", session1.get_inputs()[1].shape)
    # input_name1 = session1.get_inputs()[0].name
    # input_name11 = session1.get_inputs()[1].name
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # outputs1 = session1.run(None, {'input': outputs[0], 'input1': outputs[1]})
    # print(outputs1.shape)

    boxes = post_processing(image_src, 0.5, 0.6, outputs)
    print(boxes[0])
    class_names = load_class_names(namesfile)
    plot_boxes_cv2(image_src, boxes[0], savename='predictions_onnx.jpg', class_names=class_names)



if __name__ == '__main__':
    print("Converting to onnx and running demo ...")
    if len(sys.argv) == 6:
        cfg_file = sys.argv[1]
        namesfile = sys.argv[2]
        weight_file = sys.argv[3]
        image_path = sys.argv[4]
        batch_size = int(sys.argv[5])
        main(cfg_file, namesfile, weight_file, image_path, batch_size)
    else:
        print('Please run this way:\n')
        print('  python demo_onnx.py <cfgFile> <namesFile> <weightFile> <imageFile> <batchSize>')
