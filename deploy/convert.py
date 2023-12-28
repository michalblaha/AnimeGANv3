# -*- coding: utf-8 -*-
# @Time    : 2021/8/31 19:20
# @Author  : Xin Chen
# @File    : test_by_onnx.py
# @Software: PyCharm

import onnxruntime as ort
import time, os, cv2,argparse
import numpy as np
pic_form = ['.jpeg','.jpg','.png','.JPEG','.JPG','.PNG']
from glob import glob

class TrimmedStr(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, str):
            trimmed_values = values.strip()
        elif isinstance(values, list):
            trimmed_values = [v.strip() for v in values]
        setattr(namespace, self.dest, trimmed_values)

def parse_args():
    desc = "Tensorflow implementation of AnimeGANv3"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input_img_dir', action=TrimmedStr, default='./tmp', help='input picture file ')
    parser.add_argument('-m', '--model_path', action=TrimmedStr, default='AnimeGANv3_Hayao_36',  help='path to model')
    # parser.add_argument('-o', '--output_path', type=str, default='./tmp' ,help='output file')
    parser.add_argument('-d','--device', action=TrimmedStr, default='cpu', choices=["cpu","gpu"," cpu"," gpu"] ,help='device GPU or CPU')
    return parser.parse_args()

def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def process_image(img, model_name):
    h, w = img.shape[:2]
    # resize image to multiple of 8s
    def to_8s(x):
        # If using the tiny model, the multiple should be 16 instead of 8.
        if 'tiny' in os.path.basename(model_name) :
            return 256 if x < 256 else x - x % 16
        else:
            return 256 if x < 256 else x - x % 8
    img = cv2.resize(img, (to_8s(w), to_8s(h)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/ 127.5 - 1.0
    return img

def load_test_data(image_path, model_name):
    img0 = cv2.imread(image_path).astype(np.float32)
    img = process_image(img0, model_name)
    img = np.expand_dims(img, axis=0)
    return img, img0.shape

def save_images(images, image_path, size):
    images = (np.squeeze(images) + 1.) / 2 * 255
    images = np.clip(images, 0, 255).astype(np.uint8)
    images = cv2.resize(images, size)
    cv2.imwrite(image_path, cv2.cvtColor(images, cv2.COLOR_RGB2BGR))

def Convert(input_img_path, model ="AnimeGANv3_Hayao_36", device="cpu"):
    # result_dir = opj(output_path, style_name)

    onnx = "/app/deploy/" + model + ".onnx"
    # Check if the input file has a valid image format
    if not os.path.splitext(input_img_path)[-1] in pic_form:
        raise ValueError("The input file is not a valid image format")

    if ort.get_device() == 'GPU' and device == "gpu":
        session = ort.InferenceSession(onnx, providers = ['CUDAExecutionProvider','CPUExecutionProvider',])
    else:
        session = ort.InferenceSession(onnx, providers=['CPUExecutionProvider', ])
    x = session.get_inputs()[0].name
    y = session.get_outputs()[0].name

    begin = time.time()

    t = time.time()
    sample_image, shape = load_test_data(input_img_path, onnx)
    # output name to .done.originalExtension
    base, extension = os.path.splitext(input_img_path)
    image_path = base + ".done" + extension

 
    fake_img = session.run(None, {x : sample_image})
    save_images(fake_img[0], image_path, (shape[1], shape[0]))
    print(f'Processed image: {input_img_path}, image size: {shape[1], shape[0]}, time: {time.time() - t:.3f} s')

    end = time.time()
    print(f'Total processing time: {end - begin} s')

if __name__ == '__main__':

    # onnx_file = 'AnimeGANv3_Hayao_36.onnx'
    # input_imgs_path = 'pic'
    # output_path = 'AnimeGANv3_Hayao_36'
    # Convert(input_imgs_path, output_path, onnx_file)

    arg = parse_args()
    Convert(arg.input_img_dir, arg.model_path, arg.device.strip())

