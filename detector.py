"""
 @Usage: Python interface for YOLOv3-test using Keras
 @author: sun qian
 @date: 2019/9/25
 @Environmental needed:
    - Python 3.7.1
    - Tensorflow-gpu 1.13.1
    - Keras 2.3.0
    - CUDA 10.0
    - cudnn-v7.4.2.24
"""
import argparse
from yolo3.yolo import YOLO
from timeit import default_timer as timer
from PIL import Image
from glob import glob
import os
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings("ignore")


def detect_img(yolov3):
    # single image test
    if opt.single:
        while True:
            img = input('Input image path: ')
            if img == 'quit':
                break
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                start = timer()
                r_image = yolov3.detect_multi_class_image(image)
                end = timer()
                print('Detection time: %.2fs' % (end-start))
                r_image.save('detector.jpg', quality=95)
                r_image.show()

    # batch images test
    else:
        if not os.path.exists(opt.output):
            os.mkdir(opt.output)
        images = glob(opt.input + '/*jpg')

        start = timer()
        for img in tqdm(images):
            image = Image.open(img.replace('\\', '/'))
            name = img.replace(opt.input + '\\', '')
            r_image = yolov3.detect_multi_class_image(image)
            r_image.save(opt.output + name.replace('.jpg', '') + '_detect.jpg', quality=100)
            time.sleep(1)
        end = timer()
        print('Detection time of %d images: %.2fs' % (len(images), (end - start)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--input', type=str,
        default='./data/test_images',
        help='input path of test images'
    )

    parser.add_argument(
        '--output', type=str,
        default='./result/',
        help='output path of detect results'
    )

    parser.add_argument(
        '--model', type=str,                     # set it use command, default value is in yolo.py
        help='path to keras-model weight file'
    )

    parser.add_argument(
        '--anchors', type=str,                    # set it use command, default value is in yolo.py
        help='path to anchor definitions'
    )

    parser.add_argument(
        '--classes', type=str,                    # set it use command, default value is in yolo.py
        help='path to class definitions'
    )

    parser.add_argument(
        '--gpu_num', type=int,                     # set it use command, default value is in yolo.py
        help='Number of GPU to use'
    )

    parser.add_argument(
        '--single', default=False, action="store_true",
        help='single test or batch test, default batch'
    )

    opt = parser.parse_args()
    detect_img(YOLO(**vars(opt)))
