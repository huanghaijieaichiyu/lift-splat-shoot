import cv2
import os
from tqdm import tqdm

from src.tools import save_path


def vedio_writer(
        img_folder='./runs/prediction/train',
        vedio_path='./runs/prediction'):

    # 图片文件夹路径
    image_folder = img_folder
    # 视频输出路径
    path = save_path(vedio_path, model='vedio')
    os.mkdir(path)
    video_output = os.path.join(path, 'result.avi')
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # 图片文件名列表
    images = os.listdir(image_folder)
    images.sort()  # 不排序会乱
    # 假设所有图片尺寸相同，这里我们只读取第一张图片的尺寸
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(
        video_output, fourcc, 24, (width, height))
    pbar = tqdm(images, total=len(
        images), colour='#8762A5', ncols=200)
    # 将图片逐一写入视频
    for image in pbar:
        img = cv2.imread(os.path.join(img_folder, image))
        video.write(img)

    # 释放VideoWriter对象
    video.release()
    pbar.close()
