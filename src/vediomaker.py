import cv2
import os


def vedio_writer(
        img_folder='./imgs',
        save_path='./imgs'):

    # 图片文件夹路径
    image_folder = img_folder
    # 视频输出路径
    video_output = save_path

    # 图片文件名列表
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    # 假设所有图片尺寸相同，这里我们只读取第一张图片的尺寸
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 定义视频编码和创建VideoWriter对象
    video = cv2.VideoWriter(
        video_output, cv2.VideoWriter_fourcc(*'mp4v'), 60, (width, height))

    # 将图片逐一写入视频
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # 释放VideoWriter对象
    video.release()
