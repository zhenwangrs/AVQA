import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import os
import subprocess

from tqdm import tqdm


def clip_video(video_path, save_folder):
    # 设置保存截图和音频的文件夹路径
    # save_folder = os.path.splitext(video_path)[0]
    if os.path.exists(save_folder):
        return

    os.makedirs(save_folder, exist_ok=True)
    # 设置截图保存的帧率
    frame_rate = 1  # 每秒保存一张截图
    # 设置音频保存的采样率
    audio_sample_rate = 16000  # 16kHz
    # 加载视频
    video_capture = cv2.VideoCapture(video_path)
    # 获取视频的总帧数和帧率
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    # 计算每隔多少帧保存一张截图
    frame_interval = int(fps / frame_rate)
    # 初始化计数器
    frame_count = 0
    save_count = 0
    # 逐帧读取视频
    while True:
        ret, frame = video_capture.read()
        if frame_count == 0:
            save_path = os.path.join(save_folder, f'frame_{save_count}.jpg')
            cv2.imwrite(save_path, frame)
        if not ret:
            break

        frame_count += 1
        # 如果达到保存截图的帧率
        if frame_count % frame_interval == 0:
            save_count += 1
            # 保存截图
            save_path = os.path.join(save_folder, f'frame_{save_count}.jpg')
            cv2.imwrite(save_path, frame)
        # 如果达到保存音频的帧率
        if frame_count % int(fps) == 0:
            # 提取音频
            audio_save_path = os.path.join(save_folder, f'audio_{save_count}.wav')
            start_time = str((frame_count / fps) - 1)

            with open(os.devnull, 'w') as devnull:
                subprocess.call(['ffmpeg', '-i', video_path, '-vn', '-ac', '1', '-ar', str(audio_sample_rate),
                                 '-f', 'wav', '-ss', start_time, '-t', '1', audio_save_path],
                                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    # 释放视频对象
    video_capture.release()


def clip_video_dir(video_dir_path, save_dir):
    for video_name in tqdm(os.listdir(video_dir_path)):
        video_path = os.path.join(video_dir_path, video_name)
        save_folder = os.path.join(save_dir, video_name.split('.')[0])
        clip_video(video_path, save_folder)


if __name__ == '__main__':
    # 设置视频文件路径
    video_path = 'F:/Research/AVQA/music_avqa/MUCIS-AVQA-videos-Synthetic/'
    save_dir = 'F:/Research/AVQA/music_avqa/data_pretrain/av_1fps/'
    clip_video_dir(video_path, save_dir)
