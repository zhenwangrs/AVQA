import json
import os
import random

from tqdm import tqdm


def build_av_triple(av_dir, output_json_path, max_per_av=5):
    av_triples = []
    sub_dirs = [dir for dir in os.listdir(av_dir) if os.path.isdir(os.path.join(av_dir, dir))]
    for sub_dir in sub_dirs:
        video_dirs = os.listdir(os.path.join(av_dir, sub_dir))
        for video_dir in tqdm(video_dirs):
            video_path = os.path.join(av_dir, sub_dir, video_dir)
            audios = [audio for audio in os.listdir(video_path) if audio.endswith('.wav')]
            frames = [frame for frame in os.listdir(video_path) if frame.endswith('.jpg')]
            # if len(audios) != len(frames) - 1:
            #     print(f'{video_path} audio: {len(audios)}, frame: {len(frames)}')
            #     continue

            selected_audios = random.sample(audios, min(len(audios), max_per_av))
            for audio in selected_audios:
                audio_index = int(audio.split('.')[0].split('_')[-1])
                frame1 = f'frame_{audio_index - 1}.jpg'
                frame2 = f'frame_{audio_index}.jpg'

                audio_path = f'{video_path}/{audio}'
                frame1_path = f'{video_path}/{frame1}'
                frame2_path = f'{video_path}/{frame2}'
                if not os.path.exists(audio_path) or not os.path.exists(frame1_path) or not os.path.exists(frame2_path):
                    # print(f'{video_path} not exists')
                    continue

                av_triples.append({
                    'frame1': f'{sub_dir}/{video_dir}/{frame1}',
                    'frame2': f'{sub_dir}/{video_dir}/{frame2}',
                    'wav': f'{sub_dir}/{video_dir}/{audio}'
                })
    print(len(av_triples))
    json.dump(av_triples, open(output_json_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # selected_frame_index = [1, 3, 7, 10, 13, 19, 25, 28, 32, 36, 42, 48, 51, 54, 57, 60]
    build_av_triple(
        av_dir='D:/dataset/pretrain_data/',
        output_json_path='D:/dataset/pretrain_data/av_1fps_10.json',
        max_per_av=10
    )
