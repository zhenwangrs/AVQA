import json

j = json.load(open('D:\dataset\data_pretrain/av_finetune.json', 'r', encoding='utf-8'))
new_j = {}
for item in j:
    video_id = item['video_id']
    audios = item['audio']
    frames = item['frame']
    audios = [audio.replace('D:\\dataset\\data_pretrain/av_1fps/', '').replace('\\', '/') for audio in audios]
    frames = [frame.replace('D:\\dataset\\data_pretrain/av_1fps/', '').replace('\\', '/') for frame in frames]
    new_j[video_id] = {'audio': audios, 'frame': frames}
json.dump(new_j, open('D:\dataset\data_pretrain/av_finetune_new.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
