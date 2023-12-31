import json
import os
import random

import munch
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoImageProcessor

from models.finetune_musicvqa.utils_finetune import AV2PT


class MavqaDataset_online(Dataset):
    def __init__(self, config, tokenizer, image_processor, mode='train'):
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mode = mode
        self.qa_json_path = self.config.dataset.finetune_mavqa_json_path.format(mode)
        self.qa_data = json.load(open(self.qa_json_path, 'r', encoding='utf-8'))
        self.av_data = json.load(open(self.config.dataset.av_list_json_path, 'r', encoding='utf-8'))
        self.ans_set =  ['yes', 'no',
                         'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'more than ten',
                         'left', 'middle', 'right', 'simultaneously',
                         'indoor', 'outdoor', 'cello', 'congas', 'pipa', 'ukulele', 'piano', 'accordion', 'clarinet',
                         'guzheng', 'saxophone', 'drum', 'violin', 'bagpipe', 'bassoon', 'acoustic_guitar', 'banjo',
                         'electric_bass', 'flute', 'trumpet', 'erhu', 'xylophone', 'tuba', 'suona']
        self.av2pt = AV2PT(config, image_processor)
        # self.selected_frame_index = [1, 3, 7, 13, 19, 25, 30, 36, 42, 48, 54, 57, 60]
        self.selected_frame_index = [1, 3, 7, 10, 13, 19, 25, 28, 32, 36, 42, 48, 51, 54, 57, 60]

    def get_av_data(self, video_id):
        av_data = self.av_data[video_id]
        audio_data = av_data['audio']
        frame_data = av_data['frame']
        # 每隔N帧采样一帧
        rand_range = 60 // self.config.model.select_num
        start_frame = random.randint(1, rand_range) if not self.config.model.fix_select else rand_range
        selected_frame_index = [start_frame + rand_range * i for i in range(self.config.model.select_num)]
        if self.config.model.fix_select:
            selected_frame_index = self.selected_frame_index
        audio_list = [os.path.join(self.config.dataset.av_data_path, audio_data[index-1]) for index in selected_frame_index]
        frame_list = [os.path.join(self.config.dataset.av_data_path, frame_data[index-1]) for index in selected_frame_index]
        return audio_list, frame_list

    def compose_question(self, sentence, words):
        blanks = []
        for i in range(len(sentence)):
            if sentence[i] == '<':
                blank = '<'
                while sentence[i] != '>':
                    i += 1
                    blank += sentence[i]
                blanks.append(blank)

        if len(blanks) != len(words):
            raise ValueError('The number of blanks is not equal to the number of words.')
        for index, blank in enumerate(blanks):
            # index = blanks.index(blank)
            sentence = sentence.replace(f'{blank}', words[index], 1)

        return sentence

    def __len__(self):
        return len(self.qa_data)

    def __getitem__(self, item):
        qa_data = self.qa_data[item]
        video_id = qa_data['video_id']
        audio_list, frame_list = self.get_av_data(video_id)
        use_augment = True if self.mode == 'train' and self.config.train.use_augment else False
        audio_feats, frame_feats = self.av2pt.av_to_pt(audio_list, frame_list, use_augment=use_augment)

        question_id = qa_data['question_id']
        question_type = eval(qa_data['type'])
        question = qa_data['question_content']
        templ_values = eval(qa_data['templ_values'])
        question = (self.compose_question(question, templ_values) + ' ') * 8
        answer = qa_data['anser']
        labels = self.ans_set.index(answer)
        return question_id, question, labels, question_type, audio_feats, frame_feats

    def collate_fn(self, batch):
        question_ids, questions, labels, question_types, audio_feats, frame_feats = zip(*batch)
        question_ids = list(question_ids)
        questions = self.tokenizer(questions, padding=True, truncation=True, return_tensors='pt', max_length=77)
        text_input_ids = questions['input_ids']
        text_attention_mask = questions['attention_mask']

        labels = torch.tensor(labels)
        question_types = list(question_types)

        # audio feats [B, 10, 1, 128, 128] frame feats [B, 10, 3, 224, 224]
        audio_feats = torch.stack(audio_feats, dim=0)
        frame_feats = torch.stack(frame_feats, dim=0)
        return text_input_ids, text_attention_mask, audio_feats, frame_feats, labels, question_types, question_ids


if __name__ == '__main__':
    config = yaml.load(open('config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    config = munch.munchify(config)
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    mavqa_dataset = MavqaDataset_online(config, tokenizer, image_processor, mode='train')
    mavqa_dataloader = DataLoader(
        mavqa_dataset,
        batch_size=32,
        shuffle=True,
        # num_workers=1,
        # prefetch_factor=2,
        # persistent_workers=False,
        # pin_memory=True,
        drop_last=False,
        collate_fn=mavqa_dataset.collate_fn,
    )

    for data in tqdm(mavqa_dataloader):
        pass
