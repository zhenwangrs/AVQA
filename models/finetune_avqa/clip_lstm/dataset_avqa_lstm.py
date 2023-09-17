import json
import os
import random

import torch
import munch
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoImageProcessor
from models.finetune_musicvqa.utils_finetune import AV2PT


class AVQADataset(Dataset):
    def __init__(self, config, tokenizer, image_processor, mode='train'):
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mode = mode
        self.qa_json_path = self.config.dataset.finetune_mavqa_json_path.format(mode)
        self.qa_data = json.load(open(self.qa_json_path, 'r', encoding='utf-8'))
        self.av_data = json.load(open(self.config.dataset.av_list_json_path, 'r', encoding='utf-8'))
        self.av2pt = AV2PT(config, image_processor)
        self.selected_frame_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

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

    def __len__(self):
        return len(self.qa_data)

    def __getitem__(self, item):
        qa_data = self.qa_data[item]
        video_id = qa_data['video_id']
        video_name = qa_data['video_name']
        audio_list, frame_list = self.get_av_data(video_name)
        audio_feats, frame_feats = self.av2pt.av_to_pt(audio_list, frame_list)

        question_id = qa_data['id']
        question_type = qa_data['question_type']
        question_relation = qa_data['question_relation']
        question = qa_data['question_text']
        multi_choice = qa_data['multi_choice']
        ans_index = qa_data['answer']
        answer = multi_choice[ans_index]
        # shuffle
        if self.mode == 'train' and random.random() < 0.2:
            random.shuffle(multi_choice)
        label = multi_choice.index(answer)
        # multi choice to 0. ans0; 1. ans1; 2. ans2; 3. ans3
        multi_choice = '; '.join([f'{i}. {ans}' for i, ans in enumerate(multi_choice)])
        question = f'Question: {question} [SEP] Answer Candidates: {multi_choice}'
        return question_id, question, label, question_type, audio_feats, frame_feats

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
    mavqa_dataset = AVQADataset(config, tokenizer, image_processor, mode='train')
    mavqa_dataloader = DataLoader(
        mavqa_dataset,
        batch_size=4,
        shuffle=True,
        # num_workers=4,
        # prefetch_factor=2,
        # persistent_workers=True,
        # pin_memory=True,
        drop_last=True,
        collate_fn=mavqa_dataset.collate_fn,
    )

    for data in tqdm(mavqa_dataloader):
        pass
