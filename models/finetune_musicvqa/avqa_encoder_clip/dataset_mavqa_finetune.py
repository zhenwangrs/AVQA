import json
import os

import torch
import munch
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


class MavqaDataset(Dataset):
    def __init__(self, config, tokenizer, mode='train'):
        self.config = config
        self.tokenizer = tokenizer
        self.qa_json_path = self.config.dataset.finetune_mavqa_json_path.format(mode)
        self.qa_data = json.load(open(self.qa_json_path, 'r', encoding='utf-8'))
        self.ans_set =  ['yes', 'no',
                         'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'more than ten',
                         'left', 'middle', 'right', 'simultaneously',
                         'indoor', 'outdoor', 'cello', 'congas', 'pipa', 'ukulele', 'piano', 'accordion', 'clarinet',
                         'guzheng', 'saxophone', 'drum', 'violin', 'bagpipe', 'bassoon', 'acoustic_guitar', 'banjo',
                         'electric_bass', 'flute', 'trumpet', 'erhu', 'xylophone', 'tuba', 'suona']

    def load_data(self, qa_json_path):
        pass

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
        question_id = qa_data['question_id']
        question_type = eval(qa_data['type'])
        question = qa_data['question_content']
        templ_values = eval(qa_data['templ_values'])
        question = self.compose_question(question, templ_values)
        answer = qa_data['anser']
        ans_index = self.ans_set.index(answer)

        pt_data = torch.load(os.path.join(self.config.dataset.finetune_pt_dir, f'{video_id}.pt'))
        # audio_feats = pt_data['audio_feats']
        # frame_feats = pt_data['frame_feats']
        audio_feats = None
        frame_feats = None
        audio_feats_mean = pt_data['audio_feats_mean']
        frame_feats_mean = pt_data['frame_feats_mean']
        return question_id, question, ans_index, question_type, \
            audio_feats, frame_feats, audio_feats_mean, frame_feats_mean

    def collate_fn(self, batch):
        question_ids, question, ans_index, question_types, audio_feats, frame_feats, audio_feats_mean, frame_feats_mean = zip(*batch)
        question_ids = list(question_ids)
        question = self.tokenizer(question, padding=True, truncation=True, return_tensors='pt', max_length=40)
        text_input_ids = question['input_ids']
        text_attention_mask = question['attention_mask']
        # audio_feats = torch.stack(audio_feats)
        # frame_feats = torch.stack(frame_feats)
        audio_feats = None
        frame_feats = None
        audio_feats_mean = torch.stack(audio_feats_mean)
        frame_feats_mean = torch.stack(frame_feats_mean)
        labels = torch.tensor(ans_index)
        question_types = list(question_types)
        return text_input_ids, text_attention_mask, audio_feats, frame_feats, \
            audio_feats_mean, frame_feats_mean, labels, question_types, question_ids


if __name__ == '__main__':
    config = yaml.load(open('E:/Research/AVQA/models/siammae_pretrain/config.yaml', 'r', encoding='utf-8'),
                       Loader=yaml.FullLoader)
    config = munch.munchify(config)
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    mavqa_dataset = MavqaDataset(config, tokenizer)
    mavqa_dataloader = DataLoader(
        mavqa_dataset,
        batch_size=32,
        shuffle=True,
        # num_workers=4,
        # prefetch_factor=2,
        # persistent_workers=True,
        # pin_memory=True,
        drop_last=True,
        # collate_fn=mavqa_dataset.collate_fn,
    )

    for data in tqdm(mavqa_dataloader):
        pass
