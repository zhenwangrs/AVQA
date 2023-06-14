import sys

sys.path.append('../../siammae_pretrain')

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizerFast, ViltModel, AutoImageProcessor

from models.siammae_pretrain.cross_attn_block import CrossBlock
from models.siammae_pretrain.clip.models_siammae_clip import SiamMAE


class Frame2ClipCMA(nn.Module):
    def __init__(self):
        super(Frame2ClipCMA, self).__init__()
        self.frame2clip_cma = CrossBlock(
            dim=768,
            num_heads=8,
        )

    def forward(self, frame_feats_mean, audio_feats_mean):
        pass


class FusVQAModel(nn.Module):
    def __init__(self, config):
        super(FusVQAModel, self).__init__()
        self.config = config
        self.audio_frame_cma = CrossBlock(
            dim=768,
            num_heads=8,
        )
        self.av_qa_cma = CrossBlock(
            dim=768,
            num_heads=8,
        )
        self.ans_linear = nn.Linear(768, self.config.model.answer_types)
        self.loss = nn.CrossEntropyLoss()
        # self.text_encoder = AutoModel.from_pretrained('roberta-base')
        # self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.vilt = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

    def forward(self, text_input_ids, text_attention_mask, labels, audio_feats_mean, frame_feats_mean, audio_feats,
                frame_feats):
        # text_feats = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask).last_hidden_state
        # audio_frame_feats = self.audio_frame_cma(audio_feats_mean, frame_feats_mean, frame_feats_mean)
        # av_qa_feats = self.av_qa_cma(text_feats, audio_frame_feats, audio_feats_mean)
        # ans_logits = self.ans_linear(torch.mean(av_qa_feats, dim=1))
        # loss = self.loss(ans_logits, labels)

        av_feat = torch.cat([audio_feats_mean,
                             torch.zeros((audio_feats_mean.shape[0], 1, audio_feats_mean.shape[2])).cuda(),
                             frame_feats_mean], dim=1)
        padded_av_feat = torch.nn.functional.pad(av_feat, (0, 0, 0, 23, 0, 0))
        padded_av_feat = padded_av_feat.cuda()
        av_mask = torch.ones(av_feat.shape[:2])
        av_mask = F.pad(av_mask, (0, 23), value=0)
        av_mask = av_mask.cuda()
        fusion_feats = self.vilt(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            image_embeds=padded_av_feat,
            pixel_mask=av_mask,
        )['last_hidden_state']
        ans_logits = self.ans_linear(torch.mean(fusion_feats, dim=1))
        loss = self.loss(ans_logits, labels)
        return loss, ans_logits

    def predict(self, text_input_ids, text_attention_mask, labels, audio_feats_mean, frame_feats_mean, audio_feats,
                frame_feats):
        loss, logits = self.forward(text_input_ids, text_attention_mask, labels, audio_feats_mean, frame_feats_mean,
                                    audio_feats, frame_feats)
        return logits


#####################################################################################################################

class AV_Pairwise_CMA(nn.Module):
    def __init__(self):
        super(AV_Pairwise_CMA, self).__init__()
        self.audio_frame_cma = CrossBlock(
            dim=768,
            num_heads=8,
        )

    def forward(self, audio_feats, frame_feats):
        pass


class FusVQAModel_online(nn.Module):
    def __init__(self, config):
        super(FusVQAModel_online, self).__init__()
        self.config = config
        model = SiamMAE(config)
        model.load_state_dict(torch.load('E:/Research/AVQA/models/siammae_pretrain/ckp/model_10.pth'))
        model = model.cuda()
        self.vit = model.image_mae_encoder
        self.audio_mae = model.audio_mae
        self.vilt = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        self.ans_linear = nn.Linear(768, self.config.model.answer_types)

        self.loss = nn.CrossEntropyLoss()
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

    def forward(self, text_input_ids, text_attention_mask, labels, audio_feats, frame_feats):
        AB, AN, AC, AH, AW = audio_feats.shape  # [4, 10, 1, 128, 128]
        audio_feats = audio_feats.reshape(AB * AN, AC, AH, AW)  # [40, 1, 128, 128]
        FB, FN, FC, FH, FW = frame_feats.shape  # [4, 10, 3, 224, 224]
        frame_feats = frame_feats.reshape(FB * FN, FC, FH, FW)  # [40, 3, 224, 224]

        audio_feats = self.audio_mae.forward_encoder_no_mask(audio_feats)  # [40, 65, 768]
        B, S, H = audio_feats.shape
        audio_feats = audio_feats.reshape(AB, AN, S, H)  # [4, 10, 65, 768]
        frame_feats = self.vit(frame_feats, apply_mask=False).last_hidden_state  # [40, 197, 768]
        B, S, H = frame_feats.shape
        frame_feats = frame_feats.reshape(FB, FN, S, H)  # [4, 10, 197, 768]

        audio_feats_mean = torch.mean(audio_feats, dim=2).squeeze(2)  # [4, 10, 768]
        frame_feats_mean = torch.mean(frame_feats, dim=2).squeeze(2)  # [4, 10, 768]
        av_feat = torch.cat([audio_feats_mean,
                             torch.zeros((audio_feats_mean.shape[0], 1, audio_feats_mean.shape[2])).cuda(),
                             frame_feats_mean], dim=1)
        padded_amount = 144 - (audio_feats_mean.shape[1] + 1 + frame_feats_mean.shape[1])
        padded_av_feat = torch.nn.functional.pad(av_feat, (0, 0, 0, padded_amount, 0, 0))
        padded_av_feat = padded_av_feat.cuda()
        av_mask = torch.ones(av_feat.shape[:2])
        av_mask = F.pad(av_mask, (0, padded_amount), value=0)
        av_mask = av_mask.cuda()

        fusion_feats = self.vilt(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            image_embeds=padded_av_feat,
            pixel_mask=av_mask,
        )['last_hidden_state']
        ans_logits = self.ans_linear(torch.mean(fusion_feats, dim=1))
        loss = self.loss(ans_logits, labels)
        return loss, ans_logits

    def predict(self, text_input_ids, text_attention_mask, labels, audio_feats, frame_feats):
        loss, logits = self.forward(text_input_ids, text_attention_mask, labels, audio_feats, frame_feats)
        return logits
