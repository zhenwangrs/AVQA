import sys

sys.path.append('../../siammae_pretrain')

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizerFast, ViltModel, AutoImageProcessor

from models.siammae_pretrain.models_siammae import SiamMAE


class FusVQAModel_online(nn.Module):
    def __init__(self, config):
        super(FusVQAModel_online, self).__init__()
        model = SiamMAE(config)
        model.load_state_dict(torch.load('E:/Research/AVQA/models/siammae_pretrain/ckp/model_10.pth'))
        model = model.cuda()
        self.mae_pretrain = model.image_mae_for_pretraining
        self.frame_linear = nn.Linear(512, 768)
        self.audio_mae = model.audio_mae
        self.audio_linear = nn.Linear(512, 768)

        self.config = config
        self.ans_linear = nn.Linear(768, self.config.model.answer_types)
        self.loss = nn.CrossEntropyLoss()
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        self.vilt = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

    def forward(self, text_input_ids, text_attention_mask, labels, audio_feats, frame_feats):
        AB, AN, AC, AH, AW = audio_feats.shape  # [4, 10, 1, 128, 128]
        audio_feats = audio_feats.reshape(AB * AN, AC, AH, AW)  # [40, 1, 128, 128]
        FB, FN, FC, FH, FW = frame_feats.shape  # [4, 10, 3, 224, 224]
        frame_feats = frame_feats.reshape(FB * FN, FC, FH, FW)  # [40, 3, 224, 224]

        frame_dec_feats = self.mae_pretrain(frame_feats).decoder_last_hidden_state  # [40, 197, 512]
        _, _, audio_enc_feats, audio_dec_feats = self.audio_mae(audio_feats, frame_dec_feats, frame_dec_feats, mask_ratio=0)  # [40, 65, 512]

        B, S, H = frame_dec_feats.shape
        frame_dec_feats = frame_dec_feats.reshape(FB, FN, S, H)  # [4, 10, 197, 512]
        frame_feats = self.frame_linear(frame_dec_feats)  # [4, 10, 197, 768]
        B, S, H = audio_dec_feats.shape
        audio_dec_feats = audio_dec_feats.reshape(AB, AN, S, H)  # [4, 10, 65, 512]
        audio_feats = self.audio_linear(audio_dec_feats)  # [4, 10, 65, 768]

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

    def predict(self, text_input_ids, text_attention_mask, labels, audio_feats,
                frame_feats):
        loss, logits = self.forward(text_input_ids, text_attention_mask, labels, audio_feats, frame_feats)
        return logits
