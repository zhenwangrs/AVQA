import sys

import open_clip
from open_clip.tokenizer import tokenize
import torch
import torch.nn as nn
from transformers import RobertaModel, AutoImageProcessor, RobertaTokenizer, CLIPTextModel, AutoTokenizer, CLIPModel

sys.path.append('../../siammae_pretrain')
from models.siammae_pretrain.clip.models_siammae_clip import SiamMAE


class PairwiseCrossAttention(nn.Module):
    def __init__(self, input_dim):
        super(PairwiseCrossAttention, self).__init__()
        self.cma = nn.MultiheadAttention(input_dim, 8, batch_first=True)

    def forward(self, audio_feature, vision_feature):
        batch_size, seq_len_audio, _, audio_dim = audio_feature.shape
        batch_size, seq_len_vision, _, vision_dim = vision_feature.shape

        audio_feature = audio_feature.view(batch_size * seq_len_audio, -1, audio_dim)
        vision_feature = vision_feature.view(batch_size * seq_len_vision, -1, vision_dim)

        attended_values = self.cma(audio_feature, vision_feature, vision_feature)[0]
        attended_values = attended_values.view(batch_size, seq_len_audio, -1, vision_dim)

        return attended_values


class CMA_LSTM_Block(nn.Module):
    def __init__(self, hidden_size=768):
        super(CMA_LSTM_Block, self).__init__()
        self.text_av_cma = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.FFN = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, hidden_size)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.text_self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.self_FFN = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, hidden_size)
        )
        self.layer_norm4 = nn.LayerNorm(hidden_size)

    def forward(self, text_feats, av_feats, text_attn_mask=None):
        output = self.text_av_cma(text_feats, av_feats, av_feats, attn_mask=text_attn_mask)[0]
        output = self.layer_norm1(output + text_feats)
        output = self.layer_norm2(output + self.FFN(output))

        output = self.text_self_attn(output, output, output, attn_mask=text_attn_mask)[0]
        output = self.layer_norm3(output + text_feats)
        output = self.layer_norm4(output + self.self_FFN(output))
        return output


class CMA_LSTM(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(CMA_LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.blocks = nn.ModuleList([CMA_LSTM_Block(hidden_size) for _ in range(num_layers)])

    def forward(self, text_feats, av_feats):
        for block in self.blocks:
            text_feats = block(text_feats, av_feats)
        return text_feats


class LSTM_AVQA_Model(nn.Module):
    def __init__(self, config):
        super(LSTM_AVQA_Model, self).__init__()
        self.config = config
        model = SiamMAE(config)
        model.load_state_dict(torch.load('E:/Research/AVQA/models/siammae_pretrain/clip/ckp/model_1.pth'))
        self.vit = model.image_mae_encoder
        self.audio_mae = model.audio_mae

        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.visual_proj = clip.visual_projection
        self.text_encoder = clip.text_model
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.hidden_size = 512
        self.av_fusion = PairwiseCrossAttention(input_dim=self.hidden_size)
        self.lstm_text_audio = CMA_LSTM(num_layers=self.config.model.lstm_layers, hidden_size=self.hidden_size)
        self.lstm_text_frame = CMA_LSTM(num_layers=self.config.model.lstm_layers, hidden_size=self.hidden_size)
        self.lstm_text_av = CMA_LSTM(num_layers=self.config.model.lstm_layers, hidden_size=self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size * 3, 42)

        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, text_input_ids, text_attention_mask, labels, audio_feats, frame_feats):
        AB, AN, AC, AH, AW = audio_feats.shape  # [4, 10, 1, 128, 128]
        audio_feats = audio_feats.reshape(AB * AN, AC, AH, AW)  # [40, 1, 128, 128]
        FB, FN, FC, FH, FW = frame_feats.shape  # [4, 10, 3, 224, 224]
        frame_feats = frame_feats.reshape(FB * FN, FC, FH, FW)  # [40, 3, 224, 224]

        audio_feats = self.audio_mae.forward_encoder_no_mask(audio_feats)  # [40, 65, 768]
        audio_feats = self.visual_proj(audio_feats)  # [40, 65, 512]
        B, S, H = audio_feats.shape
        audio_feats = audio_feats.reshape(AB, AN, S, H)  # [4, 10, 65, 768]
        frame_feats = self.vit(frame_feats, apply_mask=False).last_hidden_state  # [40, 197, 768]
        frame_feats = self.visual_proj(frame_feats)  # [40, 197, 512]
        B, S, H = frame_feats.shape
        frame_feats = frame_feats.reshape(FB, FN, S, H)  # [4, 10, 197, 768]

        av_feats = self.av_fusion(audio_feats, frame_feats)

        text_feats = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask).last_hidden_state
        for step in range(AN):
            text_audio_feats = self.lstm_text_audio(text_feats, audio_feats[:, step, :, :])
            text_frame_feats = self.lstm_text_frame(text_feats, frame_feats[:, step, :, :])
            text_av_feats = self.lstm_text_av(text_feats, av_feats[:, step, :, :])

        text_feats = torch.cat([torch.mean(text_audio_feats, dim=1),
                                torch.mean(text_av_feats, dim=1),
                                torch.mean(text_frame_feats, dim=1)], dim=1)
        logits = self.classifier(text_feats)
        loss = self.loss_fn(logits, labels)
        return loss, logits

    def predict(self, text_input_ids, text_attention_mask, labels, audio_feats, frame_feats):
        _, logits = self.forward(text_input_ids, text_attention_mask, labels, audio_feats, frame_feats)
        logits = torch.softmax(logits, dim=1)
        return logits
