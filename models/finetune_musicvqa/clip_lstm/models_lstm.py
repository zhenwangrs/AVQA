import sys

import open_clip
from open_clip.tokenizer import tokenize
import torch
import torch.nn as nn
from transformers import RobertaModel, AutoImageProcessor, RobertaTokenizer, CLIPTextModel, AutoTokenizer, CLIPModel

sys.path.append('../../siammae_pretrain')
from models.siammae_pretrain.clip.models_siammae_clip import SiamMAE


class PairwiseCrossAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.0):
        super(PairwiseCrossAttention, self).__init__()
        self.cma = nn.MultiheadAttention(hidden_size, 8, batch_first=True, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.FFN = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, audio_feature, vision_feature):
        batch_size, seq_len_audio, _, audio_dim = audio_feature.shape
        batch_size, seq_len_vision, _, vision_dim = vision_feature.shape

        audio_feature = audio_feature.view(batch_size * seq_len_audio, -1, audio_dim)
        vision_feature = vision_feature.view(batch_size * seq_len_vision, -1, vision_dim)

        attended_values = self.cma(audio_feature, vision_feature, vision_feature)[0]
        attended_values = self.layer_norm1(attended_values)
        attended_values = self.FFN(attended_values)
        attended_values = self.layer_norm2(attended_values)

        attended_values = attended_values.view(batch_size, seq_len_audio, -1, vision_dim)
        return attended_values


class CMA_LSTM_Block(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(CMA_LSTM_Block, self).__init__()
        self.text_av_cma = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.FFN = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, text_feats, av_feats, text_attn_mask=None):
        output = self.text_av_cma(text_feats, av_feats, av_feats, attn_mask=text_attn_mask)[0]
        output = self.layer_norm1(output + text_feats)
        output = self.layer_norm2(output + self.FFN(output))
        return output


class CMA_LSTM(nn.Module):
    def __init__(self, num_layers, hidden_size, dropout):
        super(CMA_LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.text_av_cma = CMA_LSTM_Block(hidden_size, dropout)
        self.hidden_sa = CMA_LSTM_Block(hidden_size, dropout)
        self.hidden_av_cma = CMA_LSTM_Block(hidden_size, dropout)
        self.text_av_last_cma = CMA_LSTM_Block(hidden_size, dropout)

    def forward(self, text_feats, av_feats):
        seq_len = av_feats.shape[1]
        all_hidden_feats = []
        all_tav_hidden_feats = []
        hidden_feats = text_feats
        for step in range(seq_len):
            av_frame_feat = av_feats[:, step, :, :]
            layer1_feat = self.text_av_cma(text_feats, av_frame_feat)
            layer2_feat = self.hidden_sa(layer1_feat, layer1_feat)
            hidden_feats = self.hidden_av_cma(hidden_feats, layer2_feat)
            all_hidden_feats.append(hidden_feats)
            all_tav_hidden_feats.append(layer2_feat)
        hidden_feats = torch.cat(all_hidden_feats, dim=1)
        hidden_feats = self.text_av_last_cma(text_feats, hidden_feats)
        all_tav_hidden_feats = torch.stack(all_tav_hidden_feats, dim=1)
        return hidden_feats, all_tav_hidden_feats


class LSTM_AVQA_Model(nn.Module):
    def __init__(self, config):
        super(LSTM_AVQA_Model, self).__init__()

        self.config = config
        model = SiamMAE(config)
        model.load_state_dict(torch.load(config.model.pretrained_siammae_path))

        self.image_mae = model.image_mae_encoder
        self.set_if_finetune(self.image_mae, config.model.finetune.image_mae)
        self.image_mae_proj = model.image_mae_proj
        self.set_if_finetune(self.image_mae_proj, config.model.finetune.image_mae_proj)

        self.audio_mae = model.audio_mae
        self.set_if_finetune(self.audio_mae, config.model.finetune.audio_mae)
        self.audio_mae_proj = model.audio_mae_proj
        self.set_if_finetune(self.audio_mae_proj, config.model.finetune.audio_mae_proj)

        clip = model.clip
        self.clip_text = clip.text_model
        self.set_if_finetune(self.clip_text, config.model.finetune.clip_text)
        self.clip_text_proj = model.clip_text_proj
        self.set_if_finetune(self.clip_text_proj, config.model.finetune.clip_text_proj)

        self.hidden_size = 512
        self.av_fusion = PairwiseCrossAttention(hidden_size=self.hidden_size)
        self.lstm_text_audio = CMA_LSTM(
            num_layers=self.config.model.lstm_layers,
            hidden_size=self.hidden_size,
            dropout=self.config.model.dropout
        )
        self.lstm_text_frame = CMA_LSTM(
            num_layers=self.config.model.lstm_layers,
            hidden_size=self.hidden_size,
            dropout=self.config.model.dropout
        )
        self.lstm_text_av = CMA_LSTM(
            num_layers=self.config.model.lstm_layers,
            hidden_size=self.hidden_size,
            dropout=self.config.model.dropout
        )
        self.dropout = nn.Dropout(self.config.model.dropout)
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
        audio_feats = self.audio_mae_proj(audio_feats)  # [40, 65, 512]
        B, S, H = audio_feats.shape
        audio_feats = audio_feats.reshape(AB, AN, S, H)  # [4, 10, 65, 512]

        frame_feats = self.image_mae(frame_feats, apply_mask=False).last_hidden_state  # [40, 197, 768]
        frame_feats = self.image_mae_proj(frame_feats)  # [40, 197, 512]
        B, S, H = frame_feats.shape
        frame_feats = frame_feats.reshape(FB, FN, S, H)  # [4, 10, 197, 512]

        text_feats = self.clip_text(input_ids=text_input_ids, attention_mask=text_attention_mask).last_hidden_state
        text_feats = self.clip_text_proj(text_feats)

        text_audio_feats, all_ta_hidden_feats = self.lstm_text_audio(text_feats, audio_feats)
        text_frame_feats, all_tf_hidden_feats = self.lstm_text_frame(text_feats, frame_feats)

        # av_feats = self.av_fusion(audio_feats, frame_feats)
        # av_feats = self.av_fusion(frame_feats, audio_feats)
        av_feats = self.av_fusion(all_tf_hidden_feats, all_ta_hidden_feats)
        text_av_feats, _ = self.lstm_text_av(text_feats, av_feats)

        text_feats = torch.cat([torch.mean(text_audio_feats, dim=1),
                                torch.mean(text_av_feats, dim=1),
                                torch.mean(text_frame_feats, dim=1)], dim=1)
        text_feats = self.dropout(text_feats)
        logits = self.classifier(text_feats)
        loss = self.loss_fn(logits, labels)
        return loss, logits

    def predict(self, text_input_ids, text_attention_mask, labels, audio_feats, frame_feats):
        _, logits = self.forward(text_input_ids, text_attention_mask, labels, audio_feats, frame_feats)
        logits = torch.softmax(logits, dim=1)
        return logits

    def set_if_finetune(self, model, if_finetune):
        for param in model.parameters():
            param.requires_grad = if_finetune
