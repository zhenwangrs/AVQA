import torch
from transformers import ViTMAEForPreTraining


def load_encoder_from_pretained_audiomae(model, pretrained_path):
    pretrained_audiomae = torch.load(pretrained_path)
    pretrained_audiomae_state_dict = pretrained_audiomae['model']
    model_state_dict = model.state_dict()
    for key in list(model_state_dict.keys()):
        if not key.startswith('decoder'):
            if model_state_dict[key].shape == pretrained_audiomae_state_dict[key].shape:
                model_state_dict[key] = pretrained_audiomae_state_dict[key]
            else:
                print(key)
    # special case for decoder, since spectrogram only one channel
    model_state_dict['decoder_norm.weight'] = pretrained_audiomae_state_dict['decoder_norm.weight']
    model_state_dict['decoder_norm.bias'] = pretrained_audiomae_state_dict['decoder_norm.bias']
    model_state_dict['decoder_pred.weight'] = pretrained_audiomae_state_dict['decoder_pred.weight']
    model_state_dict['decoder_pred.bias'] = pretrained_audiomae_state_dict['decoder_pred.bias']
    model.load_state_dict(model_state_dict)
    return model


def load_decoder_from_pretrained_vitmae(model):
    pretrained_vitmae = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    pretrained_vitmae_state_dict = pretrained_vitmae.state_dict()
    model_state_dict = model.state_dict()
    key_map = {
        'decoder_mask_token': 'decoder.mask_token',
        'decoder_pos_embed': 'decoder.decoder_pos_embed',
        'decoder_embed.weight': 'decoder.decoder_embed.weight',
        'decoder_embed.bias': 'decoder.decoder_embed.bias',
        # 'decoder_norm.weight': 'decoder.decoder_norm.weight',
        # 'decoder_norm.bias': 'decoder.decoder_norm.bias',
        # 'decoder_pred.weight': 'decoder.decoder_pred.weight',
        # 'decoder_pred.bias': 'decoder.decoder_pred.bias',
    }
    for i in range(0, 8):
        key_map.update({
            f'decoder_blocks.{i}.norm1.weight': f'decoder.decoder_layers.{i}.layernorm_before.weight',
            f'decoder_blocks.{i}.norm1.bias': f'decoder.decoder_layers.{i}.layernorm_before.bias',
            f'decoder_blocks.{i}.norm2.weight': f'decoder.decoder_layers.{i}.layernorm_after.weight',
            f'decoder_blocks.{i}.norm2.bias': f'decoder.decoder_layers.{i}.layernorm_after.bias',
            f'decoder_blocks.{i}.attn.q_linear.weight': f'decoder.decoder_layers.{i}.attention.attention.query.weight',
            f'decoder_blocks.{i}.attn.q_linear.bias': f'decoder.decoder_layers.{i}.attention.attention.query.bias',
            f'decoder_blocks.{i}.attn.k_linear.weight': f'decoder.decoder_layers.{i}.attention.attention.key.weight',
            f'decoder_blocks.{i}.attn.k_linear.bias': f'decoder.decoder_layers.{i}.attention.attention.key.bias',
            f'decoder_blocks.{i}.attn.v_linear.weight': f'decoder.decoder_layers.{i}.attention.attention.value.weight',
            f'decoder_blocks.{i}.attn.v_linear.bias': f'decoder.decoder_layers.{i}.attention.attention.value.bias',
            f'decoder_blocks.{i}.attn.proj.weight': f'decoder.decoder_layers.{i}.attention.output.dense.weight',
            f'decoder_blocks.{i}.attn.proj.bias': f'decoder.decoder_layers.{i}.attention.output.dense.bias',
            f'decoder_blocks.{i}.mlp.fc1.weight': f'decoder.decoder_layers.{i}.intermediate.dense.weight',
            f'decoder_blocks.{i}.mlp.fc1.bias': f'decoder.decoder_layers.{i}.intermediate.dense.bias',
            f'decoder_blocks.{i}.mlp.fc2.weight': f'decoder.decoder_layers.{i}.output.dense.weight',
            f'decoder_blocks.{i}.mlp.fc2.bias': f'decoder.decoder_layers.{i}.output.dense.bias',
        })
    for key in list(model_state_dict.keys()):
        if key.startswith('decoder'):
            if key in key_map.keys() and model_state_dict[key].shape == pretrained_vitmae_state_dict[key_map[key]].shape:
                model_state_dict[key] = pretrained_vitmae_state_dict[key_map[key]]
            else:
                print(key)
    model.load_state_dict(model_state_dict)
    return model
