import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer, TransformerDecoder
from mmengine.config import Config
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import get_test_pipeline_cfg
from mmcv.transforms import Compose

def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output
    return hook

class DINOEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        config = Config.fromfile('/home/nhongcha/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py')
        model = init_detector(
            config,
            '/project/lt200060-capgen/palm/parasites/cp/pretrained/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth',
            device='cuda'
        )
        self.max_per_img = 50

        model.hooks = {}
        model.bbox_head.reg_branches[-1][3].register_forward_hook(get_activation(f'reg_features', model.hooks))
        model.bbox_head.cls_branches[-1].register_forward_hook(get_activation(f'cls_features', model.hooks))
        self.model = model
        config.test_dataloader.dataset = config.train_dataloader.dataset
        self.transform = Compose(get_test_pipeline_cfg(config))

    def forward(self, images):
        data = {
            'inputs': [],
            'data_samples': []
        }
        for i in range(len(images)):
            data_ = dict(img_path=images[i], img_id=i)
            data_ = self.transform(data_)
            data['inputs'].append(data_['inputs'])
            data['data_samples'].append(data_['data_samples'])

        with torch.no_grad():
            self.model.test_step(data)
        
        reg = self.model.hooks[f'reg_features']
        cls_score = self.model.hooks[f'cls_features']
        # print(f'reg: {reg.size()} - cls_score: {cls_score.size()}')
        scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
        scores, bbox_index = scores.topk(self.max_per_img)
        # print(f'scores: {scores.size()} - bbox_index: {bbox_index.size()}')
        output = torch.gather(reg, 1, bbox_index.unsqueeze(-1).expand(-1, -1, 256))
        return output
        # return reg[bbox_index.unsqueeze(-1).cpu()].cuda()


class ResidualBlock(nn.Module):
    """Represents 1D version of the residual block: https://arxiv.org/abs/1512.03385"""

    def __init__(self, input_dim):
        """Initializes the module."""
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, x):
        """Performs forward pass of the module."""
        skip_connection = x
        x = self.block(x)
        x = skip_connection + x
        return x


class Normalize(nn.Module):
    def __init__(self, eps=1e-5):
        super(Normalize, self).__init__()
        self.register_buffer("eps", torch.Tensor([eps]))

    def forward(self, x, dim=-1):
        norm = x.norm(2, dim=dim).unsqueeze(-1)
        x = self.eps * (x / norm)
        return x


class PositionalEncodings(nn.Module):
    """Attention is All You Need positional encoding layer"""

    def __init__(self, seq_len, d_model, p_dropout):
        """Initializes the layer."""
        super(PositionalEncodings, self).__init__()
        token_positions = torch.arange(start=0, end=seq_len).view(-1, 1)
        dim_positions = torch.arange(start=0, end=d_model).view(1, -1)
        angles = token_positions / (10000 ** ((2 * dim_positions) / d_model))

        encodings = torch.zeros(1, seq_len, d_model)
        encodings[0, :, ::2] = torch.cos(angles[:, ::2])
        encodings[0, :, 1::2] = torch.sin(angles[:, 1::2])
        encodings.requires_grad = False
        self.register_buffer("positional_encodings", encodings)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """Performs forward pass of the module."""
        x = x + self.positional_encodings
        x = self.dropout(x)
        return x


class CaptionDecoder(nn.Module):
    """Decoder for image captions.

    Generates prediction for next caption word given the prviously
    generated word and image features extracted from CNN.    
    """

    def __init__(self, config):
        """Initializes the model."""
        super(CaptionDecoder, self).__init__()
        model_config = config["model_configuration"]
        decoder_layers = model_config["decoder_layers"]
        attention_heads = model_config["attention_heads"]
        d_model = model_config["d_model"]
        ff_dim = model_config["ff_dim"]
        dropout = model_config["dropout"]

        embedding_dim = config["embeddings"]["size"]
        vocab_size = config["vocab_size"]
        img_feature_channels = config["image_specs"]["img_feature_channels"]

        # Load pretrained word embeddings
        word_embeddings = torch.Tensor(np.loadtxt(config["embeddings"]["path"]))
        self.embedding_layer = nn.Embedding.from_pretrained(
            word_embeddings,
            freeze=True,
            padding_idx=config["PAD_idx"]
        )

        self.entry_mapping_words = nn.Linear(embedding_dim, d_model)
        self.entry_mapping_img = nn.Linear(img_feature_channels, d_model)

        self.res_block = ResidualBlock(d_model)

        self.positional_encodings = PositionalEncodings(config["max_len"], d_model, dropout)
        transformer_decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=attention_heads,
            dim_feedforward=ff_dim,
            dropout=dropout
        )
        self.decoder = TransformerDecoder(transformer_decoder_layer, decoder_layers)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x, image_features, tgt_padding_mask=None, tgt_mask=None):
        """Performs forward pass of the module."""
        # Adapt the dimensionality of the features for image patches
        image_features = self.entry_mapping_img(image_features)
        image_features = image_features.permute(1, 0, 2)
        image_features = F.leaky_relu(image_features)

        # Entry mapping for word tokens
        x = self.embedding_layer(x)
        x = self.entry_mapping_words(x)
        x = F.leaky_relu(x)

        x = self.res_block(x)
        x = F.leaky_relu(x)

        x = self.positional_encodings(x)

        # Get output from the decoder
        x = x.permute(1, 0, 2)
        x = self.decoder(
            tgt=x,
            memory=image_features,
            tgt_key_padding_mask=tgt_padding_mask,
            tgt_mask=tgt_mask
        )
        x = x.permute(1, 0, 2)

        x = self.classifier(x)
        return x
