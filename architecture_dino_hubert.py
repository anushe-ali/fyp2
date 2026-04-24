# architectures_dino_hubert.py
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

EMBED_DIM = 512
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FaceDINOEncoder(nn.Module):
    """ViT-B/16 pretrained with DINO → 512-d face embedding."""
    def __init__(self, embed_dim=EMBED_DIM, pretrained=True, unfreeze_last_n=4):
        super().__init__()
        self.model      = timm.create_model('vit_base_patch16_224.dino', pretrained=pretrained)
        self.model.head = nn.Identity()

        for param in self.model.parameters():
            param.requires_grad = False
        for block in list(self.model.blocks)[-unfreeze_last_n:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.model.norm.parameters():
            param.requires_grad = True

        self.proj = nn.Sequential(
            nn.Linear(self.model.num_features, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        feats = self.model(x)    # (B, 768)
        return self.proj(feats)  # (B, 512)


class SpecAugment(nn.Module):
    def __init__(self, freq_mask=27, time_mask=100):
        super().__init__()
        self.freq_mask = freq_mask
        self.time_mask = time_mask

    def forward(self, x):  # x: (B, T, D)
        if not self.training:
            return x
        B, T, D = x.shape
        f  = random.randint(0, self.freq_mask)
        f0 = random.randint(0, D - f) if f < D else 0
        x[:, :, f0:f0+f] = 0
        t  = random.randint(0, min(self.time_mask, T-1))
        t0 = random.randint(0, T - t) if t < T else 0
        x[:, t0:t0+t, :] = 0
        return x


class VoiceHuBERTEncoder(nn.Module):
    """facebook/hubert-base-ls960 → mean-pool → 512-d voice embedding."""
    def __init__(self, embed_dim=EMBED_DIM, unfreeze_last_n=4, aug=False):
        super().__init__()
        model_name     = 'facebook/hubert-base-ls960'
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.hubert    = HubertModel.from_pretrained(model_name)

        for param in self.hubert.parameters():
            param.requires_grad = False
        for layer in self.hubert.encoder.layers[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.hubert.feature_projection.parameters():
            param.requires_grad = True

        self.spec_augment = SpecAugment()

        self.proj = nn.Sequential(
            nn.Linear(self.hubert.config.hidden_size, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, audio_batch, aug, device=DEVICE):
        audios = [audio_batch[i].cpu().numpy() for i in range(audio_batch.size(0))]
        inputs = self.processor(audios, sampling_rate=16000,
                                return_tensors='pt', padding=True)
        input_values   = inputs['input_values'].to(device)
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = self.hubert(input_values, attention_mask=attention_mask)
        hidden  = self.spec_augment(outputs.last_hidden_state)  # (B, T, 768)
        pooled  = hidden.mean(dim=1)                             # (B, 768)
        return self.proj(pooled)                                  # (B, 512)