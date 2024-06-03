import torch
from torch import nn
from liptospeech_extern.src.models.resnet import ResNetModel
from liptospeech_extern.src.conformer.encoder import ConformerEncoder
from einops import rearrange

class Visual_front(nn.Module):
    def __init__(self, in_channels=1, conf_layer=8, num_head=8):
        super().__init__()

        self.in_channels = in_channels
        self.frontend = nn.Sequential(
            nn.Conv3d(self.in_channels, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        self.resnet = ResNetModel(
            layers=18,
            output_dim=512,
            pretrained=False,
            large_input=False
        )

        self.dropout = nn.Dropout(0.3)
        self.conformer = Conformer_encoder(conf_layer, num_head)

    def forward(self, x, vid_len):
        #B,C,T,H,W
        T = x.size(2)
        x = self.frontend(x)
        x = self.resnet(x)  # B, T, 512
        x = self.dropout(x)

        mask = self.conformer.generate_mask(vid_len, T).cuda()

        x = self.conformer(x, mask=mask)
        return x

class Conformer_encoder(nn.Module):
    def __init__(self, num_layers=8, num_attention_heads=8):
        super().__init__()

        self.encoder = ConformerEncoder(encoder_dim=512, num_layers=num_layers, num_attention_heads=num_attention_heads, feed_forward_expansion_factor=4, conv_expansion_factor=2)

    def forward(self, x, mask):
        #x:B,T,C
        out = self.encoder(x, mask=mask)
        return out

    def generate_mask(self, length, sz):
        masks = []
        for i in range(length.size(0)):
            mask = [0] * length[i]
            mask += [1] * (sz - length[i])
            masks += [torch.tensor(mask)]
        masks = torch.stack(masks, dim=0).bool()
        return masks

class CTC_classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # B, S, 512
        size = x.size()
        x = x.view(-1, size[2]).contiguous()
        x = self.classifier(x)
        return x.view(size[0], size[1], -1)

class Speaker_embed(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(nn.Conv1d(80, 128, 7, padding=3),
                                        nn.BatchNorm1d(128),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv1d(128, 256, 7, padding=3),
                                        nn.BatchNorm1d(256),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv1d(256, 256, 7, padding=3),
                                        nn.BatchNorm1d(256),
                                        nn.LeakyReLU(0.2)
                                        )

        self.linear = nn.Linear(256, 512)

    def forward(self, x):
        # B, 1, 80, 100
        x = self.classifier(x.squeeze(1))
        x = x.mean(2)
        x = self.linear(x)
        return x    # B, 512


def get_shape(tensor):
    shapes = []
    sub_tensor = tensor
    while True:
        try:
            shapes.append(str(len(sub_tensor)))
            sub_tensor = sub_tensor[0]
        except Exception as ex:
            break
    return f"({', '.join(shapes)})"

class Mel_classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.fusion = nn.Sequential(nn.Linear(1024, 512),
                                    nn.ReLU())

        self.classifier = nn.Sequential(nn.Conv1d(512, 256, 7, 1, 3),
                                        LayerNorm(256),
                                        nn.ReLU(),
                                        nn.Conv1d(256, 128, 7, 1, 3),
                                        LayerNorm(128),
                                        nn.ReLU(),
                                        nn.Conv1d(128, 320, 7, 1, 3),
                                        )

    def forward(self, x, sp):
        sp = sp.unsqueeze(1).repeat(1, x.size(1), 1)
        # print(get_shape(x))
        x = torch.cat([x, sp], 2)
        # print(get_shape(x))
        x = self.fusion(x)  #B, T, 512
        # print(get_shape(x))
        x = x.permute(0, 2, 1).contiguous()     #B, 512, T
        # print(get_shape(x))
        x = self.classifier(x)
        # print(get_shape(x))
        # B, 320, S
        x = rearrange(x, 'b (d c f) t -> b d c (t f)', d=1, f=4)
        # print(get_shape(x))
        return x

class MatchaMel_classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.fusion = nn.Sequential(nn.Linear(1024, 512),
                                    nn.ReLU())

        self.classifier = nn.Sequential(nn.Conv1d(512, 256, 7, 1, 3),
                                        nn.ReLU(),
                                        nn.Conv1d(256, 128, 7, 1, 3),
                                        nn.ReLU(),
                                        nn.Conv1d(128, 320, 7, 1, 3),
                                        )

    def forward(self, x, sp):
        sp = sp.unsqueeze(1).repeat(1, x.size(1), 1)
        
        
        
        # print(get_shape(x))
        x = torch.cat([x, sp], 2)
        # print(get_shape(x))
        x = self.fusion(x)  #B, T, 512
        # print(get_shape(x))
        x = x.permute(0, 2, 1).contiguous()     #B, 512, T
        # print(get_shape(x))
        x = self.classifier(x)
        # print(get_shape(x))
        # B, 320, S
        x = rearrange(x, 'b (d c f) t -> b d c (t f)', d=1, f=4)
        # print(get_shape(x))

        # ALSO return a mu
        return x


# TODO: Clean away into a separate module in a neat way, or remove when we have CFM layer created
class LayerNorm(nn.Module):
    '''
    Layer normalzation module from Macha-TTS
    '''
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


## https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/text_encoder.py#L328
# class TextEncoder(nn.Module):
#     def __init__(
#         self,
#         encoder_type,
#         encoder_params,
#         duration_predictor_params,
#         n_vocab,
#         n_spks=1,
#         spk_emb_dim=128,
#     ):
#         super().__init__()
#         self.encoder_type = encoder_type
#         self.n_vocab = n_vocab
#         self.n_feats = encoder_params.n_feats
#         self.n_channels = encoder_params.n_channels
#         self.spk_emb_dim = spk_emb_dim
#         self.n_spks = n_spks

#         self.emb = torch.nn.Embedding(n_vocab, self.n_channels)
#         torch.nn.init.normal_(self.emb.weight, 0.0, self.n_channels**-0.5)

#         if encoder_params.prenet:
#             self.prenet = ConvReluNorm(
#                 self.n_channels,
#                 self.n_channels,
#                 self.n_channels,
#                 kernel_size=5,
#                 n_layers=3,
#                 p_dropout=0.5,
#             )
#         else:
#             self.prenet = lambda x, x_mask: x

#         self.encoder = Encoder(
#             encoder_params.n_channels + (spk_emb_dim if n_spks > 1 else 0),
#             encoder_params.filter_channels,
#             encoder_params.n_heads,
#             encoder_params.n_layers,
#             encoder_params.kernel_size,
#             encoder_params.p_dropout,
#         )

#         self.proj_m = torch.nn.Conv1d(self.n_channels + (spk_emb_dim if n_spks > 1 else 0), self.n_feats, 1)
#         self.proj_w = DurationPredictor(
#             self.n_channels + (spk_emb_dim if n_spks > 1 else 0),
#             duration_predictor_params.filter_channels_dp,
#             duration_predictor_params.kernel_size,
#             duration_predictor_params.p_dropout,
#         )

#     def forward(self, x, x_lengths, spks=None):
#         """Run forward pass to the transformer based encoder and duration predictor

#         Args:
#             x (torch.Tensor): text input
#                 shape: (batch_size, max_text_length)
#             x_lengths (torch.Tensor): text input lengths
#                 shape: (batch_size,)
#             spks (torch.Tensor, optional): speaker ids. Defaults to None.
#                 shape: (batch_size,)

#         Returns:
#             mu (torch.Tensor): average output of the encoder
#                 shape: (batch_size, n_feats, max_text_length)
#             logw (torch.Tensor): log duration predicted by the duration predictor
#                 shape: (batch_size, 1, max_text_length)
#             x_mask (torch.Tensor): mask for the text input
#                 shape: (batch_size, 1, max_text_length)
#         """
#         x = self.emb(x) * math.sqrt(self.n_channels)
#         x = torch.transpose(x, 1, -1)
#         x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

#         x = self.prenet(x, x_mask)
#         if self.n_spks > 1:
#             x = torch.cat([x, spks.unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1)
#         x = self.encoder(x, x_mask)
#         mu = self.proj_m(x) * x_mask

#         x_dp = torch.detach(x)
#         logw = self.proj_w(x_dp, x_mask)

#         return mu, logw, x_mask
