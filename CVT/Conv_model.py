# coding: utf-8
"""
Module to represents whole models
"""

import torch.nn as nn
from torch import Tensor
import torch

from BiLSTM import BiLSTMLayer
from discriminator_Data import DataClassifierLayers
from helpers import subsequent_mask
from initialization import initialize_model
from embeddings import Embeddings
from pre_encoders import Encoder, TransformerEncoder
from decoders import Decoder
from vocabulary import Vocabulary
import torch.nn.functional as F


class Model(nn.Module):
    """
    Base Model class
    """
    src_length: int

    def __init__(self,
                 src_length: int,
                 trg_length: int,
                 src_encoder: Encoder,
                 trg_encoder: Encoder,
                 decoder: Decoder,
                 src_embed: Embeddings,
                 trg_embed: Embeddings,
                 latent_embed: Embeddings,
                 cfg: dict,
                 out_trg_size: int,
                 ) -> None:
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super(Model, self).__init__()

        model_cfg = cfg["model"]
        self.src_embed = src_embed
        self.src_length = src_length
        self.src_encoder = src_encoder
        self.embed = latent_embed
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(src_length, trg_length, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(trg_length),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, ceil_mode=False),
            nn.Conv1d(trg_length, trg_length, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(trg_length),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, ceil_mode=False)
        )

        self.linear1 = nn.Linear(125, 512)
        #self.audio_linear = nn.Linear(125, 512)
        self.cat_linear = nn.Linear(1024, 512)
        # self.linear1 = nn.Linear(510, 512)
        self.mid = (out_trg_size - 1) // 10 + 1
        #self.mid = 151
        self.linear2 = nn.Linear(512, self.mid)

        # VAE
        self.trg_embed = trg_embed

        self.encoder = trg_encoder
        self.decoder = decoder
        self.layer_norm = nn.LayerNorm(512, eps=1e-6)
        self.output_layer = nn.Linear(512, out_trg_size, bias=False)

    # pylint: disable=arguments-differ
    def forward(self, src: Tensor):


        src = self.padding_tensor(self.src_embed(src), self.src_length)
        src_encoder_output = self.temporal_conv(src)

        middle_output = self.linear2(self.linear1(src_encoder_output))

        trg_mask = self.AEmask(middle_output)
        trg_embed = self.embed(middle_output)
        padding_mask = trg_mask
        sub_mask = subsequent_mask(trg_embed.size(1)).type_as(trg_mask)
        trg_encoder_output, _ = self.encoder(trg_embed, sub_mask, padding_mask)

        skel_out = self.AEdecode(trg_encoder_output)

        return skel_out, trg_encoder_output

    def padding_tensor(self, sequences, max_len):
        """
        :param sequences: list of tensors
        :return: padded tensor
        """
        num = len(sequences)

        out_dims = (num, max_len, sequences.shape[-1])
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor[:max_len if length > max_len else length, :]

        return out_tensor

    def AEmask(self, src: Tensor):
        trg_mask = (src != 0.0).unsqueeze(1)
        # This increases the shape of the target mask to be even (16,1,120,120) -
        # adding padding that replicates - so just continues the False's or True's
        pad_amount = src.shape[1] - src.shape[2]
        # Create the target mask the same size as target input
        trg_mask = (F.pad(input=trg_mask.double(), pad=(pad_amount, 0, 0, 0), mode='replicate') == 1.0)
        return trg_mask

    def AEencode(self, src: Tensor, src_mask: Tensor) \
            -> (Tensor, Tensor):

        trg_embed = self.trg_embed(src)
        # Encode an embedded source
        padding_mask = src_mask
        # Create subsequent mask for decoding
        sub_mask = subsequent_mask(
            trg_embed.size(1)).type_as(src_mask)
        encode_output = self.encoder(trg_embed, sub_mask, padding_mask)

        return encode_output

    def AEdecode(self, encoder_output: Tensor, hidden: Tensor = None):
        len = []
        for i in range(encoder_output.size(0)):
            len.append(encoder_output.size(0))
        lgt = torch.as_tensor(len)
        tm_outputs = self.decoder(encoder_output.transpose(0, 1), lgt, enforce_sorted=True, hidden=hidden)
        self.hidden = tm_outputs['hidden']
        skel_out = self.output_layer(self.layer_norm(tm_outputs['predictions'].transpose(0, 1)))
        skel_out = torch.cat((skel_out[:, :, :skel_out.shape[2] //10], skel_out[:, :, -1:]), dim=2)
        #skel_out = torch.cat((skel_out[:, :, :skel_out.shape[2]-1], skel_out[:, :, -1:]), dim=2)
        return skel_out

    def AE(self, trg: Tensor):
        trg_mask = self.AEmask(trg)
        trg_encoder_output, _ = self.AEencode(trg, trg_mask)
        # AE解码
        skel_out = self.AEdecode(trg_encoder_output)
        return skel_out,trg_encoder_output

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
                                    self.decoder, self.src_embed, self.trg_embed)


def build_model(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """

    full_cfg = cfg
    cfg = cfg["model"]

    src_padding_idx = 1
    trg_padding_idx = 0

    # Input target size is the joint vector length plus one for counter
    in_trg_size = cfg["trg_size"] + 1
    # Output target size is the joint vector length plus one for counter
    out_trg_size = cfg["trg_size"] + 1

    just_count_in = cfg.get("just_count_in", False)
    future_prediction = cfg.get("future_prediction", 0)

    #  Just count in limits the in target size to 1
    if just_count_in:
        in_trg_size = 1

    # Future Prediction increases the output target size
    if future_prediction != 0:
        # Times the trg_size (minus counter) by amount of predicted frames, and then add back counter
        out_trg_size = (out_trg_size - 1) * future_prediction + 1

    # 文本嵌入
    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)
    # 音频编码
    #audio_embed = nn.Linear(80, cfg["encoder"]["embeddings"]["embedding_dim"])

    src_encoder = TransformerEncoder(**cfg["encoder"],
                                     emb_size=src_embed.embedding_dim,
                                     emb_dropout=cfg["encoder"]["embeddings"].get("dropout",
                                                                                  cfg["encoder"].get("dropout", 0.)))

    latent_embed = nn.Linear(in_trg_size, cfg["decoder"]["embeddings"]["embedding_dim"])
    # 骨骼点嵌入
    trg_linear = nn.Linear(in_trg_size, cfg["decoder"]["embeddings"]["embedding_dim"])

    ## Encoder -------
    enc_dropout = cfg["encoder"].get("dropout", 0.)  # Dropout
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
           cfg["encoder"]["hidden_size"], \
        "for transformer, emb_size must be hidden_size"

    # Transformer Encoder
    encoder = TransformerEncoder(**cfg["encoder"],
                                 emb_size=src_embed.embedding_dim,
                                 emb_dropout=enc_emb_dropout)

    # Decoder
    decoder = BiLSTMLayer(rnn_type='LSTM', input_size=trg_linear.out_features, hidden_size=trg_linear.out_features,
                          num_layers=2, bidirectional=True)

    # Define the model 30 102 112 697 623
    model = Model(src_length=18,#17
                  src_encoder=src_encoder,
                  trg_length=102,
                  trg_encoder=encoder,
                  decoder=decoder,
                  src_embed=src_embed,
                  trg_embed=trg_linear,
                  latent_embed=latent_embed,
                  cfg=full_cfg,
                  out_trg_size=out_trg_size)
    disc = DataClassifierLayers(trg_size=cfg["trg_size"],
                                pose_time_dim=102,
                                hidden_size=512)
    # Custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model, disc
