# coding: utf-8
"""
Module to represents whole models
"""

import numpy as np
import torch.nn as nn
from torch import Tensor
import torch

from BiLSTM import BiLSTMLayer
from helpers import subsequent_mask
from initialization import latent_initialize_model
from embeddings import Embeddings
from encoders import Encoder, TransformerEncoder
from decoders import Decoder
from constants import TARGET_PAD
from vocabulary import Vocabulary
from CVT.batch import Batch


class Model(nn.Module):
    """
    Base Model class
    """

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: Embeddings,
                 trg_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary,
                 cfg: dict,
                 in_trg_size: int,
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
        self.trg_embed = trg_embed

        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        # self.bos_index = self.src_vocab.stoi[BOS_TOKEN]
        # self.pad_index = self.src_vocab.stoi[PAD_TOKEN]
        # self.eos_index = self.src_vocab.stoi[EOS_TOKEN]
        self.target_pad = TARGET_PAD

        self.use_cuda = cfg["training"]["use_cuda"]
        self.hidden_size = cfg["model"]["decoder"]["hidden_size"]
        self.in_trg_size = in_trg_size
        self.out_trg_size = out_trg_size
        self.count_in = model_cfg.get("count_in", True)
        # Just Counter
        self.just_count_in = model_cfg.get("just_count_in", False)
        # Gaussian Noise
        self.gaussian_noise = model_cfg.get("gaussian_noise", True)
        # Gaussian Noise
        if self.gaussian_noise:
            self.noise_rate = model_cfg.get("noise_rate", 5.0)

        # Future Prediction - predict for this many frames in the future
        self.future_prediction = model_cfg.get("future_prediction", 0)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.output_layer = nn.Linear(self.hidden_size, self.out_trg_size, bias=False)
        self.hidden = None

    # pylint: disable=arguments-differ
    def forward(self,
                trg_input: Tensor,
                trg_mask: Tensor = None) -> (
            Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_lengths: length of source inputs
        :param trg_mask: target mask
        :return: decoder outputs
        """

        # Encode the source sequence
        encoder_output, encoder_hidden = self.encode(src=trg_input,
                                                     src_mask=trg_mask)
        # Decode the target sequence
        skel_out = self.decode(encoder_output=encoder_output)

        gloss_out = None

        return skel_out, encoder_output

    def encode(self, src: Tensor, src_mask: Tensor) \
            -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        trg_embed = self.src_embed(src)
        # Encode an embedded source
        padding_mask = src_mask
        # Create subsequent mask for decoding
        sub_mask = subsequent_mask(
            trg_embed.size(1)).type_as(src_mask)
        encode_output = self.encoder(trg_embed, sub_mask, padding_mask)

        return encode_output

    def decode(self, encoder_output: Tensor, hidden: Tensor = None) \
            -> (Tensor, Tensor, Tensor, Tensor):
        len = []
        for i in range(encoder_output.size(0)):
            len.append(encoder_output.size(0))
        lgt = torch.as_tensor(len)
        tm_outputs = self.decoder(encoder_output.transpose(0, 1), lgt, enforce_sorted=True, hidden=hidden)
        self.hidden = tm_outputs['hidden']
        skel_out = self.output_layer(self.layer_norm(tm_outputs['predictions'].transpose(0, 1)))
        # decoder_output = self.decoder(trg_embed=trg_embed, encoder_output=encoder_output,
        #                               src_mask=src_mask, trg_mask=trg_mask)

        return skel_out

    def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module) \
            -> Tensor:
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """
        # Forward through the batch input
        skel_out, _ = self.forward(trg_input=batch.trg_input, trg_mask=batch.trg_mask)

        # compute batch loss using skel_out and the batch target
        skel_out_new = skel_out
        trg = batch.trg

        # ssim_loss = compute_ssim(trg.unsqueeze(1).unsqueeze(-1),skel_out_new.unsqueeze(1).unsqueeze(-1))
        batch_loss = loss_function(skel_out_new, trg)
        # batch_loss = MSE_loss + 0.02 * ssim_loss
        # If gaussian noise, find the noise for the next epoch
        if self.gaussian_noise:
            # Calculate the difference between prediction and GT, to find STDs of error
            with torch.no_grad():
                noise = skel_out.detach() - batch.trg.detach()

            if self.future_prediction != 0:
                # Cut to only the first frame prediction + add the counter
                noise = noise[:, :, :noise.shape[2] // (self.future_prediction)]

        else:
            noise = None

        # return batch loss = sum over all elements in batch that are not pad
        return batch_loss, noise, skel_out

    def run_batch(self, batch: Batch, max_output_length: int, ) -> (np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param max_output_length: maximum length of hypotheses
        :param beam_size: size of the beam for beam search, if 0 use greedy
        :param beam_alpha: alpha value for beam search
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """
        # First encode the batch, as this can be done in all one go
        encoder_output, encoder_hidden = self.encode(batch.trg, batch.trg_mask)

        skel_out = self.decode(encoder_output=encoder_output)
        # if maximum output length is not globally specified, adapt to src len
        # if max_output_length is None:
        #     max_output_length = int(max(batch.src_lengths.cpu().numpy()) * 1.5)
        #
        # # Then decode the batch separately, as needs to be done iteratively
        # # greedy decoding
        # stacked_output, stacked_attention_scores = greedy(
        #     encoder_output=encoder_output,
        #     src_mask=batch.src_mask,
        #     embed=self.trg_embed,
        #     decoder=self.decoder,
        #     trg_input=batch.trg_input,
        #     model=self)

        return skel_out

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
    # out_trg_size = 127
    just_count_in = cfg.get("just_count_in", False)
    future_prediction = cfg.get("future_prediction", 0)

    #  Just count in limits the in target size to 1
    if just_count_in:
        in_trg_size = 1

    # Future Prediction increases the output target size
    if future_prediction != 0:
        # Times the trg_size (minus counter) by amount of predicted frames, and then add back counter
        out_trg_size = (out_trg_size - 1) * future_prediction + 1

    # Define source embedding
    src_embed = nn.Linear(in_trg_size, cfg["encoder"]["embeddings"]["embedding_dim"])

    # Define target linear
    # Linear layer replaces an embedding layer - as this takes in the joints size as opposed to a token
    trg_linear = nn.Linear(in_trg_size, cfg["decoder"]["embeddings"]["embedding_dim"])

    ## Encoder -------
    enc_dropout = cfg["encoder"].get("dropout", 0.)  # Dropout
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
           cfg["encoder"]["hidden_size"], \
        "for transformer, emb_size must be hidden_size"

    # Transformer Encoder
    encoder = TransformerEncoder(**cfg["encoder"],
                                 emb_size=cfg["encoder"]["embeddings"]["embedding_dim"],
                                 emb_dropout=enc_emb_dropout)

    ## Decoder -------
    dec_dropout = cfg["decoder"].get("dropout", 0.)  # Dropout
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    decoder_trg_trg = cfg["decoder"].get("decoder_trg_trg", True)
    # Transformer Decoder
    # decoder = TransformerDecoder(
    #     **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
    #     emb_size=trg_linear.out_features, emb_dropout=dec_emb_dropout,
    #     trg_size=out_trg_size, decoder_trg_trg_=decoder_trg_trg)
    decoder = BiLSTMLayer(rnn_type='LSTM', input_size=trg_linear.out_features, hidden_size=trg_linear.out_features,
                          num_layers=2, bidirectional=True)

    # Define the model
    model = Model(encoder=encoder,
                  decoder=decoder,
                  src_embed=src_embed,
                  trg_embed=trg_linear,
                  src_vocab=src_vocab,
                  trg_vocab=trg_vocab,
                  cfg=full_cfg,
                  in_trg_size=in_trg_size,
                  out_trg_size=out_trg_size)
    # Custom initialization of model parameters
    latent_initialize_model(model, cfg, trg_padding_idx, trg_padding_idx)

    return model
