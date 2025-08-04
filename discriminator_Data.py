import torch
from torch import nn, Tensor

from embeddings import Embeddings
from transformer_layers import TransformerEncoderLayer, PositionalEncoding, PositionwiseFeedForward, FusionLayer


class Classifier(nn.Module):
    @property
    def output_size(self):
        """
        Return the output size

        :return:
        """
        return self._output_size


class DataClassifierLayers(Classifier):

    # pylint: disable=unused-argument
    def __init__(self,
                 trg_size: int = 150,
                 pose_time_dim: int = 125,
                 hidden_size: int = 512):
        super(DataClassifierLayers, self).__init__()
        embedding_dim = hidden_size  # emb_dim

        self.trg_embed = nn.Linear(trg_size, embedding_dim)
        self.pe = PositionalEncoding(embedding_dim, mask_count=True)
        size = hidden_size
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=pose_time_dim, out_channels=pose_time_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(pose_time_dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=pose_time_dim // 2, out_channels=pose_time_dim // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(pose_time_dim // 4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(pose_time_dim // 4 * 32, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Linear(embedding_dim, 1)

        self.softmax = torch.nn.LogSoftmax(dim=-1)  # dim=1)

    # pylint: disable=arguments-differ
    def forward(self, pose: Tensor) -> Tensor:
        pose_embed = self.trg_embed(pose)
        y = self.pe(pose_embed)

        # Add Dropout
        y = self.conv_layers(y)

        ## MLP for classification
        y = self.linear_layers(y.view(y.shape[0], -1))

        out = self.output_layer(y)  ## Bs, 1

        return out  # .squeeze(1)
