from torch.utils.data import Dataset
import torch
# from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def resample_sequence(sequence, target_length):
    original_length = sequence.size(0)
    if original_length == target_length:
        return sequence


    indices = torch.linspace(0, original_length - 1, target_length).long()


    resampled_sequence = sequence[indices, :]

    return resampled_sequence
def collate_fn(batch):
    token2idx = batch[0][3]
    max_length = 100
    # max_length = 112
    left_pad = 6
    new_trg = []
    new_name = []
    new_trg_length = []

    #max_audio = 623
    #left_audio = 37
    #new_audio = []

    texts = [batch[idx][0] for idx in range(len(batch))]
    tokens = [[token2idx[token] for token in text] for text in texts]
    sequences_padded = pad_sequence([torch.tensor(tokens) for tokens in tokens], batch_first=True,
                                    padding_value=token2idx['<pad>'])

    for i in range(len(batch)):

        trg_length = len(batch[i][1])
        new_trg_length.append(trg_length)
        right_pad = max_length - trg_length + 6
        padded_video = torch.cat(
            (
                batch[i][1][0][None].expand(left_pad, -1),
                torch.stack(batch[i][1]),
                batch[i][1][-1][None].expand(right_pad, -1),
            )
            , dim=0)
        new_trg.append(padded_video)
        new_name.append(batch[i][2])


    # for i in range(len(batch)):

    #     trg_length = len(batch[i][1])
    #     new_trg_length.append(trg_length)
    #     padded_video = resample_sequence(torch.stack(batch[i][1]), max_length)
    #     new_trg.append(padded_video)
    #     new_name.append(batch[i][2])
    #

    #     audio = torch.as_tensor(batch[i][4])
    #     padded_audio = resample_sequence(audio, max_audio)
    #     new_audio.append(padded_audio)

    # for i in range(len(batch)):

    #     trg_length = len(batch[i][1])
    #     new_trg_length.append(trg_length)
    #     right_pad = max_length - trg_length + 6
    #     trg_p = torch.zeros_like(batch[i][1][0][None])
    #     padded_video = torch.cat(
    #         (
    #             trg_p.expand(left_pad, -1),
    #             torch.stack(batch[i][1]),
    #             trg_p.expand(right_pad, -1),
    #         )
    #         , dim=0)
    #     new_trg.append(padded_video)
    #     new_name.append(batch[i][2])
    #

    #     audio_length = len(batch[i][4])
    #     right_audio = max_audio - audio_length + 37
    #     audio = torch.as_tensor(batch[i][4])
    #     audio_p = torch.zeros_like(audio[0][None])
    #     padded_audio = torch.cat(
    #         (
    #             audio_p.expand(left_audio, -1),
    #             audio,
    #             audio_p.expand(right_audio, -1),
    #         )
    #         , dim=0)
    #     new_audio.append(padded_audio)

    new_padded_video = torch.stack(new_trg)

    # name = [batch[idx][2] for idx in batch]  new_trg_length
    return sequences_padded, new_padded_video, new_name, new_trg_length


class make_data_iter(Dataset):

    def __init__(self, dataset, vocab):
        self.datasets = dataset
        self.vocabs = vocab

    def __getitem__(self, item):
        return self.datasets[item].src, self.datasets[item].trg, self.datasets[item].file_paths, self.vocabs

    def __len__(self):
        return len(self.datasets)


# if __name__ == "__main__":
#     train_data = DataSet(datamode="dev")
#     train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
