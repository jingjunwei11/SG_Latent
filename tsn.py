import json
import os.path

import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class read_Data(Dataset):

    def __init__(self, predicted_data, reconst_data):
        self.predicted_data = predicted_data
        self.reconst_data = reconst_data

    def __getitem__(self, item):
        return torch.tensor(self.predicted_data[item]), torch.tensor(self.reconst_data[item])

    def __len__(self):
        return len(self.predicted_data)


def tsne(input_data, name):

    new_shape = (input_data.size(0) * input_data.size(1), input_data.size(2))
    flat_data = input_data.reshape(input_data.size(0) * input_data.size(1), input_data.size(2))


    flat_data_np = flat_data.cpu().numpy()


    tsne = TSNE(n_components=2)
    dim_reduced_data = tsne.fit_transform(flat_data_np)


    dim_reduced_data = torch.from_numpy(dim_reduced_data).reshape(input_data.size(0), input_data.size(1), 2) * 1000


    color = ['r', 'b', 'g']
    for i in range(dim_reduced_data.size(0)):
        for frame in dim_reduced_data[i]:
            plt.scatter(frame[0], frame[1], s=1, c=color[i], marker='o', alpha=1)

    plt.grid(True)

    plt.axis('off')
    plt.savefig(os.path.join("images", name + ".png"))
    plt.show()


if __name__ == "__main__":
    with open('predicted_latent.json', 'r') as file:
        predicted_latent = json.load(file)
    with open('reconst_latent.json', 'r') as file:
        reconst_latent = json.load(file)

    data = read_Data(predicted_latent, reconst_latent)
    train_loader = DataLoader(data, batch_size=3, shuffle=True)
    i = 0
    for predicted, reconst in train_loader:
        tsne(predicted[:,:,:], str(i) + "pre")
        tsne(reconst[:,:,:], str(i) + "rec")
        i += 1
