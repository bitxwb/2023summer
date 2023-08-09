from torch.utils.data import Dataset, DataLoader


class DealDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        sample = self.data[index]
        image = sample.reshape(3, 32, 32)
        sample_label = self.label[index]
        return image, sample_label

    def get_targets(self):
        return self.label.tolist()
