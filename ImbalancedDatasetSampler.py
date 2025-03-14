import torch
import torch.utils.data
import torchvision


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        # self.weights = torch.FloatTensor(weights)


    # def _get_label(self, dataset, idx):
    #     if self.callback_get_label:
    #         return self.callback_get_label(dataset, idx)
    #     elif isinstance(dataset, torchvision.datasets.MNIST):
    #         return dataset.targets[idx].item()
    #     elif isinstance(dataset, torchvision.datasets.CIFAR10):
    #         return dataset.targets[idx]
    #     elif isinstance(dataset, torchvision.datasets.ImageFolder):
    #         return dataset.imgs[idx][1]
    #     elif isinstance(dataset, torch.utils.data.Subset):
    #         return dataset.dataset.imgs[idx][1]
    #     else:
    #         raise NotImplementedError

    def _get_label(self, dataset, idx):
        label = dataset.targets[idx]
        if not isinstance(label, int):
            return label.item()
        return label

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
