import torch
from torchvision.transforms.functional import resize, pad

def torch_ix_(*args):
    out = []
    nd = len(args)
    for k, new in enumerate(args):
        if not isinstance(new, torch.Tensor):
            new = torch.tensor(new)
            if new.shape[0] == 0:
                new = new.type_as(torch.int64)
        if new.ndim != 1:
            raise ValueError("Cross index must be 1 dimensional")
        if new.dtype is torch.bool:
            new, = new.nonzero().T
        new = new.reshape((1,) * k + (tuple(new.shape, )) + (1,) * (nd - k - 1))
        out.append(new)
    return tuple(out)


class CropBlackPixelsAndResize(object):
    def __init__(self, tol=0.1, normalized=True, output_size=14):
        self.tol = tol
        self.output_size = output_size
        if not normalized:
            self.tol *= 255

    def __call__(self, img):
        mask = img[0] > self.tol
        cropped_image = img[0][torch_ix_(mask.any(1), mask.any(0))]
        cropped_image = cropped_image.reshape((1, cropped_image.shape[0], cropped_image.shape[1]))
        return resize(cropped_image, [self.output_size, self.output_size])







