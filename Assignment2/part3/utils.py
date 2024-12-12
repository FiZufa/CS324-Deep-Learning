import torch


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count

    def __str__(self):
        fmtstr = '{name}: {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


@torch.no_grad()
def accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    return torch.mean((preds == labels).type(torch.float))

# def accuracy(output, target):
#     # Implementation here ...
#     pred = output.argmax(dim=1, keepdim=True) # (batch_sizex1)
#     correct = pred.eq(target.view_as(pred)).sum().item()

#     return 100 * correct / output.size(0)
