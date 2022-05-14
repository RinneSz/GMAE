from typing import Optional

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import random_split


def generate_split(num_samples: int, train_ratio: float, val_ratio: float):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


def get_idx_split(dataset, split, preload_split):
    if split[:4] == 'rand':
        train_ratio = float(split.split(':')[1])
        num_nodes = dataset[0].x.size(0)
        train_size = int(num_nodes * train_ratio)
        indices = torch.randperm(num_nodes)
        return {
            'train': indices[:train_size],
            'val': indices[train_size:2 * train_size],
            'test': indices[2 * train_size:]
        }
    elif split.startswith('wikics'):
        split_idx = int(split.split(':')[1])
        return {
            'train': dataset[0].train_mask[:, split_idx],
            'test': dataset[0].test_mask,
            'val': dataset[0].val_mask[:, split_idx]
        }
    elif split.startswith('cora') or split.startswith('citeseer') or split.startswith('pubmed'):
        return {
            'train': dataset.data.train_mask,
            'test': dataset.data.test_mask,
            'val': dataset.data.val_mask
        }
    elif split == 'preloaded':
        assert preload_split is not None, 'use preloaded split, but preloaded_split is None'
        train_mask, test_mask, val_mask = preload_split
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }
    else:
        raise RuntimeError(f'Unknown split type {split}')


def log_regression(x,
                   y,
                   dataset,
                   evaluator,
                   num_epochs: int = 5000,
                   test_device: Optional[str] = None,
                   split: str = 'rand:0.1',
                   verbose: bool = False,
                   preload_split=None):
    torch.set_grad_enabled(True)
    if torch.cuda.is_available():
        test_device = 'cuda'
    else:
        test_device = x.device if test_device is None else test_device
    x = x.detach().to(test_device)
    num_hidden = x.size(1)
    y = y.view(-1).to(test_device)
    num_classes = y.max().item() + 1
    classifier = LogReg(num_hidden, num_classes).to(test_device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)

    split = get_idx_split(dataset, split, preload_split)
    split = {k: v.to(test_device) for k, v in split.items()}
    f = nn.LogSoftmax(dim=-1)
    nll_loss = nn.NLLLoss()

    best_test_acc = 0
    best_val_acc = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()

        output = classifier(x[split['train']])
        loss = nll_loss(f(output), y[split['train']])

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            if 'val' in split:
                # val split is available
                test_acc = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(x[split['test']]).argmax(-1).view(-1, 1)
                })['acc']
                val_acc = evaluator.eval({
                    'y_true': y[split['val']].view(-1, 1),
                    'y_pred': classifier(x[split['val']]).argmax(-1).view(-1, 1)
                })['acc']
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_epoch = epoch
            else:
                acc = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(x[split['test']]).argmax(-1).view(-1, 1)
                })['acc']
                if best_test_acc < acc:
                    best_test_acc = acc
                    best_epoch = epoch
            if verbose:
                print(f'logreg epoch {epoch}: best test acc {best_test_acc}')

    return {'acc': best_test_acc}


class MulticlassEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _eval(y_true, y_pred):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        total = y_true.size(0)
        correct = (y_true == y_pred).to(torch.float32).sum()
        return (correct / total).item()

    def eval(self, res):
        return {'acc': self._eval(**res)}


def evaluate(model, dm, split):
    dm.setup()
    val_loader = dm.val_dataloader()
    dataset_name = dm.dataset_name
    dataset = dm.dataset
    model.eval()
    x = []
    y = []
    with torch.no_grad():
        for batched_data in val_loader:
            emb = model.generate_pretrain_embeddings_for_downstream_task(batched_data)
            x.append(emb)
            label = batched_data.y.reshape(-1, dm.n_val_sampler)[:, 0].reshape(-1)
            y.append(label)
        x = torch.cat(x)
        y = torch.cat(y)
    evaluator = MulticlassEvaluator()
    if dataset_name == 'WikiCS':
        accs = []
        for i in range(20):
            acc = log_regression(x, y, dataset, evaluator, split=f'wikics:{i}', num_epochs=800)['acc']
            accs.append(acc)
        acc = sum(accs) / len(accs)
    elif dataset_name == 'Cora':
        acc = log_regression(x, y, dataset, evaluator, split='cora', num_epochs=3000)['acc']
    elif dataset_name == 'CiteSeer':
        acc = log_regression(x, y, dataset, evaluator, split='citeseer', num_epochs=3000)['acc']
    elif dataset_name == 'PubMed':
        acc = log_regression(x, y, dataset, evaluator, split='pubmed', num_epochs=3000)['acc']
    else:
        acc = \
        log_regression(x, y, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)[
            'acc']
    return acc
