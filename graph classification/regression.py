import torch
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np


def process_raw_embeds(embeds):
    # embeds with shape [n graph, n nodes, n dims]
    graph_embeds = []
    for i in range(embeds.size(0)):
        graph = embeds[i]
        n_nodes, n_dims = graph.size(0), graph.size(1)
        zeros_count = (graph == 0).sum(1)  # [n nodes]
        mask = zeros_count == n_nodes
        orig_graph = graph[~mask]
        graph_embeds.append(torch.mean(orig_graph, dim=0).view(1, -1))
    return torch.cat(graph_embeds)


def svc_classify(model, dataloader,):
    model.eval()
    model.cuda()
    x = []
    y = []
    for i, data in enumerate(dataloader):
        labels = data.y.cuda()
        y.append(labels)
        with torch.no_grad():
            pretrain_embeddings = model.generate_pretrain_embeddings_for_downstream_task(data)
            pretrain_embeddings = process_raw_embeds(pretrain_embeddings)
            x.append(pretrain_embeddings)
    y = torch.cat(y).cpu().detach().numpy()
    x = torch.cat(x).cpu().detach().numpy()
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):
        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
        classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)

        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)

        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    return np.mean(accuracies_val), np.mean(accuracies)
