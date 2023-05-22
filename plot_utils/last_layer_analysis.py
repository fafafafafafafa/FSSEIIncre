import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def last_layer_analysis(heads, task, taskcla, y_lim=False, sort_weights=False):
    """Plot last layer weight and bias analysis"""
    print('Plotting last layer analysis...')
    num_classes = sum([x for (_, x) in taskcla])
    weights, biases, indexes = [], [], []
    class_id = 0
    with torch.no_grad():
        for t in range(task + 1):
            n_classes_t = taskcla[t][1]
            indexes.append(np.arange(class_id, class_id + n_classes_t))
            if type(heads) == torch.nn.Linear:  # Single head
                biases.append(heads.bias[class_id: class_id + n_classes_t].detach().cpu().numpy())
                weights.append((heads.weight[class_id: class_id + n_classes_t] ** 2).sum(1).sqrt().detach().cpu().numpy())
            else:  # Multi-head
                weights.append((heads[t].weight ** 2).sum(1).sqrt().detach().cpu().numpy())
                if type(heads[t]) == torch.nn.Linear:
                    biases.append(heads[t].bias.detach().cpu().numpy())
                else:
                    biases.append(np.zeros(weights[-1].shape))  # For LUCIR
            class_id += n_classes_t

    # Figure weights
    f_weights = plt.figure(dpi=300)
    ax = f_weights.subplots(nrows=1, ncols=1)
    for i, (x, y) in enumerate(zip(indexes, weights), 0):
        if sort_weights:
            ax.bar(x, sorted(y, reverse=True), label="Task {}".format(i))
        else:
            ax.bar(x, y, label="Task {}".format(i))
    ax.set_xlabel("Classes", fontsize=11, fontfamily='serif')
    ax.set_ylabel("Weights L2-norm", fontsize=11, fontfamily='serif')
    if num_classes is not None:
        ax.set_xlim(0, num_classes)
    if y_lim:
        ax.set_ylim(0, 5)
    ax.legend(loc='upper left', fontsize='11') #, fontfamily='serif')

    # Figure biases
    f_biases = plt.figure(dpi=300)
    ax = f_biases.subplots(nrows=1, ncols=1)
    for i, (x, y) in enumerate(zip(indexes, biases), 0):
        if sort_weights:
            ax.bar(x, sorted(y, reverse=True), label="Task {}".format(i))
        else:
            ax.bar(x, y, label="Task {}".format(i))
    ax.set_xlabel("Classes", fontsize=11, fontfamily='serif')
    ax.set_ylabel("Bias values", fontsize=11, fontfamily='serif')
    if num_classes is not None:
        ax.set_xlim(0, num_classes)
    if y_lim:
        ax.set_ylim(-1.0, 1.0)
    ax.legend(loc='upper left', fontsize='11') #, fontfamily='serif')

    return f_weights, f_biases


def last_layer_analysis_2(heads, features, targets, task, taskcla, y_lim=False, sort_weights=False):
    """Plot last layer weights and features symbol consistency"""
    print('Plotting last layer analysis2...')
    num_classes = sum([x for (_, x) in taskcla])
    weights, indexes = [], []
    class_id = 0
    with torch.no_grad():
        for t in range(task + 1):
            n_classes_t = taskcla[t][1]
            indexes.append(np.arange(class_id, class_id + n_classes_t))
            if type(heads) == torch.nn.Linear:  # Single head
                weights.append(heads.weight[class_id: class_id + n_classes_t].detach().cpu())
            else:  # Multi-head
                weights.append(heads[t].weight.detach().cpu())

            class_id += n_classes_t
    print("weights_{}******************************************".format(task))
    print(weights)
    # weights list [task][n_classes, feat_dim]
    # features list [task][n, feat_dim]
    # targets list [task][n,]
    # Figure weights
    consist_weight_feat = []
    with torch.no_grad():
        n_classes_t = 0
        for tt in range(task+1):
            n_classes_t += taskcla[tt][1]
        print(n_classes_t)
        weight = torch.cat(weights, dim=0)
        feat = torch.cat(features, dim=0)
        _target = torch.cat(targets, dim=0)
        target = _target.unsqueeze(dim=1).expand(-1, weight.size(-1))
        # scores = torch.zeros_like(feat)
        scores = torch.gather(weight, dim=0, index=target.long())
        scores = (scores * feat > 0).sum(dim=1)  # [n,]

        consist = torch.zeros((n_classes_t,), dtype=scores.dtype)
        target = _target
        consist = torch.scatter_add(consist, dim=0, index=target.long(), src=scores)
        cnts = torch.zeros((n_classes_t,), dtype=scores.dtype)
        cnts = torch.scatter_add(cnts, dim=0, index=target.long(), src=torch.ones_like(target, dtype=cnts.dtype))
        consist = consist/(cnts*feat.size(1))
        consist = consist.numpy()
    class_id = 0
    for t in range(task + 1):
        n_classes_t = taskcla[t][1]
        consist_weight_feat.append(consist[class_id: class_id+n_classes_t])
        class_id += n_classes_t

    f_consist = plt.figure(dpi=300)
    ax = f_consist.subplots(nrows=1, ncols=1)
    for i, (x, y) in enumerate(zip(indexes, consist_weight_feat), 0):
        if sort_weights:
            ax.bar(x, sorted(y, reverse=True), label="Task {}, Avg {:.6f}".format(i, y.mean()))
        else:
            ax.bar(x, y, label="Task {}, Avg {:.6f}".format(i, y.mean()))
    ax.set_xlabel("Classes", fontsize=11, fontfamily='serif')
    ax.set_ylabel("Consist", fontsize=11, fontfamily='serif')
    if num_classes is not None:
        ax.set_xlim(0, num_classes)
    if y_lim:
        # ax.set_ylim(0, 5)
        pass
    ax.legend(loc='upper left', fontsize='11') #, fontfamily='serif')

    return f_consist


