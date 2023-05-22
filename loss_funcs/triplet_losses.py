import torch
import torch.nn as nn


def _get_triplet_mask(labels):
    """
    条件1：i, j, k 不能相等
    条件2：labels[i]==labels[j] != labels[k]
    :param labels: [batch_size, 1]
    :return: bool
    """

    # i, j, k 不能相等
    indice_equal = torch.eye(labels.shape[0], dtype=bool).cuda()
    indice_not_equal = torch.logical_not(indice_equal)

    i_not_equal_j = torch.unsqueeze(indice_not_equal, 2)
    i_not_equal_k = torch.unsqueeze(indice_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indice_not_equal, 0)
    # ijk不相等时为 True
    indice_mask = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # labels[i]==labels[j] != labels[k]
    # print("labels.shape:", labels.shape)
    label_equal = torch.eq(labels, torch.transpose(labels, 1, 0))
    label_i_equal_j = torch.unsqueeze(label_equal, 2)
    label_i_equal_k = torch.unsqueeze(label_equal, 1)
    label_mask = torch.logical_and(label_i_equal_j, torch.logical_not(label_i_equal_k))

    valid_labels = torch.logical_and(indice_mask, label_mask)
    return valid_labels


def _pairwise_dist(embeddings, squared=False):
    """

    :param embeddings: [batch_size, embeddings_size]
    :param squared:
    :return:[batch_size, batch_size]
    """
    dot_product = torch.matmul(embeddings, torch.transpose(embeddings, 1, 0))

    a_2 = torch.diagonal(dot_product)
    a_2 = torch.unsqueeze(a_2, 1)

    pairwise_dist = a_2 - 2 * dot_product + torch.transpose(a_2, 1, 0)

    # 由于计算误差，可能会有小于0的元素
    pairwise_dist = torch.maximum(pairwise_dist, torch.zeros(pairwise_dist.shape).cuda())
    # print(pairwise_dist)
    if not squared:
        # 由于对0开方后，其梯度为无限，先给值为0的元素添加小量，再消去偏置
        mask = torch.eq(pairwise_dist, 0.0)
        # bool to float
        mask = mask + 0.0
        pairwise_dist = torch.sqrt(pairwise_dist + mask * 1e-16)

        pairwise_dist = torch.multiply(pairwise_dist, (1.0 - mask))
    # print(pairwise_dist)
    return pairwise_dist


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """
    求整个batch的triplet_loss
    :param labels:
    :param embeddings:
    :param margin:
    :param squared:
    :return:
    """
    # 求成对距离矩阵
    pairwise_dist = _pairwise_dist(embeddings, squared=squared)
    # triplet_loss[i, j, k] 代表 dist[i, j] - dist[i, k]
    # 此时triplet_loss 还不完善，因为不确定 j是否为i的正样本， k是否为i的负样本
    anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2)
    anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    # print("triplet_loss: ", triplet_loss)
    # triplet_loss中符合条件的部分
    mask = _get_triplet_mask(labels)
    mask = mask + 0.0
    # print("mask: ", mask)
    triplet_loss = torch.multiply(triplet_loss, mask)
    # 除去小于0的部分
    triplet_loss = torch.maximum(triplet_loss, torch.zeros(triplet_loss.shape).cuda())
    # 有效的triplet_loss
    valid_triplet_loss = torch.gt(triplet_loss, 1e-16) + 0.0
    # print("valid_triplet_loss: ", valid_triplet_loss)
    num_positive_triplet_loss = torch.sum(valid_triplet_loss)

    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplet_loss + 1e-16)
    # print("triplet_loss: ", triplet_loss)
    return triplet_loss


class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, embeddings, labels, margin, squared=False):
        tripletloss = batch_all_triplet_loss(labels, embeddings, margin, squared=squared)
        return tripletloss


'''
if __name__ == '__main__':
    labels = torch.Tensor([0, 1, 2, 0])
    embeddings = torch.rand(4,3)
    labels = torch.unsqueeze(labels, 1)
    batch_all_triplet_loss(labels, embeddings, margin=0)
'''
