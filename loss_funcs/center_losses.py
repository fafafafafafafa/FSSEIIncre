import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    def __init__(self, classes, feature_dims, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.classes = classes
        self.feature_dims = feature_dims
        self.use_gpu = use_gpu
        if use_gpu:
            centers = nn.Parameter(torch.randn(self.classes, self.feature_dims).cuda())
        else:
            centers = nn.Parameter(torch.randn(self.classes, self.feature_dims))
        self.centers = centers

    def forward(self, x, labels):

        # labels: [N_way, K_shot]
        batch_size = x.shape[0]    # x_shape: torch.Size([N_way*K_shot, 1024])
        # dist_mat: torch.Size([batch_size, classes])
        # print('x: ', x)
        # print('centers: ', self.centers)
        dist_mat = torch.sum(torch.square(x), 1, keepdim=True).expand(batch_size, self.classes) + \
             torch.sum(torch.square(self.centers), 1, keepdim=True).expand(self.classes, batch_size).t()
        dist_mat = dist_mat - 2*torch.matmul(x, self.centers.t())

        # dist_mat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.classes) + \
        #    torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.classes, batch_size).t()
        # dist_mat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        labels = torch.reshape(labels, (-1, 1)).expand(batch_size, self.classes)
        classes_mat = torch.arange(self.classes).expand(batch_size, self.classes).long()
        if self.use_gpu:
            # print('use_gpu:', self.use_gpu)
            classes_mat = classes_mat.cuda()
        mask = labels.eq(classes_mat).float()
        dist_mat = dist_mat*mask
        center_loss = torch.sum(dist_mat.clamp(min=1e-12, max=1e+12))/(batch_size*self.feature_dims)
        # 获取出现过的label
        mask1 = torch.sum(mask, 0).bool()
        support_centers = self.centers[mask1, :]
        # print('support_centers:', support_centers.shape)
        return center_loss, support_centers


if __name__ == '__main__':
    labels = torch.Tensor([1, 2, 3])
    print(torch.matmul(labels, labels.t()))
    print(torch.pow(labels, 2))
