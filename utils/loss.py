import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """
    Center loss.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

    def forward(self, x, labels, centers):
        """
        :param x: feature matrix with shape (batch_size, feat_dim).
        :param labels: ground truth labels with shape (batch_size).
        :param centers: class centers in source/target domain with shape (num_classes, feat_dim).
        :return:
        """
        batch_size = x.size(0)
        nor_x = F.normalize(x, p=2, dim=1)
        distmat = torch.pow(nor_x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(nor_x, centers.t(), beta=1, alpha=-2)
        '''
        example: batch_size = 5, feat_dim = 5, num_classes = 3

        x = tensor([[0.6257, 0.3607, 0.9866, 0.6344, 0.2980],
                    [0.5355, 0.5017, 0.3524, 0.9569, 0.4103],
                    [0.4737, 0.0169, 0.4675, 0.5581, 0.8298],
                    [0.1184, 0.8849, 0.9200, 0.3165, 0.2948],
                    [0.7789, 0.9966, 0.0986, 0.3552, 0.1055]])
        centers = tensor([[0.6270, 0.2903, 0.5297, 0.6274, 0.6271],
                          [0.8366, 0.9346, 0.2756, 0.5573, 0.6124],
                          [0.0564, 0.9708, 0.3582, 0.8947, 0.9098]])

        torch.pow(x, 2).sum(dim=1, keepdim=True): 每一行数据的平方和
        tensor([[1.9862],
                [1.7466],
                [1.4434],
                [1.8306],
                [1.7469]])

        expand(batch_size, self.num_classes): 扩展至 num_classes 列 ——> shape(batch_size, self.num_classes)
        tensor([[1.9862, 1.9862, 1.9862],
                [1.7466, 1.7466, 1.7466],
                [1.4434, 1.4434, 1.4434],
                [1.8306, 1.8306, 1.8306],
                [1.7469, 1.7469, 1.7469]])

        同上可得 torch.pow(self.centers, 2).sum(dim=1, keepdim=True)
        tensor([[1.5449],
                [2.3349],
                [2.7022]])

        expand(self.num_classes, batch_size): 扩展至 batch_size 列 ——> shape(self.num_classes, batch_size)
        tensor([[1.5449, 1.5449, 1.5449, 1.5449, 1.5449],
                [2.3349, 2.3349, 2.3349, 2.3349, 2.3349],
                [2.7022, 2.7022, 2.7022, 2.7022, 2.7022]])

        然后转置
        tensor([[1.5449, 2.3349, 2.7022],
                [1.5449, 2.3349, 2.7022],
                [1.5449, 2.3349, 2.7022],
                [1.5449, 2.3349, 2.7022],
                [1.5449, 2.3349, 2.7022]])

        distmat = tensor([[3.5311, 4.3212, 4.6884],
                          [3.2915, 4.0816, 4.4488],
                          [2.9881, 3.7782, 4.1454],
                          [3.3754, 4.1655, 4.5327],
                          [3.2918, 4.0819, 4.4491]]) batch_size*num_class
        distmat 中第 i 行第 j 列的元素 dij 表示：当前 batch 中第 i 个样本的特征向量的平方和 + 第 j 个类别中心的特征向量的平方和

        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2) <=> distmat = 1 * distmat + (-2) * (x @ self.centers.t()) 
        x @ self.centers.t(): (batch_size, num_classes) 第 i 行第 j 列的元素表示当前 batch 中第 i 个样本的特征向量和第 j 个类别中心的特征向量的内积

        distmat 中第 i 行第 j 列的元素 dij 表示：当前 batch 中第 i 个样本和第 j 个类别中心的欧式距离的平方, 即 ||xi-cyj|_2^2
        distmat = tensor([[0.3221, 0.9842, 1.5333],
                          [0.2401, 0.4845, 0.7030],
                          [0.1480, 1.0580, 1.2157],
                          [0.9716, 1.0924, 1.0394],
                          [1.0539, 0.3363, 1.5280]])
        '''

        classes = torch.arange(self.num_classes).long().cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        '''
        classes: tensor([0, 1, 2]) 即包含元素 0~num_classes-1 的 tensor
        example: labels = torch.tensor([2, 0, 1, 1, 0]) (batch_size,)
        labels.unsqueeze(1).expand(batch_size, self.num_classes): 扩展至 num_classes 列 ——> shape(batch_size, self.num_classes)
        tensor([[2, 2, 2],
                [0, 0, 0],
                [1, 1, 1],
                [1, 1, 1],
                [0, 0, 0]])

        classes.expand(batch_size, self.num_classes): 扩展至 batch_size 行 ——> shape(batch_size, self.num_classes)
        tensor([[0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2]])

        mask: tensor([[False, False,  True],
                     [ True, False, False],
                     [False,  True, False],
                     [False,  True, False],
                     [ True, False, False]])
        mask (batch_size, num_classes) 表示掩码, 每一行中有一个元素为 True, 其他均为 False, 第 i 行表示当前 batch 中第 i 个样本所属类别
        '''

        dist = distmat * mask.float()
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss = dist.sum() / batch_size
        '''
        利用 mask 掩盖掉当前样本到其他类别中心的距离，只留下样本到其对应类别中心的距离
        dist = tensor([[0.0000, 0.0000, 1.5333],
                       [0.2401, 0.0000, 0.0000],
                       [0.0000, 1.0580, 0.0000],
                       [0.0000, 1.0924, 0.0000],
                       [1.0539, 0.0000, 0.0000]])
        dist.clamp(min=1e-12, max=1e+12): 将所有元素限制在范围 [1e-12, 1e+12]内
        '''

        return loss


def softmax(x):
    e_x = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])

    return e_x / torch.sum(e_x, dim=1, keepdim=True)


def predict(X, centers):
    """
    对于查询样本，基于其到每个原型的距离通过 softmax 函数产生其在 C 个类别上的分数分布
    :param X: 二维数组，每一行表示一个样本的特征向量
    :param centers: 二维数组，表示每个类别的中心（原型）的特征向量
    :return: 二维数组，每一行表示一个样本属于每个类别的概率分布
    """
    distances = torch.cdist(X, centers, p=2).squeeze(0)
    if distances.dim() == 1:
        distances = distances.unsqueeze(0)

    return softmax(-distances)


def symmetric_kl_divergence(P, Q):
    """
    计算两个二维数组的对称成对 KL 散度
    :param P: 二维数组，表示当前数据在一组原型上的概率分布
    :param Q: 二维数组，表示当前数据在另一组原型上的概率分布
    :return: 一维数组，包含了每行对称 KL 散度的计算结果
    """
    kl_divergence_pq = torch.sum(P * torch.log(P / Q), dim=1)
    kl_divergence_qp = torch.sum(Q * torch.log(Q / P), dim=1)

    return 0.5 * (kl_divergence_pq + kl_divergence_qp)


class DiscrepancyLoss(nn.Module):
    def __init__(self):
        super(DiscrepancyLoss, self).__init__()

    def forward(self, xs, xt, center_s, center_t):
        """
        :param xs: source domain feature matrix with shape (batch_size, feat_dim).
        :param xt: target domain feature matrix with shape (pseudo_num, feat_dim).
        :param center_s: source prototype feature matrix with shape (num_classes, feat_dim).
        :param center_t: target prototype feature matrix with shape (num_classes, feat_dim).
        :return:
        """
        batch_size = xs.size(0)
        pseudo_num = xt.size(0)

        nor_xs = F.normalize(xs, p=2, dim=1)
        nor_xt = F.normalize(xt, p=2, dim=1)

        score_distri_ss = predict(nor_xs, center_s)
        score_distri_st = predict(nor_xs, center_t)
        score_distri_ts = predict(nor_xt, center_s)
        score_distri_tt = predict(nor_xt, center_t)
        '''
        score_distri_ss (batch_size, num_classes) 源域样本在源域原型上的分数分布
        score_distri_st (batch_size, num_classes) 源域样本在目标域原型上的分数分布
        score_distri_ts (pseudo_num, num_classes) 目标域样本在源域原型上的分数分布
        score_distri_tt (pseudo_num, num_classes) 目标域样本在目标域原型上的分数分布
        '''

        dkl_s = symmetric_kl_divergence(score_distri_ss, score_distri_st)
        dkl_t = symmetric_kl_divergence(score_distri_ts, score_distri_tt)
        '''
        dkl_s 一个具体的值 表示当前 batch 的所有源域样本在源域原型和目标域原型上的分数分布之间的差异, 通过成对 KL 散度衡量
        dkl_t 一个具体的值 表示当前的所有高置信度目标域样本在源域原型和目标域原型上的分数分布之间的差异, 通过成对 KL 散度衡量
        '''

        loss = dkl_s.sum() / batch_size + dkl_t.sum() / pseudo_num

        return loss
