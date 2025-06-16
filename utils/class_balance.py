import math
import torch
import numpy as np
import torch.nn.functional as F


def balance_sample_via_ratio(target_loader, target_data, fea_net, cls_net, class_num, anno_indices,
                             minority_class_ratio, sampling_rate):
    fea_net.eval()
    cls_net.eval()

    gt_lb_all = np.array([])
    pred_lb_all = np.array([])
    pred_prob_all = np.array([])

    select_indices = np.array([])

    with torch.no_grad():
        for batch_idx, data in enumerate(target_loader):
            img, lb = data
            img, lb = img.cuda(), lb.cuda()
            fea = fea_net(img)
            output = cls_net(fea)
            output = F.softmax(output, dim=1)
            pred_prob = output.data.max(1)[0]
            pred_lb = output.data.max(1)[1]

            pred_prob_all = np.append(pred_prob_all, pred_prob.cpu().numpy())
            pred_lb_all = np.append(pred_lb_all, pred_lb.cpu().numpy())
            gt_lb_all = np.append(gt_lb_all, lb.cpu().numpy())

    category, counts = np.unique(pred_lb_all, return_counts=True)
    min_count_index = np.argmin(counts)
    minority_class = int(category[min_count_index])
    '''数量最少的类'''
    minority_class_indices = np.where(pred_lb_all == minority_class)[0]
    '''被预测为该类别的所有样本的索引'''
    minority_class_pre_prob = pred_prob_all[minority_class_indices]
    '''被预测为数量最少类的样本对应的预测概率'''
    minority_class_select_num = int(len(minority_class_indices) * minority_class_ratio)
    '''最少类选择的数量 ——> 被预测为最少类的样本总数✖百分比参数，即取该类预测概率最高的前 minority_class_ratio%'''
    minority_class_select_indices = np.argpartition(minority_class_pre_prob, -minority_class_select_num)[
                                         -minority_class_select_num:]
    minority_class_select_indices = minority_class_indices[minority_class_select_indices]
    '''转化为针对整个目标域的索引'''
    minority_class_pre_acc = (
                np.sum(pred_lb_all[minority_class_select_indices] == gt_lb_all[minority_class_select_indices])
                / len(minority_class_select_indices))
    print('The accuracy of the least class prediction: ' + str(100 * minority_class_pre_acc) + '%')

    all_indices = np.arange(0, len(target_loader.dataset))
    un_anno_indices = np.setdiff1d(all_indices, anno_indices)
    un_anno_pre_lb = pred_lb_all[un_anno_indices]
    un_anno_pre_prob = pred_prob_all[un_anno_indices]
    un_anno_pre_hist = np.histogram(un_anno_pre_lb, bins=class_num, weights=un_anno_pre_prob, range=(0, class_num))[0]

    anno_lb = target_loader.dataset.tensors[1][anno_indices]
    anno_hist = np.histogram(anno_lb, bins=class_num, range=(0, class_num))[0]

    all_hist = torch.from_numpy(un_anno_pre_hist + anno_hist)
    all_hist = F.normalize(all_hist, p=1, dim=0).float()

    for class_idx in range(class_num):
        if class_idx == minority_class:
            select_indices = np.append(select_indices, minority_class_select_indices)
        else:
            imbalance_ratio = torch.div(all_hist[minority_class], all_hist[class_idx])  # 当前类的类不平衡率
            sampling_ratio = math.pow(imbalance_ratio, sampling_rate) * minority_class_ratio
            other_class_indices = np.where(pred_lb_all == class_idx)[0]
            other_class_pre_prob = pred_prob_all[other_class_indices]
            other_class_select_num = int(len(other_class_indices) * sampling_ratio)
            other_class_select_indices = np.argpartition(other_class_pre_prob, -other_class_select_num)[
                                         -other_class_select_num:]
            other_class_select_indices = other_class_indices[other_class_select_indices]
            select_indices = np.append(select_indices, other_class_select_indices)

    select_indices = select_indices.astype(int)
    select_data = target_data[select_indices]
    select_label = pred_lb_all[select_indices]

    return select_data, select_label
