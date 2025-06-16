import torch
import torch.nn.functional as F


def get_all_data(loader, net, feature_dim):
    """
    :param loader: dataloader of source dataset
    :param net: feature extractor
    :param feature_dim: dimensions of feature vectors
    :return: features and labels of all the samples in source domain (1260, 288) (1260)
    """
    all_features = torch.empty(0, feature_dim)
    all_features = all_features.cuda()
    all_labels = torch.empty(0)
    all_labels = all_labels.cuda()

    net.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            img, label = data
            img, label = img.cuda(), label.cuda()
            feature = net(img)  # (batch_size, feature_dim)
            all_features = torch.cat((all_features, feature))
            all_labels = torch.cat((all_labels, label))

    all_features = F.normalize(all_features, p=2, dim=1)
    all_labels = all_labels.long()

    return all_features, all_labels


def source_prototype(feature_s, label_s, class_num, feature_dim):
    """
    :param feature_s: features of all the samples in source domain (1260, 288)
    :param label_s: labels of all the samples in source domain (1260)
    :param class_num: total number of class
    :param feature_dim: dimensions of feature vectors
    :return: source domain prototype
    """
    class_feature_sum = torch.zeros(class_num, feature_dim).cuda()
    class_count = torch.zeros(class_num).cuda()

    for i in range(label_s.shape[0]):
        class_feature_sum[label_s[i]] += feature_s[i]
        class_count[label_s[i]] += 1

    prototypes = class_feature_sum / class_count.unsqueeze(-1)
    prototypes = F.normalize(prototypes, p=2, dim=1)

    return prototypes


def hc_target_prototype(prototype_s, feature_s_all, label_s_all, loader, class_num, feature_dim, k_num, net, fc):
    """
    :param prototype_s: prototypes in source domain with shape (class_num, feature_dim)
    :param feature_s_all: features of all the samples in source domain (1260, 288)
    :param label_s_all: labels of all the samples in source domain (1260)
    :param loader: dataloader of target dataset
    :param class_num: total number of class
    :param feature_dim: dimensions of feature vectors
    :param k_num: k nearest source sample
    :param net: feature extractor
    :param fc: classifier
    :return: target domain prototype
    """
    hc_fea_t = torch.empty(0, feature_dim).cuda()
    hc_lb_t = torch.empty(0).cuda()

    class_feature_sum = torch.zeros(class_num, feature_dim).cuda()
    class_count = torch.zeros(class_num).cuda()

    net.eval()
    fc.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            img, _ = data
            img = img.cuda()
            fea_t = net(img)
            pre_t = fc(fea_t)
            ns_prototype = nearest_source_prototypes(fea_t, prototype_s)  # (batch_size)
            ns_example = nearest_source_sample(fea_t, feature_s_all, label_s_all, k_num)  # (batch_size)
            cons_indices, pseudo_labels = easy_example_cons(ns_prototype, ns_example, pre_t)

            hc_fea_t = torch.cat((hc_fea_t, fea_t[cons_indices]))
            hc_lb_t = torch.cat((hc_lb_t, pseudo_labels))

    hc_lb_t = hc_lb_t.long()

    for i in range(hc_lb_t.shape[0]):
        class_feature_sum[hc_lb_t[i]] += hc_fea_t[i]
        class_count[hc_lb_t[i]] += 1

    class_count[class_count == 0] = 1

    prototypes = class_feature_sum / class_count.unsqueeze(-1)
    prototypes = F.normalize(prototypes, p=2, dim=1)

    return prototypes


def hc_target_prototype_with_lbt(prototype_s, feature_s_all, label_s_all, loader_ulb, loader_lb, class_num, feature_dim, k_num, net, fc):
    """
    :param prototype_s: prototypes in source domain with shape (class_num, feature_dim)
    :param feature_s_all: features of all the samples in source domain (1260, 288)
    :param label_s_all: labels of all the samples in source domain (1260)
    :param loader_ulb: dataloader of unlabeled target dataset
    :param loader_lb: dataloader of labeled target dataset
    :param class_num: total number of class
    :param feature_dim: dimensions of feature vectors
    :param k_num: k nearest source sample
    :param net: feature extractor
    :param fc: classifier
    :return: target domain prototype
    """
    hc_fea_t = torch.empty(0, feature_dim).cuda()
    hc_lb_t = torch.empty(0).cuda()
    rl_fea_t = torch.empty(0, feature_dim).cuda()  # -> real labeled
    rl_lb_t = torch.empty(0).cuda()

    class_feature_sum = torch.zeros(class_num, feature_dim).cuda()
    class_count = torch.zeros(class_num).cuda()

    net.eval()
    fc.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader_ulb):
            img, _ = data
            img = img.cuda()
            fea_t = net(img)
            pre_t = fc(fea_t)
            ns_prototype = nearest_source_prototypes(fea_t, prototype_s)  # (batch_size)
            ns_example = nearest_source_sample(fea_t, feature_s_all, label_s_all, k_num)  # (batch_size)
            cons_indices, pseudo_labels = easy_example_cons(ns_prototype, ns_example, pre_t)

            hc_fea_t = torch.cat((hc_fea_t, fea_t[cons_indices]))
            hc_lb_t = torch.cat((hc_lb_t, pseudo_labels))

    hc_lb_t = hc_lb_t.long()

    with torch.no_grad():
        for batch_idx, data in enumerate(loader_lb):
            img_t, lb_t = data
            img_t, lb_t = img_t.cuda(), lb_t.cuda()
            fea_t = net(img_t)

            rl_fea_t = torch.cat((rl_fea_t, fea_t))
            rl_lb_t = torch.cat((rl_lb_t, lb_t))

    rl_lb_t = rl_lb_t.long()
    '''rl_fea_t: (5~40, 288) rl_lb_t: (5~40)'''

    for i in range(hc_lb_t.shape[0]):
        class_feature_sum[hc_lb_t[i]] += hc_fea_t[i]
        class_count[hc_lb_t[i]] += 1
    '''首先用高置信度的目标域样本计算目标域原型'''

    for i in range(rl_lb_t.shape[0]):
        class_feature_sum[rl_lb_t[i]] += rl_fea_t[i]
        class_count[rl_lb_t[i]] += 1
    '''再加上带有 gt 的目标域样本'''

    class_count[class_count == 0] = 1

    prototypes = class_feature_sum / class_count.unsqueeze(-1)
    prototypes = F.normalize(prototypes, p=2, dim=1)

    return prototypes


def target_prototype_without_lbt(data_loader_s, data_loader_t, fea_net, fea_dim, cls_net, class_num, k_num):
    all_source_features, all_source_labels = get_all_data(data_loader_s, fea_net, fea_dim)
    source_center = source_prototype(all_source_features, all_source_labels, class_num, fea_dim)
    target_center = hc_target_prototype(source_center, all_source_features, all_source_labels, data_loader_t, class_num,
                                        fea_dim, k_num, fea_net, cls_net)

    return target_center


def target_prototype_with_lbt(data_loader_s, data_loader_ult, data_loader_lbt, fea_net, fea_dim, cls_net, class_num,
                              k_num):
    all_source_features, all_source_labels = get_all_data(data_loader_s, fea_net, fea_dim)
    '''all_source_features: (1260, 288) all_source_labels: (1260)'''

    source_center = source_prototype(all_source_features, all_source_labels, class_num, fea_dim)

    hc_fea_t = torch.empty(0, fea_dim).cuda()
    hc_lb_t = torch.empty(0).cuda()
    rl_fea_t = torch.empty(0, fea_dim).cuda()  # -> real labeled
    rl_lb_t = torch.empty(0).cuda()

    class_feature_sum = torch.zeros(class_num, fea_dim).cuda()
    class_count = torch.zeros(class_num).cuda()

    fea_net.eval()
    cls_net.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader_ult):
            img_t, _ = data
            img_t = img_t.cuda()
            fea_t = fea_net(img_t)
            pre_t = cls_net(fea_t)
            ns_prototype = nearest_source_prototypes(fea_t, source_center)  # (batch_size)
            ns_example = nearest_source_sample(fea_t, all_source_features, all_source_labels, k_num)  # (batch_size)
            cons_indices, pseudo_labels = easy_example_cons(ns_prototype, ns_example, pre_t)

            hc_fea_t = torch.cat((hc_fea_t, fea_t[cons_indices]))
            hc_lb_t = torch.cat((hc_lb_t, pseudo_labels))

    hc_lb_t = hc_lb_t.long()

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader_lbt):
            img_t, lb_t = data
            img_t, lb_t = img_t.cuda(), lb_t.cuda()
            fea_t = fea_net(img_t)

            rl_fea_t = torch.cat((rl_fea_t, fea_t))
            rl_lb_t = torch.cat((rl_lb_t, lb_t))

    rl_lb_t = rl_lb_t.long()
    '''rl_fea_t: (5~40, 288) rl_lb_t: (5~40)'''

    for i in range(hc_lb_t.shape[0]):
        class_feature_sum[hc_lb_t[i]] += hc_fea_t[i]
        class_count[hc_lb_t[i]] += 1
    '''首先用高置信度的目标域样本计算目标域原型'''

    for i in range(rl_lb_t.shape[0]):
        class_feature_sum[rl_lb_t[i]] += rl_fea_t[i]
        class_count[rl_lb_t[i]] += 1
    '''再加上带有 gt 的目标域样本'''

    class_count[class_count == 0] = 1

    prototypes = class_feature_sum / class_count.unsqueeze(-1)
    prototypes = F.normalize(prototypes, p=2, dim=1)

    return prototypes


def nearest_source_prototypes(feature_t, prototype_s):
    nor_feature = F.normalize(feature_t, p=2, dim=1)
    dis_matrix = torch.cdist(nor_feature, prototype_s, p=2).squeeze(0)
    if dis_matrix.dim() == 1:
        dis_matrix = dis_matrix.unsqueeze(0)
    nearest_indices = torch.argmin(dis_matrix, dim=1)

    return nearest_indices


def nearest_se_prototypes(feature_t, prototype):
    nor_feature = F.normalize(feature_t, p=2, dim=1)
    dis_matrix = torch.cdist(nor_feature, prototype, p=2).squeeze(0)
    if dis_matrix.dim() == 1:
        dis_matrix = dis_matrix.unsqueeze(0)
    ne_values, ne_indices = dis_matrix.topk(k=2, largest=False, sorted=True, dim=1)

    return ne_indices, ne_values


def nearest_source_sample(feature_t, feature_s, label_s, k_num):
    """
    :param feature_t: feature vector of target domain data tensor with shape (batch_size, feature_dim)
    :param feature_s: features of all the samples in source domain (1260, 288)
    :param label_s: labels of all the samples in source domain (1260)
    :param k_num: k nearest source sample
    :return: the labels of the k source domain samples closest to the target domain sample with shape (batch_size, k)
    """
    nor_feature = F.normalize(feature_t, p=2, dim=1)
    dis_matrix = torch.cdist(nor_feature, feature_s, p=2).squeeze(0)
    if dis_matrix.dim() == 1:
        dis_matrix = dis_matrix.unsqueeze(0)
    # 选出每行欧氏距离最小的三个数据对应的索引
    _, nearest_indices = torch.topk(dis_matrix, k=k_num, dim=1, largest=False)
    nearest_samples = label_s[nearest_indices]

    return nearest_samples


def easy_example_cons(nearest_pro_s, nearest_ex_s, predict_vec):
    """
    select easy target domain samples based on the consistency of the classifier’s predictions and the nearest source domain prototype
    :param nearest_pro_s: the nearest source domain prototype with shape (batch_size)
    :param nearest_ex_s: the nearest source domain examples with shape (batch_size, k)
    :param predict_vec: the prediction output of the classifier with shape (batch_size, class_num)
    :return: the index of easy target domain examples in a batch and their corresponding labels
    """
    pre_res = torch.argmax(predict_vec, dim=1)  # (batch_size)
    label_matrix = torch.cat((nearest_ex_s, nearest_pro_s.view(nearest_pro_s.shape[0], 1)), dim=1)
    label_matrix = torch.cat((label_matrix, pre_res.view(pre_res.shape[0], 1)), dim=1)
    unique_counts = torch.tensor([len(torch.unique(row)) for row in label_matrix])
    cons_indices = torch.where(unique_counts == 1)[0]
    pseudo_labels = label_matrix[cons_indices, 0]

    return cons_indices, pseudo_labels.long()
