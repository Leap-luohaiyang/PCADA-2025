import numpy as np
from sklearn import metrics
from utils.prototype import *


def mmd_rbf(X, Y, gamma=1.0):
    """
    MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)

    return XX.mean() + YY.mean() - 2 * XY.mean()


def default_gamma(X: torch.Tensor):
    gamma = 1.0 / X.shape[1]

    return gamma


def rbf_kernel(X: torch.Tensor, gamma: float = None):
    assert len(X.shape) == 2

    if gamma is None:
        gamma = default_gamma(X)
    K = torch.cdist(X, X)
    K.fill_diagonal_(0)
    K.pow_(2)
    K.mul_(-gamma)
    K.exp_()

    return K


def rbf_kernel_diff(X1: torch.Tensor, X2: torch.Tensor, gamma: float = None):
    assert len(X1.shape) == 2
    assert len(X2.shape) == 2

    if gamma is None:
        gamma = default_gamma(X1)
    K = torch.cdist(X1, X2)
    K.fill_diagonal_(0)
    K.pow_(2)
    K.mul_(-gamma)
    K.exp_()

    return K


def query_selection_adjust_num(fea_net, fea_dim, cls_net, pro_t, target_loader, selected_group_idx, eta):
    fea_net.eval()
    cls_net.eval()

    all_features = torch.empty(0, fea_dim).cuda()
    cls_pre_lb = torch.empty(0).cuda()  # 根据分类器预测的 label
    cls_pre_prob = torch.empty(0).cuda()  # 分类器预测 softmax 输出的最大和第二大值
    pro_t_pre_lb = torch.empty(0).cuda()  # 根据最近目标域原型预测的 label

    with torch.no_grad():
        for batch_idx, data in enumerate(target_loader):
            img, label = data
            img, label = img.cuda(), label.cuda()
            fea = fea_net(img)  # (batch_size, 288)
            all_features = torch.cat((all_features, fea))
            pre_prob_vec = F.softmax(cls_net(fea), dim=1)  # (batch_size, 7)
            pre_prob, pre_cls = pre_prob_vec.topk(k=2, dim=1)  # (batch_size, 2) 分类器最高预测概率和第二高预测概率对应的预测 label
            cls_pre_lb = torch.cat((cls_pre_lb, pre_cls), dim=0)  # (53200, 2)
            cls_pre_prob = torch.cat((cls_pre_prob, pre_prob), dim=0)  # (53200, 2)

            pre_pro_t, _ = nearest_se_prototypes(fea, pro_t)  # (batch_size, 2) 最近和第二近目标域原型对应的预测 label
            pro_t_pre_lb = torch.cat((pro_t_pre_lb, pre_pro_t), dim=0)  # (53200, 2)

    all_selected_idx = np.array([item for sublist in selected_group_idx.values() for item in sublist])

    cls_pre_lb = cls_pre_lb.cpu().numpy()
    cls_pre_lb = cls_pre_lb.astype(int)

    cls_pre_prob = cls_pre_prob.cpu().numpy()

    pro_t_pre_lb = pro_t_pre_lb.cpu().numpy()
    pro_t_pre_lb = pro_t_pre_lb.astype(int)

    diff_cls_nt_indices = np.where((cls_pre_lb[:, 0] != pro_t_pre_lb[:, 0]) &
                                   (cls_pre_lb[:, 1] == pro_t_pre_lb[:, 0]) &
                                   (cls_pre_lb[:, 0] == pro_t_pre_lb[:, 1]))[0]
    mask_select = np.isin(diff_cls_nt_indices, all_selected_idx)
    diff_cls_nt_indices = diff_cls_nt_indices[~mask_select]
    diff_pre_cls = cls_pre_lb[diff_cls_nt_indices]
    diff_pre_pro = pro_t_pre_lb[diff_cls_nt_indices]
    diff_pre_group = [tuple(row) for row in diff_pre_pro]

    cls_pre_set = sorted({row[0] for row in diff_pre_cls})
    group_set = set(map(tuple, diff_pre_pro))
    each_cls_group = {i: [] for i in cls_pre_set}
    each_cls_num = {i: [] for i in cls_pre_set}
    for group in group_set:
        if group[1] in cls_pre_set:
            each_cls_group[group[1]].append(group)
            each_cls_num[group[1]].append(np.sum(np.all(diff_pre_pro == group, axis=1)))

    each_group_num = []
    for key, value_list in each_cls_num.items():
        for index, group_num in enumerate(value_list):
            each_group_num.append((group_num, key, index))

    top_num_group = []
    if len(each_group_num) >= 5:
        top_group = sorted(each_group_num, key=lambda x: x[0], reverse=True)[:5]
    else:
        top_group = sorted(each_group_num, key=lambda x: x[0], reverse=True)
    for value, key, index in top_group:
        top_num_group.append(each_cls_group[key][index])
    print('The discriminant pairs selected in this round of annotation: ' + '、'.join(map(str, top_num_group)))

    group_idx_dict = dict()
    group_fea_dict = dict()
    select_idx = np.array([])

    for idx in range(len(top_num_group)):
        each_group = top_num_group[idx]  # (4, 5)
        group_idx = [index for index, value in enumerate(diff_pre_group) if value == each_group]
        group_idx_to_whole_td = diff_cls_nt_indices[group_idx]
        group_cor_prob = cls_pre_prob[group_idx_to_whole_td]
        group_uncertainty = group_cor_prob[:, 0] - group_cor_prob[:, 1]
        most_uncertain_indices = np.where(group_uncertainty <= np.percentile(group_uncertainty, eta))[0]
        group_idx_to_whole_td = group_idx_to_whole_td[most_uncertain_indices]
        group_idx_dict[each_group] = group_idx_to_whole_td
        group_fea_dict[each_group] = all_features[group_idx_to_whole_td]

    adjust_num_dict = {1: [5], 2: [3, 2], 3: [2, 2, 1], 4: [2, 1, 1, 1]}

    if len(top_num_group) == 5:
        for group, group_fea in group_fea_dict.items():
            feat_rbf = rbf_kernel(group_fea)
            mmd_dis_first_term = feat_rbf.diagonal()
            mmd_dis_second_term = 2 * feat_rbf.sum(1) / group_fea.shape[0]
            mmd_dis_third_term = torch.sum(feat_rbf) / feat_rbf.numel()
            represent_mmd_dis = mmd_dis_first_term - mmd_dis_second_term + mmd_dis_third_term

            if group in selected_group_idx:
                previous_same_group_idx = selected_group_idx[group]
                previous_same_group_fea = all_features[previous_same_group_idx]
                feat_rbf_cur_with_pre = rbf_kernel_diff(group_fea, previous_same_group_fea)
                feat_rbf_pre = rbf_kernel(previous_same_group_fea)
                previous_mmd_dis_first_term = feat_rbf.diagonal()
                previous_mmd_dis_second_term = 2 * feat_rbf_cur_with_pre.sum(1) / previous_same_group_fea.shape[0]
                previous_mmd_dis_third_term = torch.sum(feat_rbf_pre) / feat_rbf_pre.numel()
                diversity_mmd_dis = previous_mmd_dis_first_term - previous_mmd_dis_second_term + previous_mmd_dis_third_term

                mmd_dis = represent_mmd_dis - diversity_mmd_dis
                each_group_select_idx = torch.argmin(mmd_dis)
                each_group_select_idx = group_idx_dict[group][each_group_select_idx.cpu().numpy()]

                selected_group_idx[group] = np.append(selected_group_idx[group], each_group_select_idx)
                select_idx = np.append(select_idx, each_group_select_idx)
            else:
                mmd_dis = represent_mmd_dis
                each_group_select_idx = torch.argmin(mmd_dis)
                each_group_select_idx = group_idx_dict[group][each_group_select_idx.cpu().numpy()]
                if isinstance(each_group_select_idx, np.ndarray):
                    selected_group_idx[group] = each_group_select_idx
                else:
                    selected_group_idx[group] = np.array([each_group_select_idx])
                select_idx = np.append(select_idx, each_group_select_idx)
    else:
        each_select_num = adjust_num_dict[len(top_num_group)]
        for group_idx, (group, group_fea) in enumerate(group_fea_dict.items()):
            sample_num_cur_group = each_select_num[group_idx]

            feat_rbf = rbf_kernel(group_fea)
            mmd_dis_first_term = feat_rbf.diagonal()
            mmd_dis_second_term = 2 * feat_rbf.sum(1) / group_fea.shape[0]
            mmd_dis_third_term = torch.sum(feat_rbf) / feat_rbf.numel()
            represent_mmd_dis = mmd_dis_first_term - mmd_dis_second_term + mmd_dis_third_term

            if group in selected_group_idx:
                previous_same_group_idx = selected_group_idx[group]
                previous_same_group_fea = all_features[previous_same_group_idx]
                feat_rbf_cur_with_pre = rbf_kernel_diff(group_fea, previous_same_group_fea)
                feat_rbf_pre = rbf_kernel(previous_same_group_fea)
                previous_mmd_dis_first_term = feat_rbf.diagonal()
                previous_mmd_dis_second_term = 2 * feat_rbf_cur_with_pre.sum(1) / previous_same_group_fea.shape[0]
                previous_mmd_dis_third_term = torch.sum(feat_rbf_pre) / feat_rbf_pre.numel()
                diversity_mmd_dis = previous_mmd_dis_first_term - previous_mmd_dis_second_term + previous_mmd_dis_third_term

                mmd_dis = represent_mmd_dis - diversity_mmd_dis
                _, each_group_select_idx = torch.topk(mmd_dis, k=sample_num_cur_group, largest=False)
                each_group_select_idx = group_idx_dict[group][each_group_select_idx.cpu().numpy()]

                selected_group_idx[group] = np.append(selected_group_idx[group], each_group_select_idx)
                select_idx = np.append(select_idx, each_group_select_idx)
            else:
                mmd_dis = represent_mmd_dis
                _, each_group_select_idx = torch.topk(mmd_dis, k=sample_num_cur_group, largest=False)
                each_group_select_idx = group_idx_dict[group][each_group_select_idx.cpu().numpy()]

                if isinstance(each_group_select_idx, np.ndarray):
                    selected_group_idx[group] = each_group_select_idx
                else:
                    selected_group_idx[group] = np.array([each_group_select_idx])
                select_idx = np.append(select_idx, each_group_select_idx)

    return select_idx.astype(int), selected_group_idx


def query_selection(fea_net, fea_dim, cls_net, pro_t, target_loader, selected_group_idx, eta):
    fea_net.eval()
    cls_net.eval()

    all_features = torch.empty(0, fea_dim).cuda()
    cls_pre_lb = torch.empty(0).cuda()  # 根据分类器预测的 label
    cls_pre_prob = torch.empty(0).cuda()  # 分类器预测 softmax 输出的最大和第二大值
    pro_t_pre_lb = torch.empty(0).cuda()  # 根据最近目标域原型预测的 label

    with torch.no_grad():
        for batch_idx, data in enumerate(target_loader):
            img, label = data
            img, label = img.cuda(), label.cuda()
            fea = fea_net(img)  # (batch_size, 288)
            all_features = torch.cat((all_features, fea))
            pre_prob_vec = F.softmax(cls_net(fea), dim=1)  # (batch_size, 7)
            pre_prob, pre_cls = pre_prob_vec.topk(k=2, dim=1)  # (batch_size, 2) 分类器最高预测概率和第二高预测概率对应的预测 label
            cls_pre_lb = torch.cat((cls_pre_lb, pre_cls), dim=0)  # (53200, 2)
            cls_pre_prob = torch.cat((cls_pre_prob, pre_prob), dim=0)  # (53200, 2)

            pre_pro_t, _ = nearest_se_prototypes(fea, pro_t)  # (batch_size, 2) 最近和第二近目标域原型对应的预测 label
            pro_t_pre_lb = torch.cat((pro_t_pre_lb, pre_pro_t), dim=0)  # (53200, 2)

    all_selected_idx = np.array([item for sublist in selected_group_idx.values() for item in sublist])

    cls_pre_lb = cls_pre_lb.cpu().numpy()
    cls_pre_lb = cls_pre_lb.astype(int)

    cls_pre_prob = cls_pre_prob.cpu().numpy()

    pro_t_pre_lb = pro_t_pre_lb.cpu().numpy()
    pro_t_pre_lb = pro_t_pre_lb.astype(int)

    diff_cls_nt_indices = np.where((cls_pre_lb[:, 0] != pro_t_pre_lb[:, 0]) &
                                   (cls_pre_lb[:, 1] == pro_t_pre_lb[:, 0]) &
                                   (cls_pre_lb[:, 0] == pro_t_pre_lb[:, 1]))[0]
    mask_select = np.isin(diff_cls_nt_indices, all_selected_idx)
    diff_cls_nt_indices = diff_cls_nt_indices[~mask_select]
    '''剔除已经选择过的样本的索引 diff_cls_nt_indices 即为当前预测既存在歧义又没有被选择过的样本在整个目标域中的索引'''
    diff_pre_cls = cls_pre_lb[diff_cls_nt_indices]
    '''符合要求的样本中，分类器预测的标签组合'''
    diff_pre_pro = pro_t_pre_lb[diff_cls_nt_indices]
    '''符合要求的样本中，目标域原型预测的标签'''
    diff_pre_group = [tuple(row) for row in diff_pre_pro]
    '''list len=1713 [(4, 5), (4, 5)...]'''

    cls_pre_set = sorted({row[0] for row in diff_pre_cls})
    '''[1, 2, 4, 5, 6] list 歧义组合中分类器预测的 label 种类'''
    group_set = set(map(tuple, diff_pre_pro))
    '''group_set = {(0, 1), (0, 2)...} set 所有的歧义组合，(目标域原型的预测，分类器的预测) (可能实际上的类别，预测成了的类别)'''
    each_cls_group = {i: [] for i in cls_pre_set}
    each_cls_num = {i: [] for i in cls_pre_set}
    '''each_cls_group = {1: [], 2: [], 4: [], 5: [], 6: []}'''
    '''each_cls_num = {1: [], 2: [], 4: [], 5: [], 6: []}'''
    for group in group_set:
        if group[1] in cls_pre_set:
            '''当前分类器可能错误预测的类别，将实际为 group[0] 的样本预测为了 group[1]'''
            each_cls_group[group[1]].append(group)
            each_cls_num[group[1]].append(np.sum(np.all(diff_pre_pro == group, axis=1)))
            '''each_cls_group = {1: [(0, 1), (6, 1), (2, 1)], 2: [(0, 2)], 4: [(6, 4), (2, 4)], 
                                 5: [(4, 5), (6, 5)], 6: [(2, 6), (5, 6), (0, 6), (1, 6)]} 
            被当前分类器预测为 1 的类别，实际上更有可能是 0、6、2 其余同理'''
            '''each_cls_num = {1: [430, 15, 106], 2: [2], 4: [7, 117], 5: [905, 87], 6: [6, 8, 10, 20]}'''

    each_group_num = []
    '''创建一个列表来存储所有值及其键和索引信息'''
    for key, value_list in each_cls_num.items():
        for index, group_num in enumerate(value_list):
            each_group_num.append((group_num, key, index))
    '''each_group_num = [(430, 1, 0), (15, 1, 1), (106, 1, 2), (2, 2, 0), (7, 4, 0), (117, 4, 1), 
                         (905, 5, 0), (87, 5, 1), (6, 6, 0), (8, 6, 1), (10, 6, 2), (20, 6, 3)]
                         (歧义对数量，对应的 key，key 对应值中的索引)'''

    top_num_group = []
    top_group = sorted(each_group_num, key=lambda x: x[0], reverse=True)[:5]
    for value, key, index in top_group:
        top_num_group.append(each_cls_group[key][index])
    '''top_num_group = [(4, 5), (0, 1), (2, 4), (2, 1), (6, 5)]'''
    print('The discriminant pairs selected in this round of annotation: ' + '、'.join(map(str, top_num_group)))

    group_idx_dict = dict()
    group_fea_dict = dict()
    select_idx = np.array([])

    for idx in range(len(top_num_group)):
        '''遍历选定的组合'''
        each_group = top_num_group[idx]  # (4, 5)
        group_idx = [index for index, value in enumerate(diff_pre_group) if value == each_group]
        '''[0, 1, 3, 5...] list len=905'''
        group_idx_to_whole_td = diff_cls_nt_indices[group_idx]
        '''转化为针对整个目标域的索引 [39, 95, 174, 269...] np.array len=905'''
        group_cor_prob = cls_pre_prob[group_idx_to_whole_td]
        '''[[0.5212198, 0.44826528], [0.61373544, 0.37734595]...] np.array (905, 2)'''
        group_uncertainty = group_cor_prob[:, 0] - group_cor_prob[:, 1]
        '''[0.073, 0.236, 0.437] np.array len=905'''
        most_uncertain_indices = np.where(group_uncertainty <= np.percentile(group_uncertainty, eta))[0]
        '''不确定性最高的那 20 % 的样本的索引，针对当前数组，len=181'''
        group_idx_to_whole_td = group_idx_to_whole_td[most_uncertain_indices]
        '''不确定性最高的那 20 % 的样本的索引，针对整个目标域，len=181'''
        group_idx_dict[each_group] = group_idx_to_whole_td
        '''group_idx_dict = {(4, 5): 预测歧义对为 (4 ,5) 且其中最不确定的那 20% 的样本在整个目标域中的索引 len=181}...'''
        group_fea_dict[each_group] = all_features[group_idx_to_whole_td]
        '''group_fea_dict = {(4, 5): 预测歧义对为 (4 ,5) 且其中最不确定的那 20% 的样本对应的特征}'''

    for group, group_fea in group_fea_dict.items():
        '''group: (4, 5) group_fea: shape(181, 288)'''
        feat_rbf = rbf_kernel(group_fea)
        '''shape=(181, 181) feat_rbf[i ,j] = k(f(xi), f(xj)) = exp(-γ||xi - xj||²) 对角线元素均为1'''
        mmd_dis_first_term = feat_rbf.diagonal()
        '''shape=(181,) mmd_dis_first_term[i] = k(f(xi), f(xi)) = 1 xi是待选出的单个元素'''
        mmd_dis_second_term = 2 * feat_rbf.sum(1) / group_fea.shape[0]
        '''shape=(181,) mmd_dis_second_term[i] = 2/nT ∑xj∈XT k(f(xi), f(xj)) xi是待选出的单个元素'''
        mmd_dis_third_term = torch.sum(feat_rbf) / feat_rbf.numel()
        '''mmd_dis_third_term = 1/nT² ∑xi,xj∈XT k(f(xi), f(xj)) 常量'''
        represent_mmd_dis = mmd_dis_first_term - mmd_dis_second_term + mmd_dis_third_term
        '''shape=(181,) 记录了当前 group 中每个样本的代表性'''

        if group in selected_group_idx:
            '''如果当前 group 在之前的 round 已经被选择过，需要计算本次查询 group 内的样本和之前该 group 选出的所有样本的 mmd 距离，以确保多样性'''
            previous_same_group_idx = selected_group_idx[group]
            '''属于当前 group 的已经被标注过的样本在整个目标域中的索引'''
            previous_same_group_fea = all_features[previous_same_group_idx]
            feat_rbf_cur_with_pre = rbf_kernel_diff(group_fea, previous_same_group_fea)
            feat_rbf_pre = rbf_kernel(previous_same_group_fea)
            previous_mmd_dis_first_term = feat_rbf.diagonal()
            previous_mmd_dis_second_term = 2 * feat_rbf_cur_with_pre.sum(1) / previous_same_group_fea.shape[0]
            previous_mmd_dis_third_term = torch.sum(feat_rbf_pre) / feat_rbf_pre.numel()
            diversity_mmd_dis = previous_mmd_dis_first_term - previous_mmd_dis_second_term + previous_mmd_dis_third_term

            mmd_dis = represent_mmd_dis - diversity_mmd_dis
            each_group_select_idx = torch.argmin(mmd_dis)
            each_group_select_idx = group_idx_dict[group][each_group_select_idx].item()

            selected_group_idx[group] = np.append(selected_group_idx[group], each_group_select_idx)
            select_idx = np.append(select_idx, each_group_select_idx)
        else:
            mmd_dis = represent_mmd_dis
            each_group_select_idx = torch.argmin(mmd_dis)
            each_group_select_idx = group_idx_dict[group][each_group_select_idx].item()

            selected_group_idx[group] = np.array([each_group_select_idx])
            select_idx = np.append(select_idx, each_group_select_idx)

    return select_idx.astype(int), selected_group_idx


def analyze(ten_idx, target_dataset):
    print("The queried samples belong to the following classes: ")
    for index in ten_idx:
        print(target_dataset[index][1].item(), end=' ')
        # classes_of_new_samples = torch.cat((classes_of_new_samples,torch.from_numpy(target_dataset[index][1])),0)
    print('\n')
