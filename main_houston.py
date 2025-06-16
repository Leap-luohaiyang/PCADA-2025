import argparse
import os
import random
import sys
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader
from torchsampler import ImbalancedDatasetSampler

sys.path.append("..")
from utils.class_balance import *
from utils.loss import *
from utils.prototype import *
from active.active_learning import query_selection, analyze
from datasets.get_dataset import ForeverDataIterator, get_loader
from datasets.data_pre import *
from models.classifier import SingleClassifier
from models.dcrn import DCRN


# Training settings
parser = argparse.ArgumentParser(description='Hyperspectral Image Classification')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--eta', type=int, default=15, metavar='E',
                    help='proportion of uncertain subsets in active selection strategy')
parser.add_argument('--rho', type=int, default=1, metavar='R',
                    help='sampling rate of the lowest frequency category (default: 1)')
parser.add_argument('--lambda_', type=float, default=0.3, metavar='L',
                    help='lambda value in class-balanced self-training module (default: 0.3)')
parser.add_argument('--gamma', type=float, default=0.01, metavar='G',
                    help='gamma value in class-balanced self-training module (default: 0.01)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.set_device(args.gpu)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


num_epoch = args.epochs
BATCH_SIZE = args.batch_size
HalfWidth = 3
nBand = 48
patch_size = 2 * HalfWidth + 1
CLASS_NUM = 7
K_NUM = 3

data_path_s = 'datas/Houston/Houston13.mat'
label_path_s = 'data/Houston/Houston13_7gt.mat'
data_path_t = 'data/Houston/Houston18.mat'
label_path_t = 'data/Houston/Houston18_7gt.mat'

source_data, source_label = load_data_houston(data_path_s, label_path_s)
target_data, target_label = load_data_houston(data_path_t, label_path_t)
print(source_data.shape, source_label.shape)
print(target_data.shape, target_label.shape)

criterion_s = nn.CrossEntropyLoss().cuda()  # Loss Function
center_loss = CenterLoss(CLASS_NUM, feat_dim=288).cuda()
discrepancy_loss = DiscrepancyLoss().cuda()

nDataSet = 10  # sample times

acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])

seeds = [678, 681, 774, 789, 812, 896, 946, 979, 1034, 1078]
sampling_epochs = [40, 45, 50, 55, 60, 65, 70, 75]


def train_without_tlb(epoch, data_loader_s, data_loader_t, temp_loader_s, temp_loader_t):
    fea_s_all, label_s_all = get_all_data(temp_loader_s, feature_extractor, feature_extractor.inter_size)
    prototype_s = source_prototype(fea_s_all, label_s_all, CLASS_NUM, feature_extractor.inter_size)  # (7, 288)
    prototype_t = hc_target_prototype(prototype_s, fea_s_all, label_s_all, temp_loader_t, CLASS_NUM,
                                      feature_extractor.inter_size, K_NUM, feature_extractor, classifier)

    feature_extractor.train()
    classifier.train()

    len_target_loader = len(data_loader_t)

    iter_source = iter(data_loader_s)
    iter_target = iter(data_loader_t)

    ''' Train one epoch '''
    for step, data_s in enumerate(iter_source):
        ''' load S '''
        img_s, label_s = data_s

        ''' load T'''
        data_t = next(iter_target)
        img_t, _ = data_t

        if step % len_target_loader == 0:
            iter_target = iter(data_loader_t)

        img_s, label_s = img_s.cuda(), label_s.cuda()
        img_t = img_t.cuda()
        img_all = torch.cat((img_s, img_t), 0)
        bs = label_s.shape[0]

        ''' select easy target domain examples '''
        fea_all = feature_extractor(img_all)
        fea_s = fea_all[:bs, :]
        fea_t = fea_all[bs:, :]
        pre_all = classifier(fea_all)
        pre_s = pre_all[:bs, :]
        pre_t = pre_all[bs:, :]

        ns_prototype = nearest_source_prototypes(fea_t, prototype_s)  # (batch_size)
        ns_example = nearest_source_sample(fea_t, fea_s_all, label_s_all, K_NUM)  # (batch_size, K_NUM)
        cons_indices, pseudo_labels = easy_example_cons(ns_prototype, ns_example, pre_t)  # 高置信度 (easy) 的目标域样本的索引，对应伪标签
        easy_fea_t = fea_t[cons_indices]

        if epoch == 0:
            cen_loss = 0
            dis_loss = 0
        else:
            cen_loss = center_loss(fea_s, label_s, prototype_s)
            cen_loss += center_loss(fea_s, label_s, prototype_t)
            cen_loss += center_loss(easy_fea_t, pseudo_labels, prototype_s)
            cen_loss += center_loss(easy_fea_t, pseudo_labels, prototype_t)

            dis_loss = discrepancy_loss(fea_s, fea_t, prototype_s, prototype_t)

        ''' back-propagate and update network '''
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        cls_loss = criterion_s(pre_s, label_s)
        all_loss = cls_loss + cen_loss + dis_loss
        all_loss.backward()

        optimizer_g.step()
        optimizer_f.step()


def train_with_tlb(data_loader_s, data_loader_ulb_t, data_loader_lb_t, temp_loader_s, temp_loader_t):
    fea_s_all, label_s_all = get_all_data(temp_loader_s, feature_extractor, feature_extractor.inter_size)
    prototype_s = source_prototype(fea_s_all, label_s_all, CLASS_NUM, feature_extractor.inter_size)  # (7, 288)
    prototype_t = hc_target_prototype(prototype_s, fea_s_all, label_s_all, temp_loader_t, CLASS_NUM,
                                      feature_extractor.inter_size, K_NUM, feature_extractor, classifier)

    feature_extractor.train()
    classifier.train()

    len_target_loader = len(data_loader_ulb_t)

    iter_source = iter(data_loader_s)
    iter_target = iter(data_loader_ulb_t)
    iter_lb_target = ForeverDataIterator(data_loader_lb_t)

    ''' Train one epoch '''
    for step, data_s in enumerate(iter_source):
        ''' load S '''
        img_s, label_s = data_s

        ''' load T'''
        data_t = next(iter_target)
        img_t, _ = data_t

        if step % len_target_loader == 0:
            iter_target = iter(data_loader_ulb_t)

        ''' load labeled T '''
        data_lb_t = next(iter_lb_target)
        img_lbt, label_lbt = data_lb_t
        img_lbt, label_lbt = img_lbt.cuda(), label_lbt.cuda()

        img_s, label_s = img_s.cuda(), label_s.cuda()
        img_t = img_t.cuda()
        img_all = torch.cat((img_s, img_t), 0)
        bs = label_s.shape[0]

        ''' Feedforward S,T and select easy target domain examples '''
        fea_all = feature_extractor(img_all)
        fea_s = fea_all[:bs, :]
        fea_t = fea_all[bs:, :]
        pre_all = classifier(fea_all)
        pre_s = pre_all[:bs, :]
        pre_t = pre_all[bs:, :]

        ''' Feedforward labeled T '''
        fea_lbt = feature_extractor(img_lbt)
        pre_lbt = classifier(fea_lbt)

        ns_prototype = nearest_source_prototypes(fea_t, prototype_s)  # (batch_size)
        ns_example = nearest_source_sample(fea_t, fea_s_all, label_s_all, K_NUM)  # (batch_size, K_NUM)
        cons_indices, pseudo_labels = easy_example_cons(ns_prototype, ns_example, pre_t)  # 高置信度 (easy) 的目标域样本的索引，对应伪标签
        easy_fea_t = fea_t[cons_indices]

        cen_loss = center_loss(fea_s, label_s, prototype_s)
        cen_loss += center_loss(fea_s, label_s, prototype_t)
        cen_loss += center_loss(easy_fea_t, pseudo_labels, prototype_s)
        cen_loss += center_loss(easy_fea_t, pseudo_labels, prototype_t)
        cen_loss += center_loss(fea_lbt, label_lbt, prototype_s)
        cen_loss += center_loss(fea_lbt, label_lbt, prototype_t)

        dis_loss = discrepancy_loss(fea_s, fea_t, prototype_s, prototype_t)

        ''' back-propagate and update network '''
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        cls_loss = criterion_s(pre_s, label_s)
        cls_loss += criterion_s(pre_lbt, label_lbt)
        all_loss = cls_loss + cen_loss + dis_loss
        all_loss.backward()

        optimizer_g.step()
        optimizer_f.step()


def balance_self_training(data_loader_s, data_loader_ulb_t, data_loader_lb_t, temp_loader_s, temp_loader_t, rho,
                          lambda_, gamma):
    fea_s_all, label_s_all = get_all_data(temp_loader_s, feature_extractor, feature_extractor.inter_size)
    prototype_s = source_prototype(fea_s_all, label_s_all, CLASS_NUM, feature_extractor.inter_size)  # (7, 288)
    prototype_t = hc_target_prototype(prototype_s, fea_s_all, label_s_all, temp_loader_t, CLASS_NUM,
                                      feature_extractor.inter_size, K_NUM, feature_extractor, classifier)

    feature_extractor.train()
    classifier.train()

    len_target_loader = len(data_loader_ulb_t)

    iter_source = iter(data_loader_s)
    iter_target = iter(data_loader_ulb_t)
    iter_lb_target = ForeverDataIterator(data_loader_lb_t)

    ''' Train one epoch '''
    for step, data_s in enumerate(iter_source):
        ''' load S '''
        img_s, label_s = data_s

        ''' load T'''
        data_t = next(iter_target)
        img_t, _ = data_t

        if step % len_target_loader == 0:
            iter_target = iter(data_loader_ulb_t)

        ''' load labeled T '''
        data_lb_t = next(iter_lb_target)
        img_lbt, label_lbt = data_lb_t
        img_lbt, label_lbt = img_lbt.cuda(), label_lbt.cuda()

        img_s, label_s = img_s.cuda(), label_s.cuda()
        img_t = img_t.cuda()
        img_all = torch.cat((img_s, img_t), 0)
        bs = label_s.shape[0]

        ''' Feedforward S,T and select easy target domain examples '''
        fea_all = feature_extractor(img_all)
        fea_s = fea_all[:bs, :]
        fea_t = fea_all[bs:, :]
        pre_all = classifier(fea_all)
        pre_s = pre_all[:bs, :]
        pre_t = pre_all[bs:, :]

        ''' Feedforward labeled T '''
        fea_lbt = feature_extractor(img_lbt)
        pre_lbt = classifier(fea_lbt)

        ns_prototype = nearest_source_prototypes(fea_t, prototype_s)  # (batch_size)
        ns_example = nearest_source_sample(fea_t, fea_s_all, label_s_all, K_NUM)  # (batch_size, K_NUM)
        cons_indices, pseudo_labels = easy_example_cons(ns_prototype, ns_example, pre_t)  # 高置信度 (easy) 的目标域样本的索引，对应伪标签
        easy_fea_t = fea_t[cons_indices]

        cen_loss = center_loss(fea_s, label_s, prototype_s)
        cen_loss += center_loss(fea_s, label_s, prototype_t)
        cen_loss += center_loss(easy_fea_t, pseudo_labels, prototype_s)
        cen_loss += center_loss(easy_fea_t, pseudo_labels, prototype_t)
        cen_loss += center_loss(fea_lbt, label_lbt, prototype_s)
        cen_loss += center_loss(fea_lbt, label_lbt, prototype_t)

        dis_loss = discrepancy_loss(fea_s, fea_t, prototype_s, prototype_t)

        ''' back-propagate and update network '''
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        cls_loss = criterion_s(pre_s, label_s)
        cls_loss += criterion_s(pre_lbt, label_lbt)
        all_loss = cls_loss + cen_loss + dis_loss
        all_loss.backward()

        optimizer_g.step()
        optimizer_f.step()

    confi_data, confi_labels = balance_sample_via_ratio(test_loader, test_data, feature_extractor, classifier,
                                                        CLASS_NUM, selected_idx,
                                                        minority_class_ratio=rho, sampling_rate=lambda_)
    confi_dataset = TensorDataset(torch.tensor(confi_data), torch.tensor(confi_labels, dtype=torch.long))
    confi_loader = torch.utils.data.DataLoader(
        confi_dataset,
        sampler=ImbalancedDatasetSampler(confi_dataset),
        batch_size=BATCH_SIZE)

    ''' Feedforward confident T '''
    for batch_idx, data_confi_t in enumerate(confi_loader):
        img_confit, label_confit = data_confi_t
        img_confit, label_confit = img_confit.cuda(), label_confit.cuda()

        ''' load confident T '''
        fea_confit = feature_extractor(img_confit)
        pre_confit = classifier(fea_confit)

        cls_loss = gamma * criterion_s(pre_confit, label_confit)

        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        cls_loss.backward()
        optimizer_g.step()
        optimizer_f.step()


def test(data_loader):
    pred_all = []
    gt_all = []

    feature_extractor.eval()
    classifier.eval()

    correct_add = 0
    size = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            img, label = data
            img, label = img.cuda(), label.cuda()
            fea = feature_extractor(img)
            output = classifier(fea)
            pred = output.data.max(1)[1]
            correct_add += pred.eq(label.data).cpu().sum()
            size += label.data.size()[0]
            pred_all = np.concatenate([pred_all, pred.cpu().numpy()])
            gt_all = np.concatenate([gt_all, label.data.cpu().numpy()])

    overall_acc = 100. * float(correct_add) / size
    acc[iDataSet] = 100. * float(correct_add) / size
    C = metrics.confusion_matrix(gt_all, pred_all)
    A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=float)
    k[iDataSet] = metrics.cohen_kappa_score(gt_all, pred_all)

    return overall_acc


for iDataSet in range(nDataSet):
    set_seed(seeds[iDataSet])

    '''data'''
    train_data_s, train_label_s = get_source_data(source_data, source_label, HalfWidth, 180)
    test_data, test_label, test_gt, RandPerm, Row, Column = get_target_data(target_data, target_label, HalfWidth)

    train_dataset_s = TensorDataset(torch.tensor(train_data_s), torch.tensor(train_label_s))
    test_dataset = TensorDataset(torch.tensor(test_data), torch.tensor(test_label))
    '''未被标注的目标域样本，初始时为所有目标域样本'''
    unlabeled_dataset_t = TensorDataset(torch.tensor(test_data), torch.tensor(test_label))

    train_loader_s = DataLoader(train_dataset_s, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    loader_s_no_shuffle = DataLoader(train_dataset_s, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    train_loader_t = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    unlabeled_loader_t = DataLoader(unlabeled_dataset_t, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    labeled_loader_t = None
    labeled_dataset_t = None

    '''model'''
    feature_extractor = DCRN(nBand, patch_size, CLASS_NUM).cuda()
    classifier = SingleClassifier().cuda()

    '''optimizer'''
    optimizer_g = torch.optim.SGD(list(feature_extractor.parameters()), lr=args.lr, weight_decay=0.0005)
    optimizer_f = torch.optim.SGD(list(classifier.parameters()), lr=args.lr, momentum=0.9, weight_decay=0.0005)

    already_selected_group = dict()
    selected_idx = np.array([])

    for ep in range(num_epoch):
        if ep in sampling_epochs:
            '''第一次采样时，unlabeled_loader_t 将遍历所有未标记的目标域样本'''
            '''遍历未标注的目标域样本，获取高置信度目标域样本'''
            '''在高置信度目标域样本的基础上结合标注的样本最终计算得到目标域原型'''
            if labeled_loader_t is None:
                high_con_pro_t = target_prototype_without_lbt(loader_s_no_shuffle, unlabeled_loader_t,
                                                              feature_extractor,
                                                              feature_extractor.inter_size, classifier,
                                                              CLASS_NUM, K_NUM)
            else:
                high_con_pro_t = target_prototype_with_lbt(loader_s_no_shuffle, unlabeled_loader_t,
                                                           labeled_loader_t,
                                                           feature_extractor, feature_extractor.inter_size,
                                                           classifier,
                                                           CLASS_NUM, K_NUM)

            len_data_loader = len(test_dataset)
            all_indices = torch.arange(0, len_data_loader)
            idx, already_selected_group = query_selection(feature_extractor,
                                                           feature_extractor.inter_size,
                                                           classifier, high_con_pro_t, test_loader,
                                                           already_selected_group, args.eta)
            print('The sample index selected in this round of annotation: ' + '、'.join(map(str, idx)))
            '''Displays which classes the selected samples belong to'''
            analyze(idx, test_dataset)

            selected_idx = np.append(selected_idx, idx)
            labeled_dataset_t = torch.utils.data.Subset(test_dataset, torch.from_numpy(selected_idx).long())
            labeled_loader_t = get_loader(labeled_dataset_t, BATCH_SIZE, shu=True, dl=False)

            unlabeled_dataset_t = torch.utils.data.Subset(test_dataset, torch.from_numpy(
                np.setdiff1d(all_indices.numpy(), selected_idx)).long())
            unlabeled_loader_t = DataLoader(unlabeled_dataset_t, batch_size=BATCH_SIZE, shuffle=False,
                                            drop_last=False)
            train_loader_t = DataLoader(unlabeled_dataset_t, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        if ep <= 39:
            train_without_tlb(ep, train_loader_s, train_loader_t, loader_s_no_shuffle, unlabeled_loader_t)
        elif 39 < ep <= 75:
            train_with_tlb(train_loader_s, train_loader_t, labeled_loader_t,
                           loader_s_no_shuffle, unlabeled_loader_t)
        else:
            balance_self_training(train_loader_s, train_loader_t, labeled_loader_t,
                                  loader_s_no_shuffle, unlabeled_loader_t, args.rho, args.lambda_, args.gamma)

    test_acc = test(test_loader)
    torch.save(feature_extractor.state_dict(),
               'checkpoints/houston/' + str(seeds[iDataSet])
               + '_g.pth')
    torch.save(classifier.state_dict(),
               'checkpoints/houston/' + str(seeds[iDataSet])
               + '_f.pth')

print(acc)
AA = np.mean(A, 1)
AAMean = np.mean(AA, 0)
AAStd = np.std(AA)
AMean = np.mean(A, 0)
AStd = np.std(A, 0)
OAMean = np.mean(acc)
OAStd = np.std(acc)
kMean = np.mean(k)
kStd = np.std(k)

print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
print("accuracy for each class: ")
for i in range(CLASS_NUM):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))
