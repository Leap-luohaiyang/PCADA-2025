import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from datasets.data_pre import *
from models.classifier import SingleClassifier
from models.dcrn import DCRN


def classification_map(map, dpi, savePath):
    fig = plt.figure(figsize=(map.shape[1] / dpi, map.shape[0] / dpi), dpi=dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(savePath, dpi=dpi, bbox_inches='tight', pad_inches=0)

    plt.close(fig)


NUM_BAND = 198
PATCH_SIZE = 1
CLASS_NUM = 3
HALF_WIDTH = 0
BATCH_SIZE = 32

file_path = 'data/Shanghai-Hangzhou/DataCube.mat'

source_data, target_data, source_label, target_label = load_data_sh_hz(file_path)
test_data, test_label, test_gt, RandPerm, Row, Column = get_target_data(target_data, target_label, HALF_WIDTH)
test_dataset = TensorDataset(torch.tensor(test_data), torch.tensor(test_label))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

feature_extractor = DCRN(input_channels=NUM_BAND, patch_size=PATCH_SIZE, n_classes=CLASS_NUM).cuda()
classifier = SingleClassifier().cuda()
feature_extractor.load_state_dict(torch.load('checkpoints/shanghai-hangzhou/feature_extractor.pth'))
classifier.load_state_dict(torch.load('checkpoints/shanghai-hangzhou/classifier.pth'))
feature_extractor.eval()
classifier.eval()

pred_all = []

with torch.no_grad():
    for data in test_loader:
        img, label = data
        img, label = img.cuda(), label.cuda()
        fea = feature_extractor(img)
        output = classifier(fea)
        pred = output.data.max(1)[1]
        pred_all = np.concatenate([pred_all, pred.cpu().numpy()])

for i in range(len(pred_all)):
    test_gt[Row[RandPerm[i]]][Column[RandPerm[i]]] = pred_all[i] + 1

hsi_pic = np.zeros((test_gt.shape[0], test_gt.shape[1], 3))
for i in range(test_gt.shape[0]):
    for j in range(test_gt.shape[1]):
        if test_gt[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if test_gt[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if test_gt[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if test_gt[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if test_gt[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if test_gt[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if test_gt[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if test_gt[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]

classification_map(hsi_pic[:, :, :], 24,
                   "classification_map/shanghai-hangzhou_map.png")
