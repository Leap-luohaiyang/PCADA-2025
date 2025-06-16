import numpy as np
import scipy.io as sio
from sklearn import preprocessing


def load_data_pavia(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    data_all = image_data[data_key]
    GroundTruth = label_data[label_key]

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))
    data_scaler = preprocessing.scale(data)
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    print(np.max(Data_Band_Scaler), np.min(Data_Band_Scaler))

    return Data_Band_Scaler, GroundTruth


def load_data_houston(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    data_all = image_data['ori_data']
    GroundTruth = label_data['map']

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))
    data_scaler = preprocessing.scale(data)
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    return Data_Band_Scaler, GroundTruth


def load_data_sh_hz(file_path):
    total = sio.loadmat(file_path)

    data1 = total['DataCube1']  # Shanghai
    data2 = total['DataCube2']  # Hangzhou
    gt1 = total['gt1']
    gt2 = total['gt2']

    data_s = data1.reshape(np.prod(data1.shape[:2]), np.prod(data1.shape[2:]))
    data_scaler_s = preprocessing.scale(data_s)
    Data_Band_Scaler_s = data_scaler_s.reshape(data1.shape[0], data1.shape[1], data1.shape[2])

    data_t = data2.reshape(np.prod(data2.shape[:2]), np.prod(data2.shape[2:]))
    data_scaler_t = preprocessing.scale(data_t)
    Data_Band_Scaler_t = data_scaler_t.reshape(data2.shape[0], data2.shape[1], data2.shape[2])

    return Data_Band_Scaler_s, Data_Band_Scaler_t, gt1, gt2


def get_source_data(source_data, source_label, HalfWidth, num_per_class):
    print('get_source_data() run...')
    print('The original source data shape:', source_data.shape)
    nBand = source_data.shape[2]

    data = np.pad(source_data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')
    label = np.pad(source_label, HalfWidth, mode='constant')

    train = {}
    train_indices = []
    [Row, Column] = np.nonzero(label)
    m = int(np.max(label))
    print(f'num_class : {m}')

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if label[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        train[i] = indices[:num_per_class]

    for i in range(m):
        train_indices += train[i]
    np.random.shuffle(train_indices)

    # train
    print('the number of processed data:', len(train_indices))
    nTrain = len(train_indices)
    index = np.zeros([nTrain], dtype=np.int64)
    # processed_data = np.zeros([nTrain, 2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand], dtype=np.float32)
    processed_data = np.zeros([nTrain, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    processed_label = np.zeros([nTrain], dtype=np.int64)
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)

    for i in range(nTrain):
        index[i] = i
        # processed_data[i, :, :, :] = np.array(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
        #                                       Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1, :])
        processed_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                                  Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1,
                                                  :], (2, 0, 1))
        processed_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    processed_label = processed_label - 1

    print('sample data shape', processed_data.shape)
    print('sample label shape', processed_label.shape)
    print('get_sample_data() end...')
    return processed_data, processed_label


def get_target_data(target_data, target_label, HalfWidth):
    print('get_target_data() run...')
    print('The original data shape:', target_data.shape)
    nBand = target_data.shape[2]

    data = np.pad(target_data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')
    label = np.pad(target_label, HalfWidth, mode='constant')

    train = {}
    train_indices = []
    [Row, Column] = np.nonzero(label)
    num_class = int(np.max(label))
    print(f'num_class : {num_class}')

    for i in range(num_class):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if
                   label[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        train[i] = indices

    for i in range(num_class):
        train_indices += train[i]
    np.random.shuffle(train_indices)

    print('the number of target data:', len(train_indices))
    nTest = len(train_indices)
    processed_data = np.zeros([nTest, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    processed_label = np.zeros([nTest], dtype=np.int64)
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)

    for i in range(nTest):
        processed_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                                  Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1,
                                                  :], (2, 0, 1))
        processed_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    processed_label = processed_label - 1

    print('processed all data shape:', processed_data.shape)
    print('processed all label shape:', processed_label.shape)
    print('get_all_data() end...')

    return processed_data, processed_label, label, RandPerm, Row, Column
