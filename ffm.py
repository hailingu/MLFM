import os
import torch
import numpy as np

BASEDIR = os.getcwd()

features = []
fields = []
values = []
y_train = []
field_cnt = -1
feature_cnt = -1
with open(BASEDIR + '/assets/datasets/criteo_ctr/small_train.txt') as f:
    line = f.readline()
    line = line.strip('\n')
    while line:
        elems = line.split(' ')
        y_train.append(int(elems[0]))
        tmp_feature_idx = []
        tmp_field_idx = []
        tmp_feature_value = []
        for i in range(1, len(elems)):
            field, feature, value = elems[i].split(':')
            field_cnt = max(field_cnt, int(field))
            feature_cnt = max(feature_cnt, int(feature))
            tmp_feature_idx.append([0, int(feature)])
            tmp_field_idx.append(int(field))
            tmp_feature_value.append(float(value))
        features.append(tmp_feature_idx)
        fields.append(tmp_field_idx)
        values.append(tmp_feature_value)
        line = f.readline()

device = torch.device('cpu')
dtype = torch.double

X_train = []
for feature, field, value in zip(features, fields, values):
    feature.append([0, feature_cnt])
    value.append(0.0)
    X_train.append({'feature': feature, 'value': value, 'field': field})

INPUT_DIMENSION, OUTPUT_DIMENSION = feature_cnt + 1, 1
w = torch.rand(INPUT_DIMENSION, OUTPUT_DIMENSION, device=device, dtype=dtype, requires_grad=True)
k = 5
cv = torch.rand(feature_cnt + 1, field_cnt + 1, k, device=device, dtype=dtype, requires_grad=True)

LEARNING_RATE = 1e-1

EPOCH = 3
PRINT_STEP = EPOCH / 3
N = len(y_train)

BATCH_SIZE = 8
start = 0
end = start + BATCH_SIZE

for epoch in range(EPOCH):
    start = 0
    while start < N:
        if end >= N:
            end = N

        X_batch = torch.empty(feature_cnt + 1, BATCH_SIZE, dtype=torch.double)
        y_batch = torch.from_numpy(np.array(y_train[start:end], np.double)).reshape(-1, BATCH_SIZE)
        y_batch[y_batch == 0] = 1e-3
        y_batch[y_batch == 1] = 1 - 1e-3
        for idx in range(BATCH_SIZE):
            i = torch.LongTensor(X_train[start:end][idx]['feature'])
            v = torch.DoubleTensor(X_train[start:end][idx]['value'])
            X_batch[:, idx] = torch.sparse.DoubleTensor(i.t(), v).to_dense()

        linear_part = w.T.mm(X_batch)
        cross_part = torch.zeros(1, BATCH_SIZE, dtype=torch.double, requires_grad=False)

        for idx in range(BATCH_SIZE):
            x = X_train[start:end][idx]
            for f1 in range(0, len(x['field']) - 1):
                for f2 in range(f1 + 1, len(x['field'])):
                    f1_feature = x['feature'][f1][1]
                    f2_feature = x['feature'][f2][1]

                    f1_field = x['field'][f1]
                    f2_field = x['field'][f2]

                    factor = cv[f1_feature, f2_field, :].mul(cv[f2_feature, f1_field, :])
                    cross_part[0, idx] += factor.sum() * x['value'][f1] * x['value'][f2]
        y_hat = linear_part + cross_part
        y_hat[y_hat > 7] = 7
        y_hat = 1.0 / (1.0 + torch.exp(-1 * y_hat))

        logloss = -1 * torch.sum(
            torch.mul(y_hat, torch.log(y_batch)) + torch.mul((1 - y_hat), torch.log(1 - y_batch))) / BATCH_SIZE
        logloss.backward()

        with torch.no_grad():
            w -= LEARNING_RATE * w.grad
            cv -= LEARNING_RATE * cv.grad

            # Manually zero the gradients after updating weights
            w.grad.zero_()
            cv.grad.zero_()

        start = end
        end = start + BATCH_SIZE

    if epoch % PRINT_STEP == 0:
        print('EPOCH: %d, loss: %f' % (epoch, logloss))
