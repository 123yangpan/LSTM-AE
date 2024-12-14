
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tordata
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
import os

class Dataset(tordata.Dataset):
    def __init__(self, noisy_data, clean_data=None):
        self.noisy_data = noisy_data
        self.clean_data = clean_data

    def __getitem__(self, index):
        noisy_data = self.noisy_data[index]
        clean_data = -1
        if self.clean_data is not None:
            clean_data = self.clean_data[index]
        return noisy_data, clean_data

    def __len__(self):
        return self.noisy_data.shape[0]

class LSTM_AE(nn.Module):
    def __init__(self):
        super(LSTM_AE, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 11, padding=5)
        self.conv2 = nn.Conv1d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv1d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv1d(64, 32, 3, padding=1)
        self.conv6 = nn.Conv1d(64, 32, 3, padding=1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True)  # LSTM层
        self.conv7 = nn.Conv1d(64, 32, 3, padding=1)
        self.conv8 = nn.Conv1d(64, 32, 3, padding=1)
        self.conv9 = nn.Conv1d(32, 1, 3, padding=1)
        self.pool = nn.MaxPool1d(2, padding=0)
        self.up = nn.Upsample(scale_factor=2)
        self

    def forward(self, x):
        # print(x.shape)
        b_s, f_n = x.size()
        x = x.view(b_s, 1, f_n)
        x1 = x[:, :, :512]
        x2 = x[:, :, 512:]
        # print(x1.shape)
        x1 = self.pool(F.relu(self.conv1(x1)))
        x2 = self.pool(F.relu(self.conv1(x2)))
        y1 = x1
        y2 = x2
        # print(x1.shape)

        x1 = self.pool(F.relu(self.conv2(x1)))
        x2 = self.pool(F.relu(self.conv2(x2)))

        m1 = x1
        m2 = x2
        # print(x1.shape)
        x1 = self.pool(F.relu(self.conv3(x1)))
        x2 = self.pool(F.relu(self.conv3(x2)))
        n1 = x1
        n2 = x2

        x1 = self.pool(F.relu(self.conv4(x1)))
        x2 = self.pool(F.relu(self.conv4(x2)))
        k1 = x1
        k2 = x2
        batch_size, seq_len = x1.size()[0], x1.size()[2]
        h_0 = torch.randn(1, x1.size(0), 32)
        c_0 = torch.randn(1, x1.size(0), 32)
        x1, _ = self.lstm(x1.view(batch_size, seq_len, -1), (h_0.cuda(), c_0.cuda()))

        batch_size, seq_len = x2.size()[0], x2.size()[2]
        h_0 = torch.randn(1, x2.size(0), 32)
        c_0 = torch.randn(1, x2.size(0), 32)
        x2, _ = self.lstm(x2.view(batch_size, seq_len, -1), (h_0.cuda(), c_0.cuda()))

        x1 = self.up(F.relu(self.conv5(torch.cat((x1, k1), dim=1))))
        x2 = self.up(F.relu(self.conv5(torch.cat((x2, k2), dim=1))))

        x1 = self.up(F.relu(self.conv6(torch.cat((x1, n1), dim=1))))
        x2 = self.up(F.relu(self.conv6(torch.cat((x2, n2), dim=1))))

        x1 = self.up(F.relu(self.conv7(torch.cat((x1, m1), dim=1))))
        x2 = self.up(F.relu(self.conv7(torch.cat((x2, m2), dim=1))))

        x1 = self.up(F.relu(self.conv8(torch.cat((x1, y1), dim=1))))
        x2 = self.up(F.relu(self.conv8(torch.cat((x2, y2), dim=1))))

        x1 = self.conv9(x1)
        x2 = self.conv9(x2)

        # print(x1.shape)

        x = torch.cat((x1, x2), dim=2)
        b_s, _, f_n = x.size()
        x = x.view(b_s, -1)
        # print(x.shape)

        return x

if __name__ == '__main__':
    np.random.seed(50)
    subdirs = [x[0] for x in os.walk('Processed_data')]
    for file_leave_out, subdir in enumerate(subdirs[1:]):
        print(subdir)
        sim_path = os.path.join(subdir, 'SimulateData.mat')
        SimulateData = scipy.io.loadmat(sim_path)
        X_train = SimulateData['HRF_train_noised']
        X_train = X_train.reshape((-1, 512))
        n = X_train.shape[0]
        X_train = np.concatenate((X_train[0:int(n / 2), :], X_train[int(n / 2):, :]), axis=1)

        X_val = SimulateData['HRF_val_noised']
        X_val = X_val.reshape((-1, 512))
        n = X_val.shape[0];
        X_val = np.concatenate((X_val[0:int(n / 2), :], X_val[int(n / 2):, :]), axis=1)

        X_test = SimulateData['HRF_test_noised']
        X_test = X_test.reshape((-1, 512))
        n = X_test.shape[0];
        X_test = np.concatenate((X_test[0:int(n / 2), :], X_test[int(n / 2):, :]), axis=1)
        print(X_test.shape)

        Y_train = SimulateData['HRF_train']
        Y_train = Y_train.reshape((-1, 512))
        n = Y_train.shape[0]
        Y_train = np.concatenate((Y_train[0:int(n / 2), :], Y_train[int(n / 2):, :]), axis=1)

        Y_val = SimulateData['HRF_val']
        Y_val = Y_val.reshape((-1, 512))
        n = Y_val.shape[0];
        Y_val = np.concatenate((Y_val[0:int(n / 2), :], Y_val[int(n / 2):, :]), axis=1)

        Y_test = SimulateData['HRF_test']
        Y_test = Y_test.reshape((-1, 512))
        n = Y_test.shape[0];
        Y_test = np.concatenate((Y_test[0:int(n / 2), :], Y_test[int(n / 2):, :]), axis=1)

        RealData = scipy.io.loadmat('/Processed_data/RealData.mat')
        net_input = RealData['net_input']
        dc = net_input['dc']  # (1,8)
        dc_act = net_input['dc_act']  # (1,8)

        X_real = []
        X_real_act = []

        # %%
        Hb = dc[0, 0]

        n = Hb.shape[0] // 512

        HbO = np.transpose(np.squeeze(Hb[:n * 512, 0, :]))
        HbR = np.transpose(np.squeeze(Hb[:n * 512, 1, :]))

        HbO = np.reshape(HbO, (-1, 512))
        HbR = np.reshape(HbR, (-1, 512))

        X = np.concatenate((HbO, HbR), axis=1)
        X_real.append(X)

        Hb_act = dc_act[0, 0]
        n = Hb.shape[0] // 512

        HbO_act = np.transpose(np.squeeze(Hb_act[:n * 512, 0, :]))
        HbR_act = np.transpose(np.squeeze(Hb_act[:n * 512, 1, :]))

        HbO_act = np.reshape(HbO_act, (-1, 512))
        HbR_act = np.reshape(HbR_act, (-1, 512))

        X = np.concatenate((HbO_act, HbR_act), axis=1)
        X_real_act.append(X)

        X_real = np.concatenate(X_real, axis=0)
        X_real_act = np.concatenate(X_real_act, axis=0)

        # %% double check the magnitude
        X_train = X_train * 1000000
        X_val = X_val * 1000000
        X_test = X_test * 1000000

        Y_train = Y_train * 1000000
        Y_val = Y_val * 1000000
        Y_test = Y_test * 1000000

        X_real = X_real * 1000000
        X_real_act = X_real_act * 1000000

        X_train = X_train[:, :]
        Y_train = Y_train[:, :]
        X_val = X_val[:, :]
        Y_val = Y_val[:, :]
        X_test = X_test[:, :]
        Y_test = Y_test[:, :]
        X_real = X_real[:, :]
        X_real_act = X_real_act[:, :]

        train_set = Dataset(X_train, Y_train)
        val_set = Dataset(X_val, Y_val)
        test_set = Dataset(X_test, Y_test)
        real_set = Dataset(X_real)
        real_set_act = Dataset(X_real_act)

        # %% define data loaders
        trainloader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=512,
            sampler=tordata.RandomSampler(train_set),
            num_workers=2)

        valloader = torch.utils.data.DataLoader(
            dataset=val_set,
            batch_size=512,
            sampler=tordata.SequentialSampler(val_set),
            num_workers=2)

        testloader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=512,
            sampler=tordata.SequentialSampler(test_set),
            num_workers=2)

        realloader = torch.utils.data.DataLoader(
            dataset=real_set,
            batch_size=32,
            sampler=tordata.SequentialSampler(real_set),
            num_workers=2)

        realloader_act = torch.utils.data.DataLoader(
            dataset=real_set_act,
            batch_size=32,
            sampler=tordata.SequentialSampler(real_set_act),
            num_workers=2)
        # %% trian and validate
        data_loaders = {"train": trainloader, "val": valloader}

        # model = ['autoencoder']
        model_name = 'LSTM_AE'
        n_epochs = 100
        print('start')

        net = LSTM_AE().cuda()
        train_loss = []
        val_loss = []
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
        lowest_val_loss = 1e6;
        hdf5_filepath = os.path.join('/Processed_data/working', "autoencoder")

        # hdf5_filepath = "networks/8layers"
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            print('Epoch {}/{}'.format(epoch, n_epochs - 1))
            print('-' * 10)
            for phase in ['train', 'val']:
                if phase == 'train':
                    net.train()  # Set model to training mode
                else:
                    net.eval()
                running_loss = 0.0
                for i, data in enumerate(data_loaders[phase], 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, y_true = data
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    outputs = net(inputs.cuda().float())
                    outputs = outputs.double()
                    mse_loss = nn.MSELoss()
                    loss = mse_loss(outputs, y_true.cuda())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item()
                epoch_loss = running_loss / len(data_loaders[phase])
                if phase == 'train':
                    train_loss.append(epoch_loss)
                else:
                    val_loss.append(epoch_loss)
                    if epoch_loss < lowest_val_loss:
                        lowest_val_loss = epoch_loss
                        torch.save(net.state_dict(), hdf5_filepath)
                         scheduler.step()
        print('Finished Training')

        subdir = '/Processed_data/working'

        Y_test = []
        for i, data in enumerate(testloader, 0):
            inputs = data[0]
            outputs = net(inputs.cuda().float())
            Y_test.append(outputs.cpu().data.numpy())
        Y_test = np.concatenate(Y_test, axis=0)
        Y_test = Y_test / 1000000
        savefilepath = os.path.join(subdir, "Test_NN" + model_name + ".mat")
        scipy.io.savemat(savefilepath, {'Y_test': Y_test})

        print('Y_test shape is ', Y_test.shape)
        plt.figure()
        X_test_example = X_test[21, :];
        Y_test_example = Y_test[21, :] * 1000000;
        X, = plt.plot(X_test_example, 'b')
        Y, = plt.plot(Y_test_example, 'r')
        figurepath = os.path.join(subdir, "Y_test_example.png")
        plt.savefig(figurepath, transparent=True)

        Y_real = []
        for i, data in enumerate(realloader, 0):
            inputs = data[0]
            outputs = net(inputs.cuda().float())
            Y_real.append(outputs.cpu().data.numpy())
        Y_real = np.concatenate(Y_real, axis=0)
        Y_real = Y_real / 1000000
        savefilepath = os.path.join(subdir, "Real_NN" + model_name + ".mat")
        scipy.io.savemat(savefilepath, {'Y_real': Y_real})

        print('Y_real shape is ', Y_real.shape)

        plt.figure()
        X_real_example = X_real[20, :]
        Y_real_example = Y_real[20, :] * 1000000
        X, = plt.plot(X_real_example, 'b')
        Y, = plt.plot(Y_real_example, 'r')
        figurepath = os.path.join(subdir, "Y_real_example.png")
        plt.savefig(figurepath, transparent=True)
