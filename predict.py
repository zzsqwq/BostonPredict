#
# Created by Zs on 21-5-1
#

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from torch import nn
import argparse

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """

    score = r2_score(y_true, y_predict)

    return score


class boston:
    def load_data(self, choose_col=[]):
        data = load_boston()

        boston_x = data.data  # x
        if len(choose_col) != 0:
            boston_x = boston_x[:, choose_col]  # 提取 RM PTRATIO LSTAT 列
        boston_y = data.target  # y
        boston_y = boston_y.reshape(-1, 1)

        return boston_x, boston_y

    def split_data(self, x, y, split_size=0.1):
        ss = MinMaxScaler()
        x = ss.fit_transform(x)

        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=split_size)
        train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
        train_y = torch.from_numpy(train_y).type(torch.FloatTensor)
        test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
        test_y = torch.from_numpy(test_y).type(torch.FloatTensor)

        return train_x, train_y, test_x, test_y

    def init_model(self, input_layer=3, learn_rate=0.1, hidden_layer=100):
        self.model = nn.Sequential(
            nn.Linear(input_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, 1)
        )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate)

    def train(self, x, y, epoch=5000):
        self.iter_loss = []
        self.epoch = epoch
        for i in range(epoch):
            # forward
            y_pred = self.model(x)
            # calc loss
            self.loss = self.criterion(y_pred, y)
            if (i % 30 == 0):
                print("第{}次迭代的loss是:{}".format(i, self.loss))
            self.iter_loss.append(self.loss.item())
            # zero grad
            self.optimizer.zero_grad()
            # backward
            self.loss.backward()
            # adjust weights
            self.optimizer.step()

    def calc_loss(self, predict_y, test_y):
        predict_y = torch.from_numpy(predict_y).type(torch.FloatTensor)
        return self.criterion(predict_y, test_y)

    def draw_loss(self):
        x = np.arange(self.epoch)
        y = np.array(self.iter_loss)
        plt.figure()
        plt.plot(x, y)
        plt.title("The loss curve")
        plt.xlabel("iteration step")
        plt.ylabel("loss")
        plt.savefig("Loss_curve.jpg",dpi=400)
        plt.show()

    def predict(self, test_x):
        predict_y = self.model(test_x)
        predict_y = predict_y.detach().numpy()
        return predict_y

    def plot_tf(self, test_y, predict_y):  # 绘制test和predict的图
        x = np.arange(test_y.shape[0])
        y1 = np.array(predict_y)
        y2 = np.array(test_y)
        line1 = plt.scatter(x, y1, c="blue")
        line2 = plt.scatter(x, y2, c='red')
        plt.legend([line1, line2], ["y_predict", "y_groundtruth"])
        plt.title("The curve of predict and groundtruth")
        plt.ylabel("price")
        plt.savefig('predict_groundtruth.png',dpi=400)
        plt.show()

    def save_model(self,model_name='Boston.pt'):
        torch.save(self.model,'Boston.pt')

    def load_model(self,weights_name='Boston.pt',learn_rate=0.1):
        self.model = torch.load(weights_name)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate)

if __name__ == "__main__":

    load_weights = False
    load_cols = []
    input_shape = 13
    if len(load_cols)!=0:
        input_shape=len(load_cols)

    parser = argparse.ArgumentParser()

    parser.add_argument('--weights',type=str,default='Boston.pt',help='inital weights path')
    parser.add_argument('--load_weights',action='store_true', help='load weights or not')

    opt = parser.parse_args()

    bos = boston()
    x, y = bos.load_data(choose_col=load_cols)
    train_x, train_y, test_x, test_y = bos.split_data(x=x, y=y, split_size=0.2)

    if not load_weights:
        bos.init_model(input_layer=input_shape, hidden_layer=1000, learn_rate=0.01)
        bos.train(train_x, train_y, epoch=10000)
        bos.save_model(model_name='Boston1000.pt')
    else:
        bos.load_model(weights_name=opt.weights,learn_rate=0.01)

    #predict_y = bos.predict(test_x)
    #print(predict_y[:5].reshape(1, -1), '\n', test_y[:5].reshape(1, -1))
    #print(bos.calc_loss(predict_y, test_y))
    if not load_weights:
        bos.draw_loss()
    #bos.plot_tf(predict_y, test_y)
    #print(performance_metric(predict_y,test_y))

