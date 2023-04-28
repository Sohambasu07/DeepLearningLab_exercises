import torch
import torch.nn.functional as F
from agent.networks import CNN
from torchsummary import summary

class BCAgent:
    
    def __init__(self, lr=0.001, hist_len = 1, n_classes = 4):
        # TODO: Define network, loss function, optimizer
        # self.net = CNN(...)
        self.lr = lr
        self.history_length = hist_len
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.net = CNN(self.history_length, n_classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = self.lr)
        summary(self.net, (self.history_length, 96, 96))
        torch.onnx.export(self.net, torch.randn(1, hist_len, 96, 96, device = self.device), "./figs/ImitationBig.onnx", verbose=False)

    def update(self, X_batch, y_batch, grad = True):
        # TODO: transform input to tensors
        # TODO: forward + backward + optimize
        x = torch.Tensor(X_batch).to(self.device)
        y = torch.Tensor(y_batch).type(torch.LongTensor).to(self.device)

        outputs = self.net(x)
        loss = self.criterion(outputs, y)
        if grad:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        outputs = torch.argmax(outputs, dim=1)
        acc = torch.sum(outputs == y).float()/len(y)
        return loss.item(), acc.item()

    def predict(self, X):
        # TODO: forward pass
        X = torch.Tensor(X).to(self.device)
        outputs = self.net(X)
        return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))


    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)

