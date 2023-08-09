import torch
from torch import nn
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


def to_img(x):  # 将张量转化为图片
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.reshape(x.size(0), 1, 28, 28)
    return x


class CVAE(nn.Module):
    def __init__(self, feature_size=784, class_size=20, latent_size=64):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(feature_size + class_size, 200)
        self.fc2_mu = nn.Linear(200, latent_size)
        self.fc2_log_std = nn.Linear(200, latent_size)
        self.fc3 = nn.Linear(latent_size + class_size, 200)
        self.fc4 = nn.Linear(200, feature_size)

    def encode(self, x, y):
        h1 = F.relu(self.fc1(torch.cat([x, y], dim=1)))  # 连接图片和条件
        mu = self.fc2_mu(h1)
        log_std = self.fc2_log_std(h1)
        return mu, log_std

    def decode(self, z, y):
        h3 = F.relu(self.fc3(torch.cat([z, y], dim=1)))  # 连接隐变量和条件
        recon = torch.sigmoid(self.fc4(h3))
        return recon

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # 从标准正态分布采样
        z = mu + eps * std
        return z

    def forward(self, x, y):
        mu, log_std = self.encode(x, y)
        z = self.reparametrize(mu, log_std)
        recon = self.decode(z, y)
        return recon, mu, log_std

    def loss_function(self, recon, x, mu, log_std):
        recon_loss = F.mse_loss(recon, x, reduction="sum")
        kl_loss = -0.5 * (1 + 2 * log_std - mu.pow(2) - torch.exp(2 * log_std))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_loss
        return loss


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, recon_x, x, mu, logvar):
        x = x.view(-1, 784)
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


def train(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        img, label = data
        img = img.view(img.size(0), -1).to(device)
        y_one_hot = torch.zeros(label.shape[0], 10).scatter_(1, label.view(label.shape[0], 1), 1).to(device)
        optimizer.zero_grad()
        recon_batch, mean, lg_var = model(img, y_one_hot)
        loss = model.loss_function(recon_batch, img, mean, lg_var)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
    train_loss /= len(train_loader.dataset)
    print('Train Loss: {:.4f}'.format(train_loss))


def test(model, device, epoch):
    model.eval()

    # 生成0到9十个数字的图片
    num_samples = 10
    z_sample = torch.randn(num_samples, 64).to(device)
    y_one_hots = torch.eye(10).to(device)
    with torch.no_grad():
        x_decoded = model.decode(z_sample, y_one_hots)

    # 将图片保存在results文件夹中
    result_dir = 'results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_path = os.path.join(result_dir, f'{epoch + 1}-results.png')
    rel = to_img(x_decoded.cpu().detach())  # 将生成的样本转换为图片
    image_grid = torchvision.utils.make_grid(rel, nrow=num_samples)
    torchvision.utils.save_image(image_grid, result_path)


def main():
    Epoch = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CVAE(class_size=10).to(device)
    print(model)
    learning_rate = 0.01
    # 数据归一化到(-1,1)
    transform = transforms.Compose([
        transforms.ToTensor(),  # 0-1
    ])
    # 加载内置数据集
    train_ds = torchvision.datasets.MNIST('data',
                                          train=True,
                                          transform=transform,
                                          download=True)

    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=128,
                                               shuffle=True)

    test_ds = torchvision.datasets.MNIST('data',
                                         train=False,
                                         transform=transform,
                                         download=True)

    test_loader = torch.utils.data.DataLoader(test_ds,
                                              batch_size=128,
                                              shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练阶段
    for epoch in range(Epoch):
        print("Epoch:", '%04d' % (epoch + 1))
        train(model, train_loader, optimizer, device)
        test(model, device, epoch)


# 运行主函数
if __name__ == '__main__':
    main()
