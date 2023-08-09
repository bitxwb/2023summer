import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import os


class VAE(nn.Module):
    def __init__(self, z_dim=20):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z_params = self.encoder(x)
        mu, logvar = z_params[:, :self.z_dim], z_params[:, self.z_dim:]
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_data, mu, logvar = model(data)
        recon_loss = criterion(recon_data, data.view(-1, 784))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    avg_loss = train_loss / len(train_loader.dataset)
    return avg_loss


def save_images(images_dict, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for digit in images_dict:
        digits_images = images_dict[digit]
        digit_images = torch.cat(digits_images, dim=0)
        save_path = os.path.join(save_dir, f"epoch_{epoch}_digit_{digit}.png")
        torchvision.utils.save_image(digit_images, save_path, nrow=10)  # 每行显示10张图片


def test(model, test_loader, criterion, device, num_epochs):
    model.eval()
    test_loss = 0
    save_dir = "results"  # 保存目录
    images_dict = {}  # 用于保存每个数字的图片列表

    for epoch in range(1, num_epochs + 1):
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                recon_data, mu, logvar = model(data)
                recon_loss = criterion(recon_data, data.view(-1, 784))
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss
                test_loss += loss.item()

                # 保存每个数字的生成图片
                for digit in range(10):
                    digit_mask = (_ == digit)
                    digit_images = recon_data[digit_mask].view(-1, 1, 28, 28)
                    if digit not in images_dict:
                        images_dict[digit] = []
                    images_dict[digit].append(digit_images)

        save_images(images_dict, save_dir, epoch)
        images_dict = {}  # 清空图片列表，准备保存下一轮生成的图片

    avg_loss = test_loss / len(test_loader.dataset)
    return avg_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 20
num_epochs = 10
learning_rate = 1e-3
BATCH_SIZE = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
model = VAE(z_dim=z_dim).to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss(reduction='sum').to(device)

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    test_loss = test(model, test_loader, criterion, device, num_epochs=num_epochs)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
