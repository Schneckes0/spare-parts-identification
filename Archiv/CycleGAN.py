import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from collections import OrderedDict

# Hyperparameter
lr = 0.0002
batch_size = 1
epochs = 200

# Generator Block
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Die Architektur ist eine U-Net Struktur
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Downsampling
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))

        # Upsampling
        x = self.relu(self.deconv1(x5))
        x = torch.cat((x, x4), 1)
        x = self.relu(self.deconv2(x))
        x = torch.cat((x, x3), 1)
        x = self.relu(self.deconv3(x))
        x = torch.cat((x, x2), 1)
        x = self.relu(self.deconv4(x))
        x = torch.cat((x, x1), 1)
        x = self.sigmoid(self.deconv5(x))

        return x

# Diskriminator Block
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.conv5(x)
        return x

# Verlustfunktionen
def criterion_GAN(predictions, targets, mode='real'):
    if mode == 'real':
        target = torch.ones_like(predictions)
    else:
        target = torch.zeros_like(predictions)
    
    return nn.MSELoss()(predictions, target)

def criterion_cycle(predicted, real):
    return nn.L1Loss()(predicted, real)

# Optimierer
def get_optimizer(models, lr):
    parameters = []
    for model in models:
        parameters += list(model.parameters())  # Alle Parameter beider Modelle zusammenführen
    return optim.Adam(parameters, lr=lr, betas=(0.5, 0.999))

# CycleGAN Modell laden
G_A = Generator().cuda()
G_B = Generator().cuda()
D_A = Discriminator().cuda()
D_B = Discriminator().cuda()

optimizer_G = get_optimizer([G_A, G_B], lr)  # Ändere hier
optimizer_D_A = get_optimizer([D_A], lr)  # Optimierer für D_A
optimizer_D_B = get_optimizer([D_B], lr)  # Optimierer für D_B

# CycleGAN Trainingsschleife
def train_cycle_gan(dataloader_A, dataloader_B):
    best_loss_G = float('inf')  # Setze den besten Verlust anfangs auf unendlich
    best_epoch = 0
    best_G_A = None
    best_G_B = None

    for epoch in range(epochs):
        for i, (real_A, real_B) in enumerate(zip(dataloader_A, dataloader_B)):
            real_A = real_A[0].cuda()
            real_B = real_B[0].cuda()

            # Generator A (CAD -> Real)
            fake_B = G_A(real_A)
            loss_G_A = criterion_GAN(D_B(fake_B), 'real')

            # Generator B (Real -> CAD)
            fake_A = G_B(real_B)
            loss_G_B = criterion_GAN(D_A(fake_A), 'real')

            # Cycle Consistency Loss
            cycle_A = G_B(fake_B)
            cycle_B = G_A(fake_A)
            loss_cycle_A = criterion_cycle(cycle_A, real_A)
            loss_cycle_B = criterion_cycle(cycle_B, real_B)

            # Total Generator Loss
            loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # Diskriminator A
            loss_D_A_real = criterion_GAN(D_A(real_A), 'real')
            loss_D_A_fake = criterion_GAN(D_A(fake_A.detach()), 'fake')
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

            # Diskriminator B
            loss_D_B_real = criterion_GAN(D_B(real_B), 'real')
            loss_D_B_fake = criterion_GAN(D_B(fake_B.detach()), 'fake')
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

            optimizer_D_A.zero_grad()
            loss_D_A.backward()
            optimizer_D_A.step()

            optimizer_D_B.zero_grad()
            loss_D_B.backward()
            optimizer_D_B.step()

            if i % 100 == 0:
                print(f'Epoch {epoch}/{epochs}, Step {i}/{len(dataloader_A)}')
                print(f'Loss G: {loss_G.item()} Loss D_A: {loss_D_A.item()} Loss D_B: {loss_D_B.item()}')

        # Überprüfen, ob dieses Modell besser ist und es als bestes Modell speichern
        if loss_G.item() < best_loss_G:
            best_loss_G = loss_G.item()
            best_epoch = epoch
            best_G_A = G_A.state_dict()
            best_G_B = G_B.state_dict()
            print(f"Best model found at epoch {epoch} with loss {best_loss_G}")

    # Speichere das beste Modell nach der letzten Epoche
    torch.save(best_G_A, 'G_A_best.pth')
    torch.save(best_G_B, 'G_B_best.pth')
    print(f"Best model saved at epoch {best_epoch} with loss {best_loss_G}")

if __name__ == "__main__":

    # Daten laden und vorbereiten
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset_A = torchvision.datasets.ImageFolder(root='C:/Users/kej9ho/Documents/Bilderkennung/data/CycleGAN/CAD', transform=transform)
    dataset_B = torchvision.datasets.ImageFolder(root='C:/Users/kej9ho/Documents/Bilderkennung/data/CycleGAN/Real', transform=transform)

    dataloader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True)
    dataloader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True)

    train_cycle_gan(dataloader_A, dataloader_B)
