import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# ==========================================
# 1. Configurações e Hiperparâmetros
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  # Taxa de aprendizado padrão para DCGAN
BATCH_SIZE = 128
IMAGE_SIZE = 64       # Redimensionamos MNIST (28x28) para 64x64 para facilitar a arquitetura
CHANNELS_IMG = 1      # 1 canal (preto e branco)
Z_DIM = 100           # Tamanho do vetor de ruído (latente)
NUM_EPOCHS = 5        # Aumente para 20-50 para resultados perfeitos
FEATURES_DISC = 64    # Tamanho base dos canais do discriminador
FEATURES_GEN = 64     # Tamanho base dos canais do gerador

# Cria pasta para salvar os resultados
os.makedirs("generated_images", exist_ok=True)

print(f"Usando dispositivo: {device}")

# ==========================================
# 2. Definição das Redes (Discriminador e Gerador)
# ==========================================

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Entrada: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # Bloco: 64 -> 128
            self._block(features_d, features_d * 2, 4, 2, 1),
            # Bloco: 128 -> 256
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            # Bloco: 256 -> 512
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # Saída: 512 -> 1 (Probabilidade de ser real)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Entrada: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),   # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),   # img: 32x32
            # Camada final para gerar a imagem 64x64
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Tanh mapeia a saída para o intervalo [-1, 1]
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)

# ==========================================
# 3. Inicialização de Pesos
# ==========================================
# Para DCGAN, inicializar pesos com média 0 e std 0.02 ajuda na estabilidade
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# ==========================================
# 4. Preparação de Dados e Modelos
# ==========================================

# Transformações: Resize para 64x64 e Normalização para [-1, 1]
transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
])

# Baixando o MNIST
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Instanciando modelos
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weights(gen)
initialize_weights(disc)

# Otimizadores e Função de Perda
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Ruído fixo para visualizar a evolução das MESMAS "sementes" ao longo do tempo
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

# ==========================================
# 5. Loop de Treinamento
# ==========================================
print("Iniciando treinamento... Isso pode demorar se não tiver GPU.")

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
        fake = gen(noise)

        ### Treinar Discriminador: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Treinar Gerador: max log(D(G(z)))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Printar logs a cada 100 batches
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
            )

    # Ao final de cada época, salvar imagens geradas
    with torch.no_grad():
        fake = gen(fixed_noise)
        # Denormalizar de [-1, 1] para [0, 1] para visualização
        img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
        img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

        torchvision.utils.save_image(img_grid_fake, f"generated_images/epoch_{epoch}_fake.png")
        torchvision.utils.save_image(img_grid_real, f"generated_images/epoch_{epoch}_real.png")
        print(f"-> Imagens salvas em 'generated_images/epoch_{epoch}_fake.png'")

print("Treinamento concluído!")