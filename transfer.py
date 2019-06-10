import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


BATCH_SIZE = 220
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator
N_IDEAS = 2             # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 2     # it could be total point G can draw in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])




def artist_works():     # painting from the famous artist (real target)
    outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    normals = 0.5 * np.random.randn(100, 2)
    a = 0.9 * np.random.randn(100, 2)
    data = np.r_[normals + 2, normals - 2]
    # data = np.append(data,a+6,axis=0)
    data = np.append(data, outliers, axis=0)
    paintings = torch.from_numpy(data).float()

    return paintings

def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss

def CORAL_loss(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 1, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm)

    # target covariance
    xmt = torch.mean(target, 1, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)
    # frobenius norm between source and target
    loss = (xc - xct).pow(2).sum().sqrt()
    loss = loss/(4*d*d)
    return loss

def kl(p,q):
    kl1 = torch.sum(p * torch.log(p / q))
    return kl1

G = nn.Sequential(                      # Generator
    nn.Linear(N_IDEAS, 128),            # random ideas (could from normal distribution)
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS),     # making a painting from these random ideas
)

D = nn.Sequential(                      # Discriminator
    nn.Linear(ART_COMPONENTS, 128),     # receive art work either from the famous artist or a newbie like G
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),                       # tell the probability that the art work is made by artist
)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

# plt.ion()   # something about continuous plotting


artist_paintings = artist_works()  # real painting from artist
criterion = nn.BCELoss()
for step in range(10000):
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)  # random ideas
    G_paintings = G(G_ideas)                    # fake painting from G (random ideas)

    prob_artist0 = D(artist_paintings)          # D try to increase this prob
    prob_artist1 = D(G_paintings)               # D try to reduce this prob

    real_label = torch.ones(artist_paintings.size()[0])
    fake_label = torch.zeros(G_ideas.size()[0])

    # MMD loss
    compare = G(artist_paintings)

    mmd_loss = linear_mmd2(compare,G_paintings)

    # D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    # G_loss = torch.mean(torch.log(1. - prob_artist1)) + mmd_loss

    D_loss = criterion(prob_artist0,real_label) + criterion(prob_artist1,fake_label)
    G_loss = criterion(prob_artist1,fake_label) + mmd_loss

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)      # reusing computational graph
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()
    print(G_loss)

#     if step % 1000 == 0:  # plotting
#         plt.cla()
#         plt.ion()
#         for i in range(len(artist_paintings)):
#             plt.scatter(compare[i].detach().numpy()[0],compare[i].detach().numpy()[1],c='#4AD631', lw=3, label='raw data')
#             # plt.scatter(G_paintings[i].detach().numpy()[0], G_paintings[i].detach().numpy()[1], c='#FF9359', lw=3, label='generated data')
#             plt.scatter(G_paintings[i].detach().numpy()[0], G_paintings[i].detach().numpy()[1], c='#FF9359', lw=3, label='generated data')
#
#         plt.text(1, 3.5, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
#         plt.text(1, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
#         # plt.ylim((0, 3));plt.legend(loc='upper right', fontsize=10)
#         # plt.draw()
#         plt.pause(0.01)
#
# plt.ioff()
# plt.show()

result = D(artist_paintings)
print(result)
for i in range(len(artist_paintings)):
    plt.subplot(121).scatter(artist_paintings[i].detach().numpy()[0],artist_paintings[i].detach().numpy()[1],c='#4AD631', lw=3, label='result data')

plt.show()

