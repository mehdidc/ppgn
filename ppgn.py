from __future__ import print_function
import numpy as np
import torch
from collections import defaultdict
import math
from skimage.io import imsave
import pandas as pd
import time
import sys
import os
from torch.nn.init import xavier_uniform
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable


from torchvision.models import alexnet

from machinedesign.viz import grid_of_images_default

from clize import run

def compute_objectness(v):
    marginal = v.mean(dim=0)
    marginal = marginal.repeat(v.size(0), 1)
    score = v * torch.log(v / (marginal))
    score = score.sum(dim=1).mean()
    score = math.exp(score)
    return score

class Discr(nn.Module):
    def __init__(self, nc, ndf, imageSize=256):
        super(Discr, self).__init__()
        if imageSize == 256:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                #nn.Sigmoid()
            )
        elif imageSize == 128:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                #nn.Sigmoid()
            )

        elif imageSize == 64:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                #nn.Sigmoid()
            )


    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1)

class Gen(nn.Module):
    
    def __init__(self, nz, imageSize=256):
        super(Gen, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(nz, 4096), #defc7
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.3),
            
            nn.Linear(4096, 4096), #defc6
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.3),

            nn.Linear(4096, 4096), #defc5
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.3),
        )
        if imageSize == 256:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False), #deconv5
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.3),

                nn.Conv2d(256, 512, 3, 1, 1, bias=False), #conv5_1
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.3),

                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), #decon4
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.3),
                nn.Conv2d(256, 256, 3, 1, 1, bias=False), #conv4_1
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.3),
                
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), #deconv3
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.3),
                nn.Conv2d(128, 128, 3, 1, 1, bias=False), #conv3_1
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.3),
             
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), #deconv2
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.3),

                nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False), #deconv1
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.3),
                
                nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False), #deconv0
                nn.Tanh()
            )
        elif imageSize == 128:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose2d(256, 256, 3, 1, 1, bias=False), #deconv5
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.3),

                nn.Upsample(scale_factor=2),
                nn.Conv2d(256, 512, 3, 1, 1, bias=False), #conv5_1
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.3),

                nn.Upsample(scale_factor=2),
                nn.Conv2d(512, 256, 3, 1, 1, bias=False), #decon4
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.3),

                nn.Upsample(scale_factor=2),
                nn.Conv2d(256, 128, 3, 1, 1, bias=False), #deconv3
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.3),

                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, 1, 1, bias=False), #deconv3
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.3),

                nn.ConvTranspose2d(64, 3, 3, 1, 1, bias=False), #deconv3
                nn.Tanh()
            )
        elif imageSize == 64:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False), #deconv5
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.3),

                nn.Conv2d(256, 512, 3, 1, 1, bias=False), #conv5_1
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.3),

                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), #decon4
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.3),
                nn.Conv2d(256, 256, 3, 1, 1, bias=False), #conv4_1
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.3),
                
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), #deconv3
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.3),
                nn.Conv2d(128, 128, 3, 1, 1, bias=False), #conv3_1
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.3),
             
                nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False), #deconv0
                nn.Tanh()
            )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x =  x.view(x.size(0), 256, 4, 4)
        x = self.conv(x)
        return x

class PPGen(nn.Module):
    def __init__(self, nz=4096):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(nz,  4096), #defc7
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.3),
            nn.Linear(4096, 4096), #defc6
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.3),
            nn.Linear(4096, 4096), #defc5
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.3),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False), #deconv5
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.3),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), #conv5_1
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.3),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), #decon4
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.3),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False), #conv4_1
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.3),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), #deconv3
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.3),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False), #conv3_1
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.3),
         
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), #deconv2
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False), #deconv1
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.3),
            
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False), #deconv0
            nn.Tanh(),
        )

    def forward(self, input):
        x = input.view(input.size(0), input.size(1))
        x = self.fc(x)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.conv(x)
        return x


class PPDiscr(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 7 * 7)
        x = self.classifier(x)
        return x[:, 0]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        xavier_uniform(m.weight.data)
        m.bias.data.zero_()

def _flatten(x):
    return x.view(x.size(0), -1)

Generator = PPGen

class Encoder(nn.Module):

    def __init__(self, num_classes=1000):
        super(Encoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.image_mean = None
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class Enc:

    def __init__(self, clf, mu, std, image_size=224):
        self.clf = clf
        self.mu = mu
        self.std = std
        self.clfImageSize = image_size

    def norm(self, x):
        x = (x + 1) / 2.
        x = x - self.mu.repeat(x.size(0), 1, x.size(2), x.size(3))
        x = x / self.std.repeat(x.size(0), 1, x.size(2), x.size(3))
        return x
    
    def prep(self, x):
        if x.size(2) != 256:
            x = torch.nn.UpsamplingBilinear2d(scale_factor=256//x.size(2))(x)
        p = 256 - self.clfImageSize
        if p > 0:
            x = x[:, :, p//2:-p//2, p//2:-p//2]
        x = self.norm(x)
        return x

    def classify(self, x):
        features = self.clf.features
        classifier = self.clf.classifier
        x = self.prep(x)
        x = features(x)
        x = _flatten(x)
        y = classifier(x)
        return y 
 
    def encode(self, x):
        features = self.clf.features
        classifier = self.clf.classifier
        x = self.prep(x)
        x = features(x)
        x = _flatten(x)
        x = classifier[0](x)
        x = classifier[1](x)
        x = classifier[2](x)
        return x

    def encode2(self, x):
        features = self.clf.features
        x = self.prep(x)
        x = features(x)
        x = _flatten(x)
        return x



def train(*, 
         imageSize=256, 
         cropSize=256,
         crop_type='center',
         dataroot='/home/mcherti/work/data/shoes/ut-zap50k-images', 
         classifier='alexnet', 
         batchSize=64, 
         nThreads=1, 
         distributed=False,
         clfImageSize=224, 
         nz=4096,
         extra_noise=0,
         niter=100000,
         outf='out',
         resume=False,
         wasserstein=False):
    lr = 0.0002
    beta1 = 0.5
    gan_loss_coef = 0.001
    pixel_loss_coef = 0.1
    feature_loss_coef = 0.1
    rec_loss_coef = 0.1
    tf = []
    if cropSize > 0:
        if crop_type == 'center':
            tf.append(transforms.CenterCrop(cropSize))
        elif crop_type == 'random':
            tf.append(transforms.RandomCrop(cropSize))
    tf.extend([
        transforms.Scale(imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform = transforms.Compose(tf)
    traindir = os.path.join(dataroot)
    train = datasets.ImageFolder(traindir, transform)
    dataloader = torch.utils.data.DataLoader(
        train, 
        batch_size=batchSize, 
        shuffle=True, 
        num_workers=nThreads
    )
    if classifier == 'alexnet':
        clf = alexnet(pretrained=True)
    elif classifier == 'ppgn':
        clf = torch.load('/home/mcherti/work/code/external/ppgn/encoder.th')
    else:
        sys.path.append(os.path.dirname(classifier))
        clf = torch.load(classifier)
    clf.eval()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    clf_mean = Variable(torch.FloatTensor(mean).view(1, -1, 1, 1)).cuda()
    clf_std = Variable(torch.FloatTensor(std).view(1, -1, 1, 1)).cuda()

    enc = Enc(clf, clf_mean, clf_std)
    torch.save(enc, '{}/netE.pth'.format(outf))

    if resume:
        netG = torch.load('{}/netG.pth'.format(outf))
        netD = torch.load('{}/netD.pth'.format(outf))
    else:
        netG = Gen(nz=4096 + extra_noise, imageSize=imageSize)
        netG.apply(weights_init)
        netD = Discr(nc=3, ndf=64, imageSize=imageSize)
        netD.apply(weights_init)
    
    if distributed:
        netG = torch.nn.parallel.DataParallel(netG)
        netD = torch.nn.parallel.DataParallel(netD)

    input = torch.FloatTensor(batchSize, 3, imageSize, imageSize)
    label = torch.FloatTensor(batchSize)
    input = Variable(input)
    label = Variable(label)
    if extra_noise:
        noise = torch.FloatTensor(batchSize, extra_noise, 1, 1)
        noise = Variable(noise).cuda()
        fixed_noise = torch.FloatTensor(batchSize, extra_noise, 1, 1)
        fixed_noise.normal_(0, 1)
        fixed_noise = Variable(fixed_noise).cuda()
    mse = lambda x,y: 0.5 * ((x - y)**2).view(x.size(0), -1).mean()
    if wasserstein:
        real_label = 1
        fake_label = -1
        criterion = lambda output, label:(output*label).mean()
    else:
        real_label = 1
        fake_label = 0
        criterion = nn.BCELoss()
    clf = clf.cuda()
    netG = netG.cuda()
    netD = netD.cuda()
    input = input.cuda()
    label = label.cuda()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas = (beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas = (beta1, 0.999))
    if resume:
        stats = pd.read_csv('{}/stats.csv'.format(outf))
        stats = [(col, list(stats[col])) for col in stats.columns]
        stats = dict(stats)
        start_epoch = max(stats['iter']) // len(dataloader)
        nb_updates = max(stats['iter'])
    else:
        stats = defaultdict(list)
        start_epoch = 0
        nb_updates = 0
    print(netG)
    for epoch in range(start_epoch, start_epoch + niter):
        for i, data in enumerate(dataloader):
            if wasserstein:
                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(-0.01, 0.01)
            t = time.time()
            netD.zero_grad()
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            input.data.resize_(real_cpu.size()).copy_(real_cpu)
            label.data.resize_(batch_size).fill_(real_label)
            if extra_noise:
                noise.data.resize_(batch_size, extra_noise, 1, 1).normal_(0, 1)
            output = netD(input)
            if not wasserstein: output = nn.Sigmoid()(output)
            errD_real = criterion(output, label)
            errD_real.backward()

            # train with fake
            hid = enc.encode(input)
            hid = hid.view(batch_size, nz, 1, 1)
            if extra_noise:
                hid = torch.cat((hid, noise), 1)
            fake = netG(hid)
            label.data.fill_(fake_label)
            output = netD(fake.detach())
            if not wasserstein: output = nn.Sigmoid()(output)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.data.fill_(real_label) # fake labels are real for generator cost
            output = netD(fake)
            if not wasserstein: output = nn.Sigmoid()(output)
            gan_loss = criterion(output, label)
            pixel_loss =  mse(fake, input)
            feature_loss = mse(enc.encode2(fake), enc.encode2(input))
            rec_loss = mse(enc.encode(fake), enc.encode(input))
            errG = (pixel_loss_coef * pixel_loss + 
                    feature_loss_coef * feature_loss + 
                    rec_loss_coef * rec_loss + 
                    gan_loss_coef * gan_loss)
            y = enc.classify(fake)
            y = nn.Softmax()(y)
            objectness = compute_objectness(y.data)
            errG.backward()
            optimizerG.step()
            stats['iter'].append(nb_updates)
            stats['objectness'].append(objectness)
            stats['loss'].append(errG.data[0])
            stats['gan_loss'].append(gan_loss.data[0])
            stats['pixel_loss'].append(pixel_loss.data[0])
            stats['feature_loss'].append(feature_loss.data[0])
            stats['rec_loss'].append(rec_loss.data[0])
            stats['delta_t'].append(time.time() - t)
            labels = ('iter', 'objectness', 'loss', 'gan_loss', 
                      'pixel_loss', 'feature_loss', 'rec_loss', 'delta_t')
            p = '  '.join(['{} : {:.6f}'.format(k, stats[k][-1]) for k in labels])
            print(p)
            if nb_updates % 100 == 0:
                pd.DataFrame(stats).to_csv('{}/stats.csv'.format(outf), index=False)
                # the first 64 samples from the mini-batch are saved.
                im = real_cpu.cpu().numpy()
                im = grid_of_images_default(im, normalize=True)
                imsave('{}/real.png'.format(outf), im)
                im = fake.data.cpu().numpy()
                im = grid_of_images_default(im, normalize=True)
                imsave('{}/fake_{:05d}.png'.format(outf, epoch), im)
            nb_updates += 1
        # do checkpointing
        torch.save(netG, '%s/netG.pth' % (outf))
        torch.save(netD, '%s/netD.pth' % (outf))


def generate(*, folder, eps1=1., eps2=0., eps3=0., unit_id=0, nb_iter=100, outf='gen.png'):
    bs = 16
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    clf_mean = Variable(torch.FloatTensor(mean).view(1, -1, 1, 1)).cuda()
    clf_std = Variable(torch.FloatTensor(std).view(1, -1, 1, 1)).cuda()
    clf = alexnet(pretrained=True)
    enc = Enc(clf, clf_mean, clf_std)
    G = torch.load('%s/netG.pth' % folder)
    grads = {}
    def save_grads(g):
        grads['h'] = g
    G.eval()
    H = torch.rand(bs, 4096, 1, 1) 
    H = H.cuda()
    G = G.cuda()
    clf = clf.cuda()
    x = []
    
    for i in range(nb_iter):
        print(i)
        Hvar = Variable(H, requires_grad=True)
        Hvar.register_hook(save_grads)
        X = G(Hvar)
        y = enc.classify(X)
        loss = y[:, unit_id].mean()
        pr = nn.Softmax()(y)
        print(pr[:, unit_id].mean().data[0])
        loss.backward()
        dh = grads['h'].data
        Hrec = enc.encode(X)
        Hrec = Hrec.view(Hrec.size(0), Hrec.size(1), 1, 1)
        H += eps1 * (Hrec.data - H) + eps2 * dh# + eps3 * torch.randn(H.size())
        x.append(X.data.cpu().numpy())
        #H.clamp_(0, 30)
    x = np.array(x)
    shape = (x.shape[0], x.shape[1])
    im = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
    im = grid_of_images_default(im, normalize=True, shape=shape)
    imsave('{}/gen/{}'.format(folder, outf), im)


if __name__ == '__main__':
    run([train, generate])
