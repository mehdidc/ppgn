from __future__ import print_function
from collections import defaultdict
from datetime import datetime
from visdom import Visdom

import pandas as pd
import time
import sys
import numpy as np
import argparse
import os
import random
import torch
from torch.nn.init import xavier_uniform
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from torchvision.models import alexnet

from clize import run

class Discr(nn.Module):
    def __init__(self, nc, ndf):
        super(Discr, self).__init__()
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
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1)

class Gen(nn.Module):
    
    def __init__(self, nz):
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

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x =  x.view(x.size(0), 256, 4, 4)
        x = self.conv(x)
        return x


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

def main(*, imageSize=128, dataroot='.', classifier='alexnet', batchSize=128, 
         nThreads=1, niter=100000, lr=0.0002, beta1=0.5, outf='samples',  
         clfImageSize=224, nz=4096):
    
    viz = Visdom('http://romeo163')
    win = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='ppgn, started {}'.format(datetime.now())))

    viz.line(X=np.array([0]), Y=np.array([0]), update='append', win=win)

    transform = transforms.Compose([
        transforms.RandomSizedCrop(imageSize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    #traindir = os.path.join(dataroot, 'train')
    #valdir = os.path.join(dataroot, 'val')
    traindir = os.path.join(dataroot)
    valdir = traindir
    train = datasets.ImageFolder(traindir, transform)
    val = datasets.ImageFolder(valdir, transform)
    dataloader = torch.utils.data.DataLoader(
        train, batch_size=batchSize, shuffle=True, num_workers=nThreads)
   
    if classifier == 'alexnet':
        clf = alexnet(pretrained=True)
    else:
        sys.path.append(os.path.dirname(classifier))
        clf = torch.load(classifier)
    clf.eval()

    clf_mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    clf_mean = clf_mean[np.newaxis, :, np.newaxis, np.newaxis]
    clf_mean = torch.from_numpy(clf_mean)
    clf_mean = Variable(clf_mean)
    clf_mean = clf_mean.cuda()
    clf_std = np.array([0.229, 0.224, 0.225], dtype='float32')
    clf_std = clf_std[np.newaxis, :, np.newaxis, np.newaxis]
    clf_std = torch.from_numpy(clf_std)
    clf_std = Variable(clf_std)
    clf_std = clf_std.cuda()

    def norm(x):
        x = (x + 1) / 2.
        x = x - clf_mean.repeat(x.size(0), 1, x.size(2), x.size(3))
        x = x / clf_std.repeat(x.size(0), 1, x.size(2), x.size(3))
        return x

    def encode(x):
        p = imageSize - clfImageSize
        if p > 0:
            x = x[:, :, p//2:-p//2, p//2:-p//2]
        x = norm(x)
        features = clf.features
        classifier = clf.classifier
        x = features(x)
        x = _flatten(x)
        x = classifier[0](x)
        x = classifier[1](x)
        x = classifier[2](x)
        return x

    def encode2(x):
        p = imageSize - clfImageSize
        if p > 0:
            x = x[:, :, p:-p, p:-p]
        features = clf.features
        x = norm(x)
        x = features(x)
        return x

    netG = Gen(nz=nz)
    netG.apply(weights_init)
 
    netD = Discr(nc=3, ndf=64)
    netD.apply(weights_init)

    input = torch.FloatTensor(batchSize, 3, imageSize, imageSize)
    label = torch.FloatTensor(batchSize)
    input = Variable(input)
    label = Variable(label)
 
    real_label = 1
    fake_label = 0
    criterion = nn.BCELoss()
    #criterion = nn.MSELoss()
    #mse = lambda x,y: 0.5 * ((x - y)**2).view(x.size(0), -1).mean()
    mse = lambda x,y: 0.5 * (((x - y)**2).view(x.size(0), -1).sum()) / x.size(0)

    clf = clf.cuda()
    netG = netG.cuda()
    netD = netD.cuda()
    criterion = criterion.cuda()
    input = input.cuda()
    label = label.cuda()

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas = (beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas = (beta1, 0.999))

    stats = defaultdict(list)
    j = 0
    
    #dataloader_iter = iter(dataloader)
    #data_ = next(dataloader_iter)

    for epoch in range(niter):
        for i, data in enumerate(dataloader):
            #data = data_
            t = time.time()
            netD.zero_grad()
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            input.data.resize_(real_cpu.size()).copy_(real_cpu)
            label.data.resize_(batch_size).fill_(real_label)

            output = netD(input)
            errD_real = mse(output, label)
            errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            hid = encode(input)
            hid = hid.view(batch_size, nz, 1, 1)
            fake = netG(hid)
            label.data.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = mse(output, label)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.data.fill_(real_label) # fake labels are real for generator cost
            output = netD(fake)

            gan_loss = 2. * criterion(output, label)
            pixel_loss =  1e-3 * mse(fake, input)
            feature_loss = 0.01 * mse(encode2(fake), encode2(input))
            rec_loss = 0.01 * mse(encode(fake), encode(input))
            errG = pixel_loss + feature_loss + rec_loss + gan_loss

            stats['iter'].append(j)
            stats['loss'].append(errG.data[0])
            stats['gan_loss'].append(gan_loss.data[0])
            stats['pixel_loss'].append(pixel_loss.data[0])
            stats['feature_loss'].append(feature_loss.data[0])
            stats['rec_loss'].append(rec_loss.data[0])
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()
            delta_t = time.time() - t
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f, time : %.4f'
                  % (epoch, niter, i, len(dataloader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2, delta_t))
            j += 1
            viz.updateTrace(X=np.array([stats['iter'][-1]]), Y=np.array([stats['loss'][-1]]), win=win, name='loss')
            viz.updateTrace(X=np.array([stats['iter'][-1]]), Y=np.array([stats['gan_loss'][-1]]), win=win, name='gan_loss')
            viz.updateTrace(X=np.array([stats['iter'][-1]]), Y=np.array([stats['pixel_loss'][-1]]), win=win, name='pixel_loss')
            viz.updateTrace(X=np.array([stats['iter'][-1]]), Y=np.array([stats['feature_loss'][-1]]), win=win, name='feature_loss')
            viz.updateTrace(X=np.array([stats['iter'][-1]]), Y=np.array([stats['rec_loss'][-1]]), win=win, name='rec_loss')

            if i % 100 == 0:
                pd.DataFrame(stats).to_csv('{}/stats.csv'.format(outf), index=False)
                # the first 64 samples from the mini-batch are saved.
                vutils.save_image((real_cpu[0:64,:,:,:]+1)/2.,
                        '%s/real_samples.png' % outf, nrow=8)
                #fake = netG(hid)
                vutils.save_image((fake.data[0:64,:,:,:]+1)/2.,
                        '%s/fake_samples_epoch_%03d.png' % (outf, epoch), nrow=8)

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))

if __name__ == '__main__':
    run(main)
