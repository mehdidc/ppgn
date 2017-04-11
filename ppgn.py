from __future__ import print_function
from collections import defaultdict
from datetime import datetime
import math
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
               
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), #deconv0
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

class DCGAN(nn.Module):
    def __init__(self, nz, ngf=64, imageSize=256):
        super(DCGAN, self).__init__()
        if imageSize == 256:
            self.conv = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
     
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(    ngf,      3, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )
        elif imageSize == 128:
            self.conv = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
     
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(    ngf,      3, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        elif imageSize == 64:
            self.conv = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(    ngf,      3, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

    def forward(self, input):
        return self.conv(input)

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

def main(*, imageSize=64, dataroot='.', classifier='alexnet', 
         batchSize=128, nThreads=1, clfImageSize=224, nz=4096,
         niter=100):
    outf = '{{folder}}'
    lr = 0.0002
    beta1 = 0.5
    gan_loss_coef = {{'gan_loss_coef' | loguniform(-3, 2) }}
    pixel_loss_coef = {{'pixel_loss_coef' | loguniform(-3, 2) }}
    feature_loss_coef = {{'feature_loss_coef' | loguniform(-3, 2) }}
    rec_loss_coef = {{'rec_loss_coef' | loguniform(-3, 2) }}


    viz = Visdom('http://romeo163')
    win = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='ppgn, started at {}, folder={}'.format(datetime.now(), outf)))

    viz.line(X=np.array([0]), Y=np.array([0]), update='append', win=win)

    transform = transforms.Compose([
        transforms.Scale(imageSize),
        transforms.CenterCrop(imageSize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    traindir = os.path.join(dataroot)
    train = datasets.ImageFolder(traindir, transform)
    dataloader = torch.utils.data.DataLoader(
        train, batch_size=batchSize, shuffle=True, num_workers=nThreads)
   
    if classifier == 'alexnet':
        clf = alexnet(pretrained=True)
    else:
        sys.path.append(os.path.dirname(classifier))
        clf = torch.load(classifier)
    clf.eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    clf_mean = Variable(torch.FloatTensor(mean).view(1, -1, 1, 1)).cuda()
    clf_std = Variable(torch.FloatTensor(std).view(1, -1, 1, 1)).cuda()

    def norm(x):
        x = (x + 1) / 2.
        x = x - clf_mean.repeat(x.size(0), 1, x.size(2), x.size(3))
        x = x / clf_std.repeat(x.size(0), 1, x.size(2), x.size(3))
        return x
    
    def classify(x):
        if x.size(2) != 256:
            x = torch.nn.UpsamplingBilinear2d(scale_factor=256//x.size(2))(x)
        p = 256 - clfImageSize
        if p > 0:
            x = x[:, :, p//2:-p//2, p//2:-p//2]
        x = norm(x)
        features = clf.features
        classifier = clf.classifier
        x = features(x)
        x = _flatten(x)
        y = classifier(x)
        return y 
 
    def encode(x):
        if x.size(2) != 256:
            x = torch.nn.UpsamplingBilinear2d(scale_factor=256//x.size(2))(x)
        p = 256 - clfImageSize
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
        if x.size(2) != 256:
            x = torch.nn.UpsamplingBilinear2d(scale_factor=256//x.size(2))(x)
        p = 256 - clfImageSize
        if p > 0:
            x = x[:, :, p:-p, p:-p]
        features = clf.features
        x = norm(x)
        x = features(x)
        x = _flatten(x)
        return x

    netG = Gen(nz=nz, imageSize=imageSize)
    netG.apply(weights_init)
 
    netD = Discr(nc=3, ndf=64, imageSize=imageSize)
    netD.apply(weights_init)

    input = torch.FloatTensor(batchSize, 3, imageSize, imageSize)
    label = torch.FloatTensor(batchSize)
    input = Variable(input)
    label = Variable(label)
 
    real_label = 1
    fake_label = 0
    #criterion = nn.BCELoss()
    #criterion = nn.MSELoss()
    mse = lambda x,y: 0.5 * ((x - y)**2).view(x.size(0), -1).mean()
    #mse = lambda x,y: 0.5 * (((x - y)**2).view(x.size(0), -1).sum()) / x.size(0)
    criterion = mse

    clf = clf.cuda()
    netG = netG.cuda()
    netD = netD.cuda()
    #criterion = criterion.cuda()
    input = input.cuda()
    label = label.cuda()

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas = (beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas = (beta1, 0.999))

    stats = defaultdict(list)
    nb_updates = 0
    avg_objectness = 0.
    for epoch in range(niter):
        for i, data in enumerate(dataloader):
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

            gan_loss = gan_loss_coef * criterion(output, label)
            pixel_loss =  pixel_loss_coef * mse(fake, input)
            feature_loss = feature_loss_coef * mse(encode2(fake), encode2(input))
            rec_loss = rec_loss_coef * mse(encode(fake), encode(input))
            errG = pixel_loss + feature_loss + rec_loss + gan_loss

            y = classify(fake)
            y = nn.Softmax()(y)
            objectness = compute_objectness(y.data)
            avg_objectness = avg_objectness * 0.9 + objectness * 0.1
            stats['objectness'].append(objectness)
            stats['iter'].append(nb_updates)
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
            viz.updateTrace(X=np.array([stats['iter'][-1]]), Y=np.array([stats['loss'][-1]]), win=win, name='loss')
            viz.updateTrace(X=np.array([stats['iter'][-1]]), Y=np.array([stats['gan_loss'][-1]]), win=win, name='gan_loss')
            viz.updateTrace(X=np.array([stats['iter'][-1]]), Y=np.array([stats['pixel_loss'][-1]]), win=win, name='pixel_loss')
            viz.updateTrace(X=np.array([stats['iter'][-1]]), Y=np.array([stats['feature_loss'][-1]]), win=win, name='feature_loss')
            viz.updateTrace(X=np.array([stats['iter'][-1]]), Y=np.array([stats['objectness'][-1]]), win=win, name='objectness')
            viz.updateTrace(X=np.array([stats['iter'][-1]]), Y=np.array([stats['rec_loss'][-1]]), win=win, name='rec_loss')
            if nb_updates % 100 == 0:
                pd.DataFrame(stats).to_csv('{}/stats.csv'.format(outf), index=False)
                # the first 64 samples from the mini-batch are saved.
                vutils.save_image((real_cpu[0:64,:,:,:]+1)/2.,
                        '%s/real_samples.png' % outf, nrow=8)
                #fake = netG(hid)
                vutils.save_image((fake.data[0:64,:,:,:]+1)/2.,
                        '%s/fake_samples_epoch_%03d.png' % (outf, epoch), nrow=8)

            nb_updates += 1
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG.pth' % (outf))
        torch.save(netD.state_dict(), '%s/netD.pth' % (outf))
    return avg_objectness

if __name__ == '__main__':
    result = run(main)
