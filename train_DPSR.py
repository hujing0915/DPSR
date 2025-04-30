from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import math
import sys
from sklearn import preprocessing
import csv
import model
import util
import classifier as classifier_zero
import ot
from sklearn.metrics.pairwise import cosine_similarity
from config import opt
import time
import classifier_cls as classifier2
from loss import TripCenterLoss_min_margin, TripCenterLoss_margin
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from itertools import chain


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)
cls_criterion = nn.NLLLoss()
mse_loss = nn.MSELoss()

########################################### the FREE model ############################################
if opt.dataset in ['CUB']:
    center_criterion = TripCenterLoss_margin(num_classes=opt.nclass_seen, feat_dim=opt.attSize, use_gpu=opt.cuda)
elif opt.dataset in ['AWA2']:
    center_criterion = TripCenterLoss_min_margin(num_classes=opt.nclass_seen, feat_dim=opt.attSize, use_gpu=opt.cuda)
elif opt.dataset in ['SUN']:
    center_criterion = TripCenterLoss_margin(num_classes=opt.nclass_seen, feat_dim=opt.attSize, use_gpu=opt.cuda)
else:
    raise ValueError('Dataset %s is not supported' % (opt.dataset))


netE = model.Encoder(opt)
netG = model.Generator(opt)
netD = model.Discriminator(opt)
netFR = model.FR(opt, opt.attSize)
netSD = model.SDGZSL(opt)
# netGA = model.GSModule(vertices, opt.attSize)
netGA = model.GA(opt)
netR = model.RelationModule(opt)
netI = model.IndependenceConstraint(opt)
netSC = model.SemanticSelector(opt)

# Init Tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)  # attSize class-embedding size
noise = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)
# one = torch.FloatTensor([1])
one = torch.tensor(1, dtype=torch.float)
mone = one * -1
beta = 0
Attribute = torch.LongTensor(opt.nclass_all, opt.attSize)


###########################################
if opt.cuda:
    netD.cuda()
    netE.cuda()
    netG.cuda()
    netFR.cuda()

    netI.cuda()
    netSD.cuda()
    netR.cuda()
    netGA.cuda()
    netSC.cuda()

    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    input_label = input_label.cuda()
    Attribute = Attribute.cuda()

# Attribute = data.attribute

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(), reduction='sum')
    BCE = BCE.sum() / x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    return (BCE + KLD)

def kl_divergence_loss(visual_features, semantic_features):
    visual_log_probs = F.log_softmax(visual_features, dim=1)
    semantic_probs = F.softmax(semantic_features, dim=1)
    loss = F.kl_div(visual_log_probs, semantic_probs, reduction='batchmean')

    return loss

def ce_loss(vsp_output, updated_prototypes, labels):
    updated_prototypes = updated_prototypes.float()
    logits = torch.mm(vsp_output, updated_prototypes.t())
    logits = logits - logits.max(dim=1, keepdim=True)[0]
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

def cosine_loss(x, y):
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    return 1 - torch.sum(x_norm * y_norm, dim=1).mean()

def add_cross_weight(img, att):
    Q = img.view(-1, opt.attSize, 1)
    K = att.view(-1, 1, opt.resSize)
    R = torch.bmm(Q, K)
    soft_R = F.softmax(R, dim=2)
    _img = torch.bmm(soft_R, att.unsqueeze(2)).squeeze(2)
    _att = torch.bmm(soft_R.transpose(1, 2), img.unsqueeze(2)).squeeze(2)

    return _img, _att

def sample():
    batch_feature, batch_label, batch_att = data.next_seen_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def WeightedL1(pred, gt):
    wt = (pred - gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0), wt.size(1))
    loss = wt * (pred-gt).abs()
    return loss.sum() / loss.size(0)

def WeightedL2(x1, x2):
    return ((x1 - x2) ** 2).mean()

def generate_syn_feature(generator, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)

    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            syn_noisev = Variable(syn_noise)
            syn_attv = Variable(syn_att)
        fake = generator(syn_noisev, c=syn_attv)
        output = fake
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

def generate_syn_feature_all(generator, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)

    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            syn_noisev = Variable(syn_noise)
            syn_attv = Variable(syn_att)
        fake = generator(syn_noisev, c=syn_attv)
        output = fake
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label


main_params = chain(netSD.parameters(), netR.parameters())
optimizer = optim.Adam(netE.parameters(), lr=opt.lr)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizerG = optim.RMSprop(
#     netG.parameters(), lr=0.0001, weight_decay=0.0001, momentum=0.9)
optimizerFR = optim.Adam(netFR.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))
optimizerI = optim.Adam(netI.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))
optimizerSD = optim.Adam(main_params, lr=opt.dec_lr, betas=(opt.beta1, 0.999))
optimizerGA = optim.Adam(netGA.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))
optimizerSC = optim.Adam(netSC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_center = optim.Adam(center_criterion.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

def calc_gradient_penalty_FR(netFR, real_data, fake_data):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)
    _, _, disc_interpolates, _, _, _ = netFR(interpolates)
    ones = torch.ones(disc_interpolates.size())

    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

def MI_loss(mus, sigmas, i_c, alpha=1e-8):
    kl_divergence = (0.5 * torch.sum((mus ** 2) + (sigmas ** 2)
                                  - torch.log((sigmas ** 2) + alpha) - 1, dim=1))
    MI_loss = (torch.mean(kl_divergence) - i_c)
    return MI_loss

def optimize_beta(beta, MI_loss, alpha2=1e-6):
    beta_new = max(0, beta + (alpha2 * MI_loss))
    return beta_new


if not os.path.exists(os.path.join(opt.result_root, opt.dataset)):
    os.makedirs(os.path.join(opt.result_root, opt.dataset))

best_gzsl_acc = 0
best_zsl_acc = 0

for epoch in range(0, opt.nepoch):
    for loop in range(0, opt.loop):
        mean_lossD = 0
        mean_lossG = 0
        for i in range(0, data.ntrain, opt.batch_size):
            for p in netD.parameters():
                p.requires_grad = True

            for p in netFR.parameters():
                p.requires_grad = True

            for p in netR.parameters():
                p.requires_grad = True

            for p in netSD.parameters():
                p.requires_grad = True

            for p in netI.parameters():
                p.requires_grad = True

            for p in netGA.parameters():
                p.requires_grad = True

            for p in netSC.parameters():
                p.requires_grad = True

            gp_sum = 0
            for iter_d in range(opt.critic_iter):

                sample()
                netD.zero_grad()

                input_resv = Variable(input_res)
                input_attv = Variable(input_att)

                netFR.zero_grad()
                muR, varR, criticD_real_FR, latent_pred, _, recons_real = netFR(input_resv)
                criticD_real_FR = criticD_real_FR.mean()
                R_cost = opt.recons_weight * WeightedL1(recons_real, input_attv)

                ############################
                netI.zero_grad()
                hs, hn, _ = netSD(input_resv)
                disc_loss, _ = netI(hs.detach(), hn.detach())
                I_loss = opt.NI_weight * disc_loss
                I_loss.backward()
                optimizerI.step()

                netSD.zero_grad()
                netR.zero_grad()
                means, log_var, recon_x = netSD(input_resv)
                with torch.no_grad():
                    _, tc_penalty = netI(means, log_var)
                scores = netR(means, Attribute.detach())
                relation_loss = netR.compute_loss(scores, input_label.detach())
                recon_loss = F.mse_loss(recon_x, input_resv)
                SD_loss = opt.SD_weight * (recon_loss + relation_loss + tc_penalty)
                SD_loss.backward()
                optimizerSD.step()

                netGA.zero_grad()
                att = netGA(means.detach())
                GA_cost = opt.GA_weight * ce_loss(att, Attribute.detach(), input_label.detach())
                GA_cost.backward()
                optimizerGA.step()
                ############################

                att_muR = netGA(muR.detach())
                netSC.zero_grad()
                cls_feat, sem1, sem2 = netSC(att_muR.detach(), means.detach(), input_resv, epoch)
                # SC_cost = 1.5 * (mse_loss(sem1, input_attv) + mse_loss(sem2, input_attv))
                SC_cost = 1.5 * ((1 - cosine_loss(sem1, input_attv)) + (1 - cosine_loss(sem2, input_attv)))

                SC_cost.backward()
                optimizerSC.step()

                if opt.encoded_noise:
                    means, log_var = netE(input_resv, input_attv)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
                    eps = Variable(eps.cuda())
                    z = eps * std + means
                else:
                    noise.normal_(0, 1)
                    z = Variable(noise)

                # netFR.zero_grad()
                # muR, varR, criticD_real_FR, latent_pred, _, recons_real = netFR(input_resv)
                # criticD_real_FR = criticD_real_FR.mean()
                # R_cost = opt.recons_weight * WeightedL1(recons_real, input_attv)

                fake = netG(z, c=input_attv)
                muF, varF, criticD_fake_FR, _, _, recons_fake = netFR(fake.detach())

                criticD_fake_FR = criticD_fake_FR.mean()
                gradient_penalty = calc_gradient_penalty_FR(netFR, input_resv, fake.data)
                center_loss_real = center_criterion(muR, input_label, margin=opt.center_margin,
                                                    incenter_weight=opt.incenter_weight)
                D_cost_FR = center_loss_real * opt.center_weight + R_cost
                D_cost_FR.backward()
                optimizerFR.step()
                optimizer_center.step()

                criticD_real = netD(input_resv, input_attv)
                criticD_real = opt.gammaD * criticD_real.mean()
                criticD_real.backward(mone)

                criticD_fake = netD(fake.detach(), input_attv)
                criticD_fake = opt.gammaD * criticD_fake.mean()
                criticD_fake.backward(one)
                gradient_penalty = opt.gammaD * calc_gradient_penalty(netD, input_res, fake.data, input_att)
                gp_sum += gradient_penalty.data
                gradient_penalty.backward()
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty
                optimizerD.step()

                ############################
                netI.zero_grad()
                hs, hn, _ = netSD(fake.detach())
                disc_loss, _ = netI(hs.detach(), hn.detach())
                I_loss = opt.NI_weight * disc_loss
                I_loss.backward()
                optimizerI.step()

                netSD.zero_grad()
                netR.zero_grad()
                means, log_var, recon_x = netSD(fake.detach())
                with torch.no_grad():
                    _, tc_penalty = netI(means, log_var)
                scores = netR(means, Attribute.detach())
                relation_loss = netR.compute_loss(scores, input_label.detach())
                recon_loss = F.mse_loss(recon_x, input_resv).mean()
                SD_loss = opt.SD_weight * (recon_loss + relation_loss + tc_penalty)
                SD_loss.backward()
                optimizerSD.step()

                netGA.zero_grad()
                att = netGA(means.detach())
                GA_cost = opt.GA_weight * ce_loss(att, Attribute.detach(), input_label.detach())
                GA_cost.backward()
                optimizerGA.step()

                ############################
                # att_muR = netGA(muR.detach())
                # netSC.zero_grad()
                # cls_feat, sem1, sem2 = netSC(muF.detach(), means.detach(), input_resv)
                # SC_cost = 0.5 * (mse_loss(sem1, input_attv) + mse_loss(sem2, input_attv))
                # SC_cost.backward()
                # optimizerSC.step()

            gp_sum /= (opt.gammaD * opt.lambda1 * opt.critic_iter)
            if (gp_sum > 1.05).sum() > 0:
                opt.lambda1 *= 1.1
            elif (gp_sum < 1.001).sum() > 0:
                opt.lambda1 /= 1.1

            for p in netD.parameters():
                p.requires_grad = False

            if opt.recons_weight > 0 and opt.freeze_dec:
                for p in netFR.parameters():
                    p.requires_grad = False

            for p in netI.parameters():
                p.requires_grad = False

            for p in netSD.parameters():
                p.requires_grad = False

            for p in netR.parameters():
                p.requires_grad = False

            for p in netGA.parameters():
                p.requires_grad = False

            netE.zero_grad()
            netG.zero_grad()

            input_resv = Variable(input_res)
            input_attv = Variable(input_att)

            means, log_var = netE(input_resv, input_attv)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
            eps = Variable(eps.cuda())
            z = eps * std + means
            recon_x = netG(z, c=input_attv)
            vae_loss_seen = loss_fn(recon_x, input_resv, means, log_var)
            errG = vae_loss_seen

            if opt.encoded_noise:
                criticG_fake = netD(recon_x, input_attv).mean()
                fake = recon_x
            else:
                noise.normal_(0, 1)
                noisev = Variable(noise)
                fake = netG(z, c=input_attv)
                criticG_fake = netD(fake, input_attv).mean()

            G_cost = -criticG_fake
            errG += opt.gammaG * G_cost

            netFR.zero_grad()
            _, varR, criticG_fake_FR, latent_pred_fake, _, recons_fake = netFR(fake, train_G=True)
            R_cost = opt.recons_weight * WeightedL1(recons_fake, input_attv)
            errG += R_cost

            ############################
            netI.zero_grad()
            hs, hn, _ = netSD(fake.detach())
            disc_loss, _ = netI(hs.detach(), hn.detach())
            errG += opt.NI_weight * disc_loss
            optimizerI.step()

            netSD.zero_grad()
            netR.zero_grad()
            means, log_var, recon_x = netSD(fake.detach())
            with torch.no_grad():
                _, tc_penalty = netI(means, log_var)
            scores = netR(means, Attribute.detach())
            relation_loss = netR.compute_loss(scores, input_label.detach())
            recon_loss = F.mse_loss(recon_x, input_resv)
            SD_loss = recon_loss + relation_loss + tc_penalty
            errG += opt.SD_weight * SD_loss

            netGA.zero_grad()
            att = netGA(means.detach())
            GA_cost = opt.GA_weight * ce_loss(att, Attribute.detach(), input_label.detach())
            errG += GA_cost
            ############################

            errG.backward()
            optimizer.step()
            optimizerG.step()
            optimizerFR.step()
            optimizerSD.step()
            optimizerGA.step()
            optimizerI.step()

    print('[%d/%d]  Loss_D: %.2f Loss_G: %.2f, Wasserstein_dist:%.2f, vae_loss_seen:%.2f' % (epoch, opt.nepoch,
                                                                                            D_cost.item(),
                                                                                             G_cost.item(),
                                                                                            Wasserstein_D.item(),
                                                                                            vae_loss_seen.item()))
    netG.eval()
    netGA.eval()
    netFR.eval()
    netSD.eval()
    netSC.eval()

    syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
    train_X = torch.cat((data.train_feature, syn_feature), 0)
    train_Y = torch.cat((data.train_label, syn_label), 0)
    nclass = opt.nclass_all

    ### Train ZSL classifier
    # zsl_cls = classifier_zero.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5,
    #                                       25, opt.syn_num, netFR=netFR, dec_size=opt.attSize,
    #                                       dec_hidden_size=(opt.latensize * 2), netGA=netGA, netSD=netSD, netSC=netSC)
    # acc = zsl_cls.acc
    # if best_zsl_acc < acc:
    #     best_zsl_acc = acc
    # print('ZSL: unseen accuracy=%.3f' % (acc), end=' ')
    # if epoch % 10 == 0 and epoch != 0:
    #     print('ZSL: best_zsl_acc=%.3f' % best_zsl_acc)

    ### Train GZSL classifier
    gzsl_cls = classifier_zero.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5,
                                          25, opt.syn_num, netFR=netFR, dec_size=opt.attSize,
                                          dec_hidden_size=(opt.latensize*2), netGA=netGA, netSD=netSD, netSC=netSC)

    if best_gzsl_acc <= gzsl_cls.H:
        best_gzsl_epoch = epoch
        best_acc_seen, best_acc_unseen, best_gzsl_acc = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H
    print('GZSL: seen=%.3f, unseen=%.3f, h=%.3f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H), end=" ")

    if epoch % 10 == 0 and epoch != 0:
        print('GZSL: epoch=%d, best_seen=%.3f, best_unseen=%.3f, best_h=%.3f' % (best_gzsl_epoch,
                                                                                 best_acc_seen,
                                                                                 best_acc_unseen,
                                                                                 best_gzsl_acc))

    # reset G to training mode
    netG.train()
    netFR.train()
    netGA.train()
    netSD.train()
    netSC.train()

print(time.strftime('ending time:%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
print('Dataset', opt.dataset)
print('the best ZSL unseen accuracy is', best_zsl_acc)

if opt.gzsl:
    print('Dataset', opt.dataset)
    print('the best GZSL seen accuracy is', best_acc_seen)
    print('the best GZSL unseen accuracy is', best_acc_unseen)
    print('the best GZSL H is', best_gzsl_acc)
