import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import opt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def reparameter(mu, sigma):
    return (torch.randn_like(mu) * sigma) + mu

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        layer_sizes = opt.encoder_layer_sizes
        latent_size = opt.latent_size
        layer_sizes[0] += latent_size
        self.fc1 = nn.Linear(layer_sizes[0], layer_sizes[-1])
        self.fc3 = nn.Linear(layer_sizes[-1], latent_size*2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.linear_means = nn.Linear(latent_size * 2, latent_size)
        self.linear_log_var = nn.Linear(latent_size * 2, latent_size)
        self.apply(weights_init)

    def forward(self, x, c=None):
        if c is not None:
            x = torch.cat((x, c), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

# Decoder / Generator
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        layer_sizes = opt.decoder_layer_sizes
        latent_size = opt.latent_size
        input_size = latent_size * 2
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)
        self.MLP = nn.Sequential(*[nn.Linear(input_size, layer_sizes[0]),
                                nn.LeakyReLU(0.2, True),
                                nn.Linear(layer_sizes[0], layer_sizes[1])]
                        )

    def forward(self, z, c=None):
        z = torch.cat((z, c), dim=-1)
        out = self.MLP(z)
        self.out = out
        x = torch.sigmoid(out)
        return x

# Traditional Discriminator
class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        return h

class FR(nn.Module):
    def __init__(self, opt, attSize):
        super(FR, self).__init__()

        self.hidden = None
        self.lantent = None

        self.latensize = opt.latensize
        self.attSize = opt.attSize

        self.layer1 = 1024
        self.layer2 = 2048

        self.fc1 = nn.Linear(opt.resSize, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, attSize * 2)

        self.discriminator = nn.Linear(opt.attSize, 1)
        self.classifier = nn.Linear(opt.attSize, opt.nclass_seen)

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

        self.fc_mu = nn.Linear(attSize * 2, attSize)
        self.fc_std = nn.Linear(attSize * 2, attSize)

        self.up = nn.Sequential(
            nn.Linear(opt.attSize * 2, self.layer1),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.layer1, self.layer2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.layer2, opt.attSize)
        )

        self.down = nn.Sequential(
            nn.Linear(opt.attSize * 2, self.layer1),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.layer1, self.layer2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.layer2, opt.attSize)
        )

    def forward(self, feat, train_G=False):
        h = feat

        self.hidden = self.lrelu(self.fc1(h))
        self.lantent = F.sigmoid(self.fc3(self.hidden))

        mus = self.up(self.lantent)
        stds = F.sigmoid(self.down(self.lantent))

        encoder_out = reparameter(mus, stds)
        h = encoder_out

        # For Discriminator
        if not train_G:
            dis_out = self.discriminator(encoder_out)
        else:
            dis_out = self.discriminator(mus)
        pred = self.logic(self.classifier(mus))

        h = h/h.pow(2).sum(1).sqrt().unsqueeze(1).expand(h.size(0), h.size(1))

        return mus, stds, dis_out, pred, encoder_out, h

    def getLayersOutDet(self):
        return self.hidden.detach()


class GA(nn.Module):
    def __init__(self, opt):
        super(GA, self).__init__()
        latent_size = opt.latent_size
        self.apply(weights_init)
        self.MLP = nn.Sequential(
            nn.Linear(latent_size, 2048),
            nn.LeakyReLU(0.2, True),
            nn.Linear(2048, latent_size),
            nn.Sigmoid()
            )

    def forward(self, z):
        out = self.MLP(z)
        return out


class RelationModule(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(opt.latent_size + opt.latent_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, hs, a):
        B, hs_dim = hs.shape
        Nc, a_dim = a.shape
        hs_exp = hs.unsqueeze(1).expand(-1, Nc, -1)
        a_exp = a.unsqueeze(0).expand(B, -1, -1)
        pair_features = torch.cat([hs_exp, a_exp], dim=-1)
        scores = self.net(pair_features).squeeze(-1)
        return scores

    def compute_loss(self, scores, labels):
        labels = labels.cpu().long()
        B, Nc = scores.shape
        target = torch.zeros_like(scores)
        target[torch.arange(B), labels] = 1.0
        return F.mse_loss(scores, target)

class IndependenceConstraint(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(opt.latent_size * 2, 1),
            nn.Sigmoid()
        )

    def permute_features(self, hs, hn):
        B = hs.size(0)
        perm_idx1 = torch.randperm(B)
        perm_idx2 = torch.randperm(B)
        hs_perm = hs[perm_idx1, :]
        hn_perm = hn[perm_idx2, :]
        tilde_h = torch.cat([hs_perm, hn_perm], dim=1)
        return tilde_h

    def forward(self, hs, hn):
        h = torch.cat([hs, hn], dim=1)
        tilde_h = self.permute_features(hs, hn)
        pred_real = self.discriminator(h.detach())
        pred_fake = self.discriminator(tilde_h.detach())
        real_loss = -torch.log(pred_real + 1e-8).mean()
        fake_loss = -torch.log(1 - pred_fake + 1e-8).mean()
        disc_loss = real_loss + fake_loss
        density_ratio = torch.log(pred_real / (1 - pred_real + 1e-8))
        tc_penalty = density_ratio.mean()

        return disc_loss, tc_penalty

class SDGZSL(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(opt.resSize, opt.latent_size * 2),
            nn.ReLU(),
            nn.Linear(opt.latent_size * 2, opt.latent_size * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(opt.latent_size, opt.latent_size * 2),
            nn.ReLU(),
            nn.Linear(opt.latent_size * 2, opt.resSize),
        )

    def forward(self, prototypes):
        h = self.encoder(prototypes)
        hs, hn = h.chunk(2, dim=-1)
        recon = self.decoder(hs + hn)
        return hs, hn, recon


class SemanticSelector(nn.Module):
    def __init__(self, opt):
        super().__init__()
        semantic_dim = opt.attSize
        num_heads = opt.head_number
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.proj_global_up = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, semantic_dim),
            nn.Sigmoid()
        )
        self.proj_global_down = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, semantic_dim),
        )
        self.proj_local_up = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, semantic_dim),
            nn.Sigmoid()
        )
        self.proj_local_down = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, semantic_dim),
        )
        self.proj_visual = nn.Linear(opt.resSize, semantic_dim)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=semantic_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(2 * semantic_dim, semantic_dim),
            nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Linear(2 * semantic_dim, semantic_dim),
            nn.Sigmoid()
        )
        self.g = nn. Sequential(
            nn.Linear(semantic_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, semantic_dim)
        )
        self.g1 = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, semantic_dim)
        )
        self.w = self.add_cross_weight

    def forward(self, semantic_global, semantic_local, visual_feat, epoch=None):
        if epoch == 10:
            # plot_semantic_correlations(semantic_global, semantic_global, "bg")
            plot_semantic_correlations(semantic_local, semantic_local, "bl")
            # plot_semantic_correlations(semantic_global, semantic_local, "bm")

        h = visual_feat
        semantic_global = self.proj_global_up(semantic_global) * self.proj_global_down(semantic_global)
        semantic_local = self.proj_local_up(semantic_local) * self.proj_local_down(semantic_local)
        h = self.proj_visual(h)

        # global_semantic
        semantic_global_output, _ = self.multihead_attn(
            query=h.unsqueeze(1),
            key=semantic_global.unsqueeze(1),
            value=semantic_global.unsqueeze(1)
        )
        # semantic_global = semantic_global.squeeze(1)
        # semantic_global = F.normalize(semantic_global)
        # h = F.normalize(semantic_global + h)
        # semantic_global = F.normalize(self.g(semantic_global) + semantic_global)

        semantic_global_output = semantic_global_output.squeeze(1)
        combined = torch.cat([semantic_global_output, h], dim=1)
        gate = self.gate(combined)
        fused_feat1 = F.normalize(gate * semantic_global_output + (1 - gate) * h)
        # fused_feat1 = F.normalize(fused_feat1 + semantic_global)

        # local_semantic
        semantic_local_output, _ = self.multihead_attn(
            query=semantic_global.unsqueeze(1),
            key=semantic_local.unsqueeze(1),
            value=semantic_local.unsqueeze(1)
        )
        # semantic_local = semantic_local.squeeze(1)
        # semantic_local = F.normalize(semantic_local)
        # semantic_local = F.normalize(self.g1(semantic_local) + semantic_local)
        # semantic_global = F.normalize(semantic_local + semantic_global)

        semantic_local_output = semantic_local_output.squeeze(1)
        combined1 = torch.cat([semantic_local_output, h], dim=1)
        gate1 = self.gate(combined1)
        fused_feat2 = F.normalize(gate1 * semantic_local_output + (1 - gate1) * h)

        if epoch == 10:
            # plot_semantic_correlations(fused_feat1, fused_feat1, "ag")
            plot_semantic_correlations(fused_feat2, fused_feat2, "al")
            # plot_semantic_correlations(fused_feat1, fused_feat2, "am")

        # fused_feat2 = F.softmax(h, dim=1)*fused_feat2
        # fused_feat2 = F.normalize(fused_feat2 + semantic_local)
        # _, fused_feat2 = self.w(visual_feat, fused_feat2)
        fused_feat = torch.cat([visual_feat, fused_feat1, fused_feat2], dim=1)

        return fused_feat, fused_feat1, fused_feat2

    def add_cross_weight(self, semantic_global, semantic_local):
        Q = semantic_global.view(-1, opt.resSize, 1)
        K = semantic_local.view(-1, 1, opt.attSize)
        R = torch.bmm(Q, K)
        soft_R = F.softmax(R, dim=2)
        _semantic_global = torch.bmm(soft_R, semantic_local.unsqueeze(2)).squeeze(2)
        _semantic_local = torch.bmm(soft_R.transpose(1, 2), semantic_global.unsqueeze(2)).squeeze(2)

        return _semantic_global, _semantic_local

# class SemanticSelector(nn.Module):
#     def __init__(self, opt):
#         super().__init__()
#         semantic_dim = opt.attSize
#         num_heads = opt.head_number
#         self.lrelu = nn.LeakyReLU(0.2, True)
#         self.sigmoid = nn.Sigmoid()
#         def create_proj():
#             return nn.Sequential(
#                 nn.Linear(semantic_dim, semantic_dim),
#                 nn.ReLU(),
#                 nn.Linear(semantic_dim, semantic_dim)
#             )
#         self.proj_global_up = create_proj()
#         self.proj_global_down = create_proj()
#         self.proj_local_up = create_proj()
#         self.proj_local_down = create_proj()
#         self.proj_global_up.add_module("3", nn.Sigmoid())
#         self.proj_local_up.add_module("3", nn.Sigmoid())
#         self.proj_visual = nn.Linear(opt.resSize, semantic_dim)
#         self.attn_base = nn.MultiheadAttention(
#             embed_dim=semantic_dim,
#             num_heads=num_heads,
#             batch_first=True
#         )
#         def create_gate():
#             return nn.Sequential(
#                 nn.Linear(2 * semantic_dim, semantic_dim),
#                 nn.Sigmoid()
#             )
#
#         self.gate = create_gate()
#         self.gate2 = create_gate()
#         self.g = nn.Sequential(
#             nn.Linear(semantic_dim, semantic_dim),
#             nn.ReLU()
#         )
#
#         self.iterations = 20
#
#     def forward(self, semantic_global, semantic_local, visual_feat, epoch=None):
#         h = self.proj_visual(visual_feat).unsqueeze(1)  # [B, 1, D]
#         g = (self.proj_global_up(semantic_global) *
#              self.proj_global_down(semantic_global)).unsqueeze(1)
#         for _ in range(self.iterations):
#             g, _ = self.attn_base(
#                 query=h,
#                 key=g,
#                 value=g,
#                 need_weights=False
#             )
#             g = F.normalize(g + self.g(g), dim=-1)
#         global_feat = g.squeeze(1)
#         gate = self.gate(torch.cat([global_feat, h.squeeze(1)], dim=1))
#         fused_feat1 = F.normalize(gate * global_feat + (1 - gate) * h.squeeze(1))
#         l = (self.proj_local_up(semantic_local) *
#              self.proj_local_down(semantic_local)).unsqueeze(1)
#
#         for _ in range(self.iterations):
#             l, _ = self.attn_base(
#                 query=g.detach(),
#                 key=l,
#                 value=l,
#                 need_weights=False
#             )
#             l = F.normalize(l + self.g(l), dim=-1)
#         local_feat = l.squeeze(1)
#         gate2 = self.gate2(torch.cat([local_feat, h.squeeze(1)], dim=1))
#         fused_feat2 = F.normalize(gate2 * local_feat + (1 - gate2) * h.squeeze(1))
#         fused_feat = torch.cat([visual_feat, fused_feat1, fused_feat2], dim=1)
#
#         return fused_feat, fused_feat1, fused_feat2


# FREE Model
class FREE(nn.Module):
    def __init__(self, opt, attSize):
        super(FREE, self).__init__()
        self.embedSz = 0
        self.hidden = None
        self.lantent = None
        self.latensize = opt.latensize
        self.attSize = opt.attSize
        self.fc1 = nn.Linear(opt.resSize, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, attSize * 2)
        self.discriminator = nn.Linear(opt.attSize, 1)
        self.classifier = nn.Linear(opt.attSize, opt.nclass_seen)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self, feat, train_G=False):
        h = feat
        self.hidden = self.lrelu(self.fc1(h))
        self.lantent = self.fc3(self.hidden)
        mus, stds = self.lantent[:, :self.attSize], self.lantent[:, self.attSize:]
        stds = self.sigmoid(stds)
        encoder_out = reparameter(mus, stds)
        h = encoder_out
        if not train_G:
            dis_out = self.discriminator(encoder_out)
        else:
            dis_out = self.discriminator(mus)
        pred = self.logic(self.classifier(mus))
        if self.sigmoid is not None:
            h = self.sigmoid(h)
        else:
            h = h / h.pow(2).sum(1).sqrt().unsqueeze(1).expand(h.size(0), h.size(1))
        return mus, stds, dis_out, pred, encoder_out, h

    def getLayersOutDet(self):
        return self.hidden.detach()


