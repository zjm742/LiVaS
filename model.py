import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, RobertaModel, BertModel
from sklearn.covariance import EmpiricalCovariance
import torch.distributions as distributions
# 计算KL散度损失
def calculate_kl_divergence(mu, logvar):
    # 创建正态分布
    normal_distribution = distributions.Normal(0, 1)

    # 计算先验分布（标准正态分布）和后验分布之间的KL散度
    posterior_distribution = distributions.Normal(mu, logvar.exp().sqrt())
    kl_divergence = torch.distributions.kl.kl_divergence(posterior_distribution, normal_distribution).sum(-1)
    
    return kl_divergence




class ConvexSampler(nn.Module):
    def __init__(self, oos_label_id):
        super(ConvexSampler, self).__init__()
        # self.num_convex = round(args.n_oos/5)
        self.num_convex = 50
        self.num_convex_val = 50
        self.oos_label_id = oos_label_id
    def forward(self, z, label_ids, data_type, mode=None):
        convex_list = []

        if mode == 'train':
            if torch.unique(label_ids).size(0) > 3 and self.num_convex!=0:
                while len(convex_list) < self.num_convex:
                    # # 计算距离
                    # distances_1, distances_2,label_1,label_2 = compute_distances_to_class_centers(z, label_ids)
                    cdt = np.random.choice(label_ids.size(0), 2, replace=False)
                    # print(cdt)
                    if label_ids[cdt[0]] != label_ids[cdt[1]]:
                        s = np.random.uniform(0, 1, 1)
                        convex_list.append(s[0] * z[cdt[0]] + (1 - s[0]) * z[cdt[1]])                

                convex_samples = torch.cat(convex_list, dim=0).view(self.num_convex, -1)
                z = torch.cat((z, convex_samples), dim=0)
                label_ids = torch.cat((label_ids, torch.tensor([self.oos_label_id] * len(convex_list)).cuda()), dim=0)
                data_type = torch.cat((data_type, torch.ones(len(convex_list))), dim=0)
        elif mode == 'dev':
            if torch.unique(label_ids).size(0) > 3 and self.num_convex_val!=0:
                val_num = self.num_convex_val
                while len(convex_list) < val_num:
                    cdt = np.random.choice(label_ids.size(0), 2, replace=False)
                    if label_ids[cdt[0]] != label_ids[cdt[1]]:
                        s = np.random.uniform(0, 1, 1)
                        convex_list.append(s[0] * z[cdt[0]] + (1 - s[0]) * z[cdt[1]])

                convex_samples = torch.cat(convex_list, dim=0).view(val_num, -1)
                z = torch.cat((z, convex_samples), dim=0)
                label_ids = torch.cat((label_ids, torch.tensor([self.oos_label_id] * val_num).cuda()), dim=0)
        return z, label_ids,data_type



class SupervisedConstrastiveLoss(nn.Module):
    def __init__(self, temperature=0.3, scale_by_temperature=True):
        super(SupervisedConstrastiveLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.size(0)

        # Normalize feature vectors
        features = F.normalize(features, p=2, dim=1)

        # Compute logits
        similarities = torch.matmul(features, features.t()) / self.temperature
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)

        positives_mask = mask - torch.eye(batch_size, device=device)
        negatives_mask = 1. - mask

        # Compute log probabilities
        exp_similarities = torch.exp(similarities)
        numerator = torch.sum(exp_similarities * positives_mask, dim=1, keepdim=True)

        denominator = numerator + torch.sum(exp_similarities * negatives_mask, dim=1, keepdim=True)
        log_probs = similarities - torch.log(denominator)
        # Avoid NaNs in log_probs
        if torch.isnan(log_probs).any():
            raise ValueError("Log_probs has NaN!")

        # Compute loss
        loss = -torch.sum(log_probs * positives_mask) / torch.sum(positives_mask)

        if self.scale_by_temperature:
            loss *= self.temperature

        return loss

class VAE(nn.Module):
    def __init__(self, in_dim, num_rec,num_classes):
        super(VAE, self).__init__()
        latent_dim =5
        self.num_rec = num_rec
        self.rec = False
        # 编码器
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256)

        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, in_dim),
        
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_classes)
        )
        # 编码器部分

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def compute_distance_loss(self, mu, label_ids):
        # 计算不同类别样本在潜在空间的分布之间的距离
        # 这里简单地使用欧氏距离来衡量距离
        distance_loss = 0
        num_samples = mu.size(0)
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                if label_ids[i] != label_ids[j]:  # 如果两个样本属于不同的类别
                    distance_loss += torch.dist(mu[i], mu[j], 2)  # 计算欧氏距离
        return distance_loss


    def forward(self, emb, label_ids=None):
        if self.rec:
            emb_enlarged = emb.repeat(self.num_rec, 1)
            # 编码
            h = self.encoder(emb_enlarged)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            z = self.reparameterize(mu, logvar)
            # 解码
            recset = self.decoder(z)
            recset = torch.cat((emb, recset), dim=0)
            label_ids = label_ids.repeat(self.num_rec + 1)
            data_type = torch.zeros(recset.size()[0])  
            # 更新标识原始数据和合成数据的张量
            # 假设emb是原始数据，recset是合成数据
            num_original = len(emb)
            num_synthetic = len(recset) - num_original
            # 将前num_original个元素标记为原始数据类型
            data_type[:num_original] = 0
            # 将剩余的元素标记为合成数据类型
            data_type[num_original:] = 1
            indices = torch.randperm(recset.size()[0])
            recset = recset[indices]
            label_ids = label_ids[indices]
            data_type = data_type[indices]

            return recset, label_ids,data_type
                  
        else:
            # 这里执行重建任务
            h = self.encoder(emb)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            z = self.reparameterize(mu, logvar)
            recset = self.decoder(z)
            predicted_labels = self.classifier(recset)
            reconstruction_loss = nn.MSELoss()(recset, emb)
            kl_divergence = calculate_kl_divergence(mu, logvar).mean()
            total_loss = reconstruction_loss + kl_divergence 
            total_loss = total_loss.mean() 
            return total_loss



class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.3)
        # self.rec_drop = config.rec_drop
        self.rec_num = config.rec_num
        self.train_rec = config.train_rec
        self.insampler = None
        self.convex = config.convex
        self.sampler = ConvexSampler(config.num_labels)
        # if config.freeze:
        #     for name, param in self.bert.named_parameters():
        #         param.requires_grad = False
                # if "encoder.layer.11" in name or "pooler" in name:
                #     # print("set last layer trainable")
                #     param.requires_grad = True

        if self.convex:
            self.classifier = nn.Sequential(
                # nn.Linear(768, 768),
                # nn.ReLU()
                # nn.Linear(768, 768),
                # nn.Linear(config.hidden_size,256),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                # nn.Linear(feat_dim, n_way)
                nn.Linear(config.hidden_size, self.num_labels + 1)
            )
        else:
            self.classifier = nn.Sequential(
                # nn.Linear(in_dim, in_dim),
                # nn.ReLU()
                # nn.Linear(in_dim, in_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                # nn.Linear(feat_dim, n_way)
                nn.Linear(config.hidden_size, self.num_labels)
            )

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            epoch=1,
            mode=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )

        # pooled_output = pooled = outputs[1]
        attention = outputs.attentions[-1] 
        pooled_output = pooled = outputs[0].mean(dim=1)
        data_type = None
        if mode == "train" and self.train_rec and labels is not None:
            self.insampler.rec = True
            pooled_output, labels,data_type = self.insampler(pooled_output, label_ids=labels)
            self.insampler.rec = False
        if self.convex and labels is not None:

            pooled_output, labels, data_type = self.sampler(pooled_output, labels, data_type, mode=mode)

        # pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = {}
        if labels is not None and self.training:
            loss_fct = CrossEntropyLoss()
            con_loss = SupervisedConstrastiveLoss()
            if self.convex:
                loss_all = loss_fct(logits.view(-1, self.num_labels + 1), labels.view(-1))
                distances = con_loss(pooled_output, labels)
            else:
                loss_all = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss["loss"] = 0.7*loss_all + 0.3*distances
            # loss["loss"] = loss_all
            # loss = loss + self.config.alpha * cos_loss
            # loss = loss

        # return ((loss,) + output)
        return loss, logits, labels, pooled, attention
