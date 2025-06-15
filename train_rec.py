from utils import set_seed, collate_fn, AverageMeter, accuracy
# from model import Dreconstruction
import torch
from tqdm import tqdm
import numpy as np
import sys
from model import VAE

# def train_rec(args, model, train_dataloader, dev_dataloader):
#     # num_steps_pre = 0
#     # num_steps_rec = 0
#     loss_avg = AverageMeter()
#     eval_loss_avg = AverageMeter()

#     loss_avg.reset()
#     eval_loss_avg.reset()

#     rector = Dreconstruction(model.config.hidden_size, args.rec_num, args.rec_drop)
#     rector.cuda()
#     optimizer_mlp = torch.optim.Adam(rector.parameters(), lr=1e-4)
#     # schedulor_rec = torch.optim.lr_scheduler.ExponentialLR(optimizer_mlp, 0.7)
#     rector.rec = False
#     best_eval = float('inf')
#     patient = 0
#     model.eval()

#     for epoch in range(int(1000)):

#         rector.train()
#         rector.zero_grad()
#         for step, batch in enumerate(train_dataloader):
#             batch = {key: value.to(args.device) for key, value in batch.items()}
#             # batch = {key: value.cuda() for key, value in batch.items()}
#             # labels = batch['labels']
#             outputs = model.bert(
#                 input_ids=batch['input_ids'],
#                 attention_mask=batch['attention_mask'],
#             )
#             pooled = outputs[0].mean(dim=1)
#             label_ids = batch['labels']
#             loss = rector(pooled,label_ids) * 10000
#             loss.backward()
#             loss_avg.update(loss.item(), n=len(batch['labels']))
#             optimizer_mlp.step()
#             # scheduler.step()
#             rector.zero_grad()
#             # eval
#             # schedulor_rec.step()
#         print("\n")
#         print("_" * 15)
#         print("mse loss of epoch", epoch + 1, loss_avg.avg)

#         for step, batch in enumerate(dev_dataloader):
#             rector.eval()
#             # batch = {key: value.to(args.device) for key, value in batch.items()}
#             batch = {key: value.cuda() for key, value in batch.items()}
#             # labels = batch['labels']
#             label_ids = batch['labels']
#             outputs = model.bert(
#                 input_ids=batch['input_ids'],
#                 attention_mask=batch['attention_mask'],
#             )
#             pooled = outputs[0].mean(dim=1)
#             val_loss = rector(pooled,label_ids) * 10000
#             eval_loss_avg.update(val_loss.item(), n=len(batch['labels']))
#         # print("\n")
#         print("val rec loss:", eval_loss_avg.avg)
#         sys.stdout.flush()
#         if eval_loss_avg.avg < best_eval:
#             best_eval = eval_loss_avg.avg
#             patient = 0
#         else:
#             patient += 1

#         loss_avg.reset()
#         eval_loss_avg.reset()
#         if patient > 3:
#             break

#     return rector

def train_rec(args, model, train_dataloader, dev_dataloader):
    loss_avg = AverageMeter()
    eval_loss_avg = AverageMeter()

    loss_avg.reset()
    eval_loss_avg.reset()
    vae_model = VAE(model.config.hidden_size, args.rec_num,model.config.num_labels)
    # vae_model = VAE(model.config.hidden_size, args.rec_num, args.rec_drop)
    # for name, param in vae_model.named_parameters():
    #     print(f'{name}: {param.numel()}')
    vae_model.cuda()
    optimizer_mlp = torch.optim.Adam(vae_model.parameters(), lr=1e-4)
    best_eval = float('inf')
    vae_model.rec = False
    patient = 0
    model.eval()
    for epoch in range(int(10000)):
        vae_model.train()
        vae_model.zero_grad()
        for step, batch in enumerate(train_dataloader):
            batch = {key: value.to(args.device) for key, value in batch.items()}
            outputs_text = model.bert(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            # pooled = outputs_text[0]
            pooled = outputs_text[0].mean(dim=1)  
            label_ids = batch['labels']
            total_loss =  vae_model(pooled,label_ids) * 10000
            # 反向传播和参数更新
            optimizer_mlp.zero_grad()
            total_loss.backward()
            optimizer_mlp.step()

            loss_avg.update(total_loss.item(), n=len(batch['labels']))
            

        print("\n")
        print("_" * 15)
        print("Total VAE loss of epoch", epoch + 1, loss_avg.avg)
        
        for step, batch in enumerate(dev_dataloader):
            vae_model.eval()
            batch = {key: value.to(args.device) for key, value in batch.items()}

            outputs_text = model.bert(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            # pooled = outputs_text[0].mean(dim=1)
            pooled = outputs_text[0].mean(dim=1)
            # pooled = outputs_text[0]
            label_ids = batch['labels']
            total_loss =  vae_model(pooled,label_ids) * 10000
            eval_loss_avg.update(total_loss.item(), n=len(batch['labels']))

        print("Validation VAE loss:", eval_loss_avg.avg)
        sys.stdout.flush()

        if eval_loss_avg.avg < best_eval:
            best_eval = eval_loss_avg.avg
            patient = 0
        else:
            patient += 1

        loss_avg.reset()
        eval_loss_avg.reset()
        if patient > 3:
            break
        
    return vae_model