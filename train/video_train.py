# Training
import sys
sys.path.append("..")
sys.path.append(".")
from pytorch_lightning.loggers import CSVLogger
import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import torch
from utils.losses import LossComputer
from utils.util import Evaluator
from models.video_seg_model.network import XMem
from dataset.uavid_dataset import *
from torch import nn
import os
import torch.optim as optim
from option.xmem_config import Con
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = XMem(config={})
        self.num_frames = config.num_frames
        self.num_ref_frames = config.num_ref_frames
        self.deep_update_prob = config.deep_update_prob
        
        self.loss_computer = LossComputer({"start_warm":config.start_warm, "end_warm":config.end_warm})

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

    def forward(self, data, it = 0):
        # only net is used in the prediction/inference
        # print(data["rgb"])
        out = {}
        frames = data["rgb"]
        # print(frames.device)
        first_frame_gt = data['first_frame_gt'].float().to(self.device)
        b = frames.shape[0]
        num_filled_objects = [o.item() for o in data['info']['num_objects']]
        num_objects = first_frame_gt.shape[2]
        selector = data['selector'].unsqueeze(2).unsqueeze(2)
        key, shrinkage, selection, f16, f8, f4 = self.net("encode_key", frames)
        filler_one = torch.zeros(1, dtype=torch.int64).to(self.device)
        hidden = torch.zeros((b, num_objects, 64, *key.shape[-2:])).to(self.device)
        # print(f16.device)
        # print(first_frame.device)
        # print(hidden.device)
        v16, hidden = self.net('encode_value', frames[:,0], f16[:,0], hidden, first_frame_gt[:,0])
        values = v16.unsqueeze(3) # add the time dimension

        for ti in range(1, self.num_frames):
            if ti <= self.num_ref_frames:
                    ref_values = values
                    ref_keys = key[:,:,:ti]
                    ref_shrinkage = shrinkage[:,:,:ti] if shrinkage is not None else None
            else:
                # pick num_ref_frames random frames
                # this is not very efficient but I think we would 
                # need broadcasting in gather which we don't have
                indices = [
                    torch.cat([filler_one, torch.randperm(ti-1)[:self.num_ref_frames-1]+1])
                for _ in range(b)]
                ref_values = torch.stack([
                    values[bi, :, :, indices[bi]] for bi in range(b)
                ], 0)
                ref_keys = torch.stack([
                    key[bi, :, indices[bi]] for bi in range(b)
                ], 0)
                ref_shrinkage = torch.stack([
                    shrinkage[bi, :, indices[bi]] for bi in range(b)
                ], 0) if shrinkage is not None else None
            memory_readout = self.net('read_memory', key[:,:,ti], selection[:,:,ti] if selection is not None else None, 
                                            ref_keys, ref_shrinkage, ref_values)
            hidden, logits, masks = self.net('segment', (f16[:,ti], f8[:,ti], f4[:,ti]), memory_readout, 
                            hidden, selector, h_out=(ti < (self.num_frames-1)))
            if ti < (self.num_frames-1):
                    is_deep_update = np.random.rand() < self.deep_update_prob
                    v16, hidden = self.net('encode_value', frames[:,ti], f16[:,ti], hidden, masks, is_deep_update=is_deep_update)
                    values = torch.cat([values, v16.unsqueeze(3)], 3)

            out[f'masks_{ti}'] = masks
            out[f'logits_{ti}'] = logits
        losses = self.loss_computer.compute({**data, **out}, num_filled_objects, it)
        return out, losses, num_filled_objects
    
    def training_step(self, batch, batch_idx):
        # img, mask = batch['img'], batch['gt_semantic_seg']
        # print(batch)
        data = batch
        out, losses, num_obj = self.forward(data)
        # loss = self.loss(prediction, mask)
        b,t = data["rgb"].shape[:2]
        softmax = nn.Softmax(dim = 1)
        for ti in range(1, t):
            for bi in range(b):
                logits = out[f'logits_{ti}'][bi:bi+1, :num_obj[bi]+ 1]
                label = data['cls_gt'][bi:bi+1,ti,0]
                logit = softmax(logits)
                logit = logit.argmax(dim = 1)
                # print(logit)
                self.metrics_train.add_batch(label.detach().cpu().numpy(),logit.detach().cpu().numpy())
        return {"loss": losses['total_loss']}
                

    def on_train_epoch_end(self):
        mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
        F1 = np.nanmean(self.metrics_train.F1())
        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        eval_value = {'mIoU': mIoU,
                        'F1': F1,
                        'OA': OA}
        print('train:', eval_value)
        

        iou_value = {}
        for class_name, iou in zip(CLASSES, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)
        self.metrics_train.reset()
        log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        data = batch
        # print(data["rgb"])
        out, losses, num_obj = self.forward(data)
        # loss = self.loss(prediction, mask)
        b,t = data["rgb"].shape[:2]
        softmax = nn.Softmax(dim = 1)
        for ti in range(1, t):
            for bi in range(b):
                logits = out[f'logits_{ti}'][bi:bi+1, :num_obj[bi]+ 1]
                label = data['cls_gt'][bi:bi+1,ti,0]
                logit = softmax(logits)
                logit = logit.argmax(dim = 1)
                # print(logit)
                self.metrics_val.add_batch(label.detach().cpu().numpy(),logit.detach().cpu().numpy())
        
        return {"loss_val": losses['total_loss']}

    def on_validation_epoch_end(self):
        mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
        F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()

        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('val:', eval_value)
        iou_value = {}
        for class_name, iou in zip(CLASSES, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)

        self.metrics_val.reset()
        log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(filter(
            lambda p: p.requires_grad, self.net.parameters()), lr=self.config.lr, weight_decay=self.config.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [self.config.steps], self.config.gamma)
      
      

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        train_dataset = UAVID_video(im_root=path.join(self.config.uavid_root, 'uavid_train'), max_jump=self.config.max_jump, num_frames=self.config.num_frames, cropsize=self.config.cropsize)
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        valid_dataset = UAVID_video(im_root=path.join(self.config.uavid_root, 'uavid_val'), max_jump=self.config.max_jump, num_frames=self.config.num_frames, cropsize=1024)
        valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False)
        return valid_dataloader

if __name__ == "__main__":
    config = Con
    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    model = Supervision_Train(config)
    logger = CSVLogger('lightning_logs', name=config.log_name)
    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='auto',
                            check_val_every_n_epoch=config.check_val_every_n_epoch,
                            callbacks=[checkpoint_callback], strategy='auto',
                            logger=logger)
    trainer.fit(model=model, ckpt_path=None)