from pathlib import Path
import argparse
import os
import sys
import time
import torch
import math
import numpy as np
import torchvision

from utils import (
    reduce_dict,
    LabeledDataset,
    MetricLogger,
    SmoothedValue,
    collate_fn,
    evaluate
)

def get_arguments():
    parser = argparse.ArgumentParser(description="Finetuning Retina net", add_help=False)

    parser.add_argument("--data-dir", type=Path, default="../data/labeled", required=True,
                        help='Path to the labeled images and yaml target files')

    parser.add_argument("--batch-size", type=int, default=4,
                        help='Batch size')

    parser.add_argument("--lr", type=float, default=0.0006,
                        help='Learning rate')
                        
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    return parser

def get_model(num_classes=100):

    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None, \
        weights_weights=None,num_classes=num_classes)

    print('This is retina net structure:', model)
    print('Loading state dict...')
    state_d = torch.load('model.pth', map_location='cpu')
    model.load_state_dict(state_d)

    # print('BEFORE', model.state_dict()['backbone.body.conv1.weight'][0][0][0])
    # filename = args.model_name + '.pth'
    # model.load_state_dict(torch.load(args.exp_dir / filename, map_location='cpu'))
    # model.load_state_dict(torch.load('../output/ckpt.pth', map_location='cpu')
    # print('AFTER', model.state_dict()['backbone.body.conv1.weight'][0][0][0])

    model.eval()
    return model

# reference 
# https://github.com/pytorch/vision/blob/main/references/detection/engine.py

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    epoch_loss = []

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()

        losses.backward()
        optimizer.step()

        epoch_loss.append(float(losses))
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print('EPOCH LOSS', np.mean(epoch_loss))
    return metric_logger, np.mean(epoch_loss)


def train(model, optimizer, lr_scheduler, train_loader, valid_loader, device, start_epoch):
    for epoch in range(start_epoch, args.epochs):
        meric_logger, losses = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1)
        optimizer.step()
        # ReduceLROnPlateau
        lr_scheduler.step(losses)
        print('Saving Retina Net Model State Dict ...')
        filename = 'model_out.pth'
        torch.save(model.state_dict(), filename)
        # print the mAP score every 4 epochs
        if epoch % 4 == 0:
            evaluate(model, valid_loader, device=device)

def main(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    
    num_classes = 100

    train_dataset = LabeledDataset(root=args.data_dir, split="training", \
        transforms=lambda x, y: (torchvision.transforms.functional.to_tensor(x), y))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, \
         num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    valid_dataset = LabeledDataset(root=args.data_dir, split="validation", \
        transforms=lambda x, y: (torchvision.transforms.functional.to_tensor(x), y))

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, \
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    
    model = get_model(num_classes).cuda(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    train(model, optimizer, lr_scheduler, train_loader, valid_loader, device, start_epoch=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Finetuning Retina Net', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
