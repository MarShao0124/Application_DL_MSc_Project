import torch
import torch.nn as nn
import numpy as np
import os
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
from optimizers import StableAdamW
from INP_utils import evaluation_batch, WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger


#data-related modules
from MVTecAD2_public_code_utils.mvtec_ad_2_public_offline import MVTecAD2

# Model-Related Modules
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block


warnings.filterwarnings("ignore")
def main(args):
    # Adopting a grouping-based reconstruction strategy similar to Dinomaly
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # Encoder info
    encoder = vit_encoder.load(args.encoder)
    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    # Model Preparation
    Bottleneck = []
    INP_Guided_Decoder = []
    INP_Extractor = []

    # bottleneck
    Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
    Bottleneck = nn.ModuleList(Bottleneck)

    # INP
    INP = nn.ParameterList(
                    [nn.Parameter(torch.randn(args.INP_num, embed_dim))
                     for _ in range(1)])

    # INP Extractor
    for i in range(1):
        blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Extractor.append(blk)
    INP_Extractor = nn.ModuleList(INP_Extractor)

    # INP_Guided_Decoder
    for i in range(8):
        blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Guided_Decoder.append(blk)
    INP_Guided_Decoder = nn.ModuleList(INP_Guided_Decoder)

    model = INP_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
                             target_layers=target_layers,  remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                             fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP)
    model = model.to(device)

    if args.phase == 'train':
        # Model Initialization

        # Data Preparation
        train_data = MVTecAD2(
        mad2_object=args.item,
        split='train')

        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                                       drop_last=True)

        validation_data = MVTecAD2(
        mad2_object=args.item,
        split='validation')

        validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

        trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # define optimizer
        optimizer = StableAdamW([{'params': trainable.parameters()}],
                                lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
        lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-4, total_iters=args.total_epochs*len(train_dataloader),
                                           warmup_iters=100)
        print_fn('train image number:{}'.format(len(train_data)))

        # Train
        for epoch in range(args.total_epochs):
            model.train()
            loss_list = []
            for img, _ in tqdm(train_dataloader, ncols=80):
                img = img.to(device)
                en, de, g_loss = model(img)
                loss = global_cosine_hm_adaptive(en, de, y=3)
                loss = loss + 0.2 * g_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                loss_list.append(loss.item())
                lr_scheduler.step()
            print_fn('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, args.total_epochs, np.mean(loss_list)))
            # if (epoch + 1) % args.total_epochs == 0:
        results = evaluation_batch(model, validation_dataloader, device, max_ratio=0.01, resize_mask=256)
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
        print_fn(
            '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                args.item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
        os.makedirs(os.path.join(args.save_dir, args.save_name, args.item), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, args.item, 'model.pth'))
        return results
    elif args.phase == 'test':
        # Data Preparation
        test_data = MVTecAD2(
            mad2_object=args.item,
            split='test_public')
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        # Test
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name, args.item, 'model.pth')), strict=True)
        model.eval()
        results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
        return results



if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(description='')

    # model info
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14') # 'dinov2reg_vit_small_14' or 'dinov2reg_vit_base_14' or 'dinov2reg_vit_large_14'
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--INP_num', type=int, default=6)

    # training info
    parser.add_argument('--total_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--phase', type=str, default='train')

    args = parser.parse_args()
    args.save_name = args.save_name + f'_dataset={args.dataset}_Encoder={args.encoder}_Resize={args.input_size}_Crop={args.crop_size}_INP_num={args.INP_num}'
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    args.item_list = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
    split_class = ["train", "validation", "test_public", "test_private", "test_private_mixed"]


    result_list = []
    
    for item in args.item_list:
        args.item = item
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = main(args)
        result_list.append([args.item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px])

    mean_auroc_sp = np.mean([result[1] for result in result_list])
    mean_ap_sp = np.mean([result[2] for result in result_list])
    mean_f1_sp = np.mean([result[3] for result in result_list])

    mean_auroc_px = np.mean([result[4] for result in result_list])
    mean_ap_px = np.mean([result[5] for result in result_list])
    mean_f1_px = np.mean([result[6] for result in result_list])
    mean_aupro_px = np.mean([result[7] for result in result_list])

    print_fn(result_list)
    print_fn(
        'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
            mean_auroc_sp, mean_ap_sp, mean_f1_sp,
            mean_auroc_px, mean_ap_px, mean_f1_px, mean_aupro_px))