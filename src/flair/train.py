"""
This code is adapted from OpenCLIP:
https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/train.py

The code integrates additional modifications and extensions to support the FLAIR models.
Original authors: ML Foundations.
"""
import json
import logging
import math
import os
import time
from unicodedata import normalize

import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from typing import Any, Dict, Optional, Tuple, Union

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def get_reordered_indices(batch_size, num_batches):
    """
    The original order: [(I_1, T_1), ..., (I_1, T_B), (I_2, T_1), ..., (I_2, T_B), ...
                          (T_1, T_B+1)...]
    reorder to [(I_1, T_1), ..., (I_1, T_N), (I_2, T_1), ..., (I_2, T_N), ... , (I_N, T_1), ..., I(I_N, T_N)]
    returning a list of reordered indices
    """
    reordered_indices = []
    for k in range(batch_size):
        for n in range(num_batches):
            base_idx = n * batch_size * batch_size
            img_idx_start = base_idx + k * batch_size
            img_idx_end = img_idx_start + batch_size
            reordered_indices.extend(list(range(img_idx_start, img_idx_end)))

    return reordered_indices


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader

    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        #images, texts, img_ids = batch
        images, texts = batch
        # values, counts = torch.unique(img_ids, return_counts=True)
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)
                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {math.log(logit_scale_scalar):.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": math.log(logit_scale_scalar),
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()
    zero_shot_metrics = zero_shot_eval(
        model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs):
        if 'retrieval_coco' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_coco']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_coco', model, txt_loader, img_loader, img2txt_dict, txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)
        if 'retrieval_flickr' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_flickr']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_flickr', model, txt_loader, img_loader, img2txt_dict, txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)
        if 'retrieval_cc3m_train' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_cc3m_train']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_cc3m_train', model, txt_loader, img_loader, img2txt_dict,
                                         txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)

        if 'retrieval_docci' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_docci']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_docci', model, txt_loader, img_loader, img2txt_dict,
                                         txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)

        if 'retrieval_urban_1k' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_urban_1k']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_urban_1k', model, txt_loader, img_loader, img2txt_dict,
                                         txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)

        if 'retrieval_iiw' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_iiw']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_iiw', model, txt_loader, img_loader, img2txt_dict,
                                         txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)

        if 'retrieval_dci' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_dci']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_dci', model, txt_loader, img_loader, img2txt_dict,
                                         txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)

        if 'retrieval_sharegpt4v-1k' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_sharegpt4v-1k']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_sharegpt4v-1k', model, txt_loader, img_loader, img2txt_dict,
                                         txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)

        if 'retrieval_sharegpt4v-10k' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_sharegpt4v-10k']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_sharegpt4v-10k', model, txt_loader, img_loader, img2txt_dict,
                                         txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def get_conditioned_clip_metrics(logits_per_image_list):
    """
    :param logits_per_image_list: A list of containing all logits_per_images.
    Totally N batches, each batch contains B samples. Each logits_per_image should be of shape (B, N*B), already ordered.
    :return:
    """
    metrics = {}
    logits_per_image = torch.cat(logits_per_image_list, dim=0)  # shape: (N*B, N*B)
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(logits_per_image)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)


def remap_indices(merged_img_ids, cap_ids, img2txt_dict, txt2img_dict):
    """
    params:
    merged_img_ids: tensor of shape (M, D)
    cap_ids: tensor of shape (N) (But the ordering might be random)
    img2txt_dict: dict mapping each img_id to a list of cap_ids
    txt2img_dict: dict mappint each cap_id to an img_id (a list of one element)
    text_features: tensor of shape (N, D)
    """
    # so now ideally the cap_ids should be (0, ...N), so do the text_features
    # step2: re-index the merged_image_ids and re-do the mapping in the dict.
    # As the original image ids might just be random numbers, they don't represent the real ordering.

    img_id_mapping = {old_id.item(): new_idx for new_idx, old_id in enumerate(merged_img_ids)}
    reindexed_img_ids = torch.tensor([img_id_mapping[img_id.item()] for img_id in merged_img_ids])

    # Update the img2txt_dict and txt2img_dict with new indices
    new_img2txt_dict = {img_id_mapping[img_id]: [cap_id for cap_id in cap_id_list]
                        for img_id, cap_id_list in img2txt_dict.items()}

    new_txt2img_dict = {cap_id: img_id_mapping[txt2img_dict[cap_id][0]]
                        for cap_id in txt2img_dict.keys()}

    return new_img2txt_dict, new_txt2img_dict


def compute_retrieval(similarity_scores, txt2img, img2txt):
    if isinstance(similarity_scores, tuple):
        i2t_similarity_score, t2i_similarity_score = similarity_scores
    else:
        # Otherwise, treat similarity_scores as a single matrix for t2i
        t2i_similarity_score = similarity_scores.t()
        i2t_similarity_score = similarity_scores

    t2i_ranks = torch.zeros(t2i_similarity_score.shape[0])

    for index, score in enumerate(t2i_similarity_score):
        inds = torch.argsort(score, descending=True)
        t2i_ranks[index] = torch.where(inds == txt2img[index])[0][0]

    # Compute metrics
    tr1 = len(torch.where(t2i_ranks < 1)[0]) / len(t2i_ranks)
    tr5 = len(torch.where(t2i_ranks < 5)[0]) / len(t2i_ranks)
    tr10 = len(torch.where(t2i_ranks < 10)[0]) / len(t2i_ranks)
    t2i_report_dict = {
        "text_to_image_R@1": tr1,
        "text_to_image_R@5": tr5,
        "text_to_image_R@10": tr10,
        "text_to_image_mean_rank": t2i_ranks.mean().item() + 1,
        "text_to_image_median_rank": np.floor(np.median(t2i_ranks.numpy())) + 1
    }

    # comput image -> text
    i2t_ranks = torch.zeros(i2t_similarity_score.shape[0])
    for index, score in enumerate(i2t_similarity_score):
        inds = torch.argsort(score, descending=True)
        # Score
        rank = 1e10
        for i in img2txt[index]:
            tmp = torch.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        i2t_ranks[index] = rank

    # Compute metrics
    ir1 = len(torch.where(i2t_ranks < 1)[0]) / len(i2t_ranks)
    ir5 = len(torch.where(i2t_ranks < 5)[0]) / len(i2t_ranks)
    ir10 = len(torch.where(i2t_ranks < 10)[0]) / len(i2t_ranks)

    i2t_report_dict = {
        "image_to_text_R@1": ir1,
        "image_to_text_R@5": ir5,
        "image_to_text_R@10": ir10,
        "image_to_text_mean_rank": i2t_ranks.mean().item() + 1,
        "image_to_text_median_rank": np.floor(np.median(i2t_ranks.numpy())) + 1
    }
    metrics = {**t2i_report_dict, **i2t_report_dict}
    return metrics


def compute_retrieval_topk(similarity_scores, txt2img, img2txt, topk_indices, num_texts, num_images):
    """
    Reconstruct a (num_images x num_texts) i2t similarity matrix from per-image top-k
    scores/indices, then reuse compute_retrieval(...).
    similarity_scores: Tensor (M, K)      — per-image scores for the selected K texts
    topk_indices:      LongTensor (M, K)  — original global text indices for those K
    """
    i2t_similarity_score = torch.full(
        (num_images, num_texts), -1e10,
        device=similarity_scores.device, dtype=similarity_scores.dtype
    )
    for i in range(num_images):
        i2t_similarity_score[i, topk_indices[i]] = similarity_scores[i]
    return compute_retrieval(i2t_similarity_score, txt2img, img2txt)


def retrieval_on_split(keyword, model, txt_loader, img_loader, img2txt_dict, txt2img_dict, args, epoch, metrics, device,
                       input_dtype, autocast):
    num_txt_samples = txt_loader.num_samples
    num_img_samples = img_loader.num_samples
    all_image_features, all_text_tokens, all_text_features = [], [], []
    all_local_text_tokens = []
    all_img_ids, all_cap_ids = [], []

    with torch.no_grad():
        for i, batch in enumerate(txt_loader):
            texts, cap_id = batch
            texts = texts.to(device=device, non_blocking=True)
            with autocast():
                if args.inference_with_flair:
                    global_text_token, local_text_tokens = unwrap_model(model).encode_text(texts, normalize=False)
                    global_text_token, local_text_tokens = unwrap_model(model).text_post(
                        global_text_token), unwrap_model(model).text_post(local_text_tokens)
                    text_features = F.normalize(global_text_token, dim=-1)
                    all_text_tokens.append(global_text_token.squeeze(1))
                    all_local_text_tokens.append(local_text_tokens)
                elif hasattr(args, "inference_with_flair_topk") and args.inference_with_flair_topk:
                    global_text_token, local_text_tokens = unwrap_model(model).encode_text(texts, normalize=False)
                    global_text_token, local_text_tokens = unwrap_model(model).text_post(
                        global_text_token), unwrap_model(model).text_post(local_text_tokens)
                    text_features = F.normalize(global_text_token, dim=-1)
                    all_text_tokens.append(global_text_token.squeeze(1))
                    all_local_text_tokens.append(local_text_tokens)
                elif hasattr(args, "direct_global_matching") and args.direct_global_matching:
                    global_text_token, _ = unwrap_model(model).encode_text(texts, normalize=False)
                    global_text_token = unwrap_model(model).text_post(global_text_token)
                    text_features = F.normalize(global_text_token, dim=-1)
                else:
                    text_features = unwrap_model(model).encode_text(texts, normalize=True)

                all_text_features.append(text_features.detach().cpu())
                all_cap_ids.append(cap_id.detach().cpu())
        all_text_features_tensor = torch.cat(all_text_features)
        cap_ids = torch.cat(all_cap_ids)

        if args.inference_with_flair:
            mode = "inference_with_flair"
            all_text_tokens_tensor = torch.cat(all_text_tokens)
            similarity_scores, img_ids = compute_similarity_scores_attn_pool(
                model, img_loader, all_text_features_tensor, all_text_tokens_tensor, device, input_dtype, autocast, mode
            )
        elif hasattr(args, "inference_with_flair_topk") and args.inference_with_flair_topk:
            mode = "inference_with_flair_topk"
            top_k = int(args.topk)
            all_text_tokens_tensor = torch.cat(all_text_tokens)
            similarity_scores, img_ids, topk_i2t_ids = compute_similarity_scores_attn_pool(
                model, img_loader, all_text_features_tensor, all_text_tokens_tensor,
                device, input_dtype, autocast, mode, top_k=top_k
            )
        elif hasattr(args, "direct_global_matching") and args.direct_global_matching:
            mode = "direct_global_matching"
            similarity_scores, img_ids = compute_similarity_scores_attn_pool(
                model, img_loader, all_text_features_tensor, torch.empty(0, device=device),
                device, input_dtype, autocast, mode
            )
        else:
            similarity_scores, img_ids = compute_similarity_scores_original_clip(model, img_loader,
                                                                                 all_text_features_tensor, device,
                                                                                 input_dtype,
                                                                                 autocast,
                                                                                 mode='original_clip')
        new_img2txt_dict, new_txt2img_dict = remap_indices(merged_img_ids=img_ids, cap_ids=cap_ids,
                                                           img2txt_dict=img2txt_dict, txt2img_dict=txt2img_dict)

        if hasattr(args, "inference_with_flair_topk") and args.inference_with_flair_topk:
            retrieval_metrics = compute_retrieval_topk(
                similarity_scores=similarity_scores,
                txt2img=new_txt2img_dict,
                img2txt=new_img2txt_dict,
                topk_indices=topk_i2t_ids,
                num_images=num_img_samples,
                num_texts=num_txt_samples
            )
        else:
            retrieval_metrics = compute_retrieval(similarity_scores=similarity_scores,
                                                  txt2img=new_txt2img_dict,
                                                  img2txt=new_img2txt_dict)

        if keyword != '':
            temp_retrieval_metrics = {}
            keyword = keyword + '_'
            for k, v in retrieval_metrics.items():
                temp_retrieval_metrics[keyword + k] = v
            retrieval_metrics = temp_retrieval_metrics

        if "epoch" in metrics:
            metrics.update(
                {**retrieval_metrics,
                 f"{keyword}num_text_samples": num_txt_samples,
                 f"{keyword}num_image_samples": num_img_samples
                 }
            )
        else:
            metrics.update(
                {**retrieval_metrics,
                 f"epoch": epoch,
                 f"{keyword}num_text_samples": num_txt_samples,
                 f"{keyword}num_image_samples": num_img_samples
                 }
            )

    return metrics


def compute_similarity_scores_original_clip(model, img_loader, all_text_features_tensor, device, input_dtype,
                                            autocast, mode='original_clip'):
    all_image_features = []
    all_img_ids = []

    for i, batch in enumerate(img_loader):
        images, img_id = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        all_img_ids.append(img_id.detach().cpu())
     
        with autocast():
            if mode == 'original_clip':
                image_features = unwrap_model(model).encode_image(images, normalize=True)
            elif mode == 'imgcon':
                _, local_image_tokens = unwrap_model(model).encode_image(images)
                local_image_tokens = unwrap_model(model).image_post(local_image_tokens)
                image_features = unwrap_model(model).visual_proj(local_image_tokens.mean(dim=1, keepdim=True), local_image_tokens, local_image_tokens)
                image_features = image_features.squeeze(1)
                image_features = F.normalize(image_features, dim=-1)
            logit_scale = unwrap_model(model).logit_scale.exp()
            all_image_features.append(image_features.detach().cpu())

    all_image_features_tensor = torch.cat(all_image_features)
    img_ids = torch.cat(all_img_ids)

    similarity_scores = logit_scale.cpu() * all_image_features_tensor @ all_text_features_tensor.t()
    return similarity_scores, img_ids


def compute_similarity_scores_attn_pool(model, img_loader, all_text_features_tensor, all_text_tokens_tensor, device,
                                        input_dtype,
                                        autocast, mode, top_k: int = 0):
    logits_per_image_list = []
    all_img_ids = []
    all_topk_ids = []

    for i, batch in enumerate(img_loader):
        images, img_id = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        all_img_ids.append(img_id.detach().cpu())
        with autocast():
            if mode == 'inference_with_flair':
                _, image_embeddings = unwrap_model(model).encode_image(images, normalize=False)
                image_embeddings = unwrap_model(model).image_post(image_embeddings)
                img_features_after_conditioning = unwrap_model(model).visual_proj(
                    all_text_tokens_tensor.unsqueeze(0),
                    image_embeddings,
                    image_embeddings
                )
                img_features_after_conditioning = F.normalize(img_features_after_conditioning, dim=-1).detach().cpu()
                embed_dim = img_features_after_conditioning.shape[-1]
                img_features_after_conditioning = img_features_after_conditioning.contiguous().view(-1, embed_dim)
                logit_scale = unwrap_model(model).logit_scale.exp()
                logits_per_image = (logit_scale.cpu() * torch.einsum('ij,ij->i',
                                   img_features_after_conditioning, all_text_features_tensor)).unsqueeze(0).detach().cpu()
            elif mode == 'inference_with_flair_topk':
                global_image_embeddings, image_embeddings = unwrap_model(model).encode_image(images, normalize=False)
                global_image_embeddings = unwrap_model(model).image_post(global_image_embeddings)
                image_embeddings = unwrap_model(model).image_post(image_embeddings)
                global_image_embeddings = F.normalize(global_image_embeddings, dim=-1)

                per_image_logits = []
                per_image_topk = []
                g = global_image_embeddings.squeeze(1)
                sim_g2t = g @ all_text_features_tensor.t().to(g.device)

                logit_scale = unwrap_model(model).logit_scale.exp()
                for b in range(sim_g2t.size(0)):
                    topk_sim, topk_idx = sim_g2t[b].topk(k=top_k, dim=-1)
                    per_image_topk.append(topk_idx.detach().cpu().unsqueeze(0))

                    topk_text_tokens = all_text_tokens_tensor[topk_idx].to(image_embeddings.device)
                    topk_text_features = all_text_features_tensor[topk_idx].to(image_embeddings.device)

                    img_feat_k = unwrap_model(model).visual_proj(
                        topk_text_tokens.unsqueeze(0),
                        image_embeddings[b:b+1],
                        image_embeddings[b:b+1]
                    ).squeeze(0)
                    img_feat_k = F.normalize(img_feat_k, dim=-1)

                    logits_k = (logit_scale * torch.einsum('ij,ij->i', img_feat_k, topk_text_features)).unsqueeze(0)
                    per_image_logits.append(logits_k.detach().cpu())

                logits_per_image = torch.cat(per_image_logits, dim=0)
                topk_idx_batch = torch.cat(per_image_topk, dim=0)
                all_topk_ids.append(topk_idx_batch)
            elif mode == 'direct_global_matching':
                g_img, _ = unwrap_model(model).encode_image(images, normalize=False)
                g_img = unwrap_model(model).image_post(g_img).squeeze(1)
                g_img = F.normalize(g_img, dim=-1)
                sim = g_img @ all_text_features_tensor.to(g_img.device).t()
                logit_scale = unwrap_model(model).logit_scale.exp()
                logits_per_image = (logit_scale * sim).detach().cpu()
            else:
                raise ValueError(f"Unknown mode: {mode}")

        logits_per_image_list.append(logits_per_image)

    img_ids = torch.cat(all_img_ids)
    similarity_scores = torch.cat(logits_per_image_list)

    if mode == 'inference_with_flair_topk':
        topk_ids = torch.cat(all_topk_ids)
        return similarity_scores, img_ids, topk_ids
    else:
        return similarity_scores, img_ids
