import argparse
import torch
from torch import distributed
import torch.nn.functional as F
import json
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import numpy as np


from flair.params import parse_args
from flair import create_model_and_transforms, get_tokenizer
from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from open_clip_train.precision import get_autocast

def center_square_crop_box(w, h):
    """Return (left, top, right, bottom) for the center square crop based on the shorter side."""
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return (left, top, left + side, top + side), side

def upsample_heatmap_to_size(hm_2d, size):
    """
    hm_2d: float32 array in [0,1], shape (H, W) (e.g., 14x14).
    Returns PIL.Image (L) with values 0..255, resized to (size, size).
    """
    hm = np.clip(hm_2d.astype(np.float32), 0.0, 1.0)
    hm_img = Image.fromarray((hm * 255.0).astype(np.uint8), mode="L")
    return hm_img.resize((size, size), resample=Image.BILINEAR)

def make_red_rgba_from_alpha(alpha_img):
    """
    alpha_img: PIL L (0..255). Create an RGBA image where RGB=(255,0,0) and A=alpha_img.
    """
    a = np.array(alpha_img, dtype=np.uint8)
    r = np.full_like(a, 255, dtype=np.uint8)
    g = np.zeros_like(a, dtype=np.uint8)
    b = np.zeros_like(a, dtype=np.uint8)
    rgba = np.dstack([r, g, b, a])  # (H, W, 4)
    return Image.fromarray(rgba, mode="RGBA")

def overlay_heatmap_on_original(original_img_path, heatmap_2d, out_path, overlay_strength=1.0):
    """
    Align like CLIP and save ONLY the center-cropped overlay (no full-image save).
    - original_img_path: path to RGB image
    - heatmap_2d: numpy float array in [0,1] with shape (14,14) (already normalized)
    - overlay_strength: multiplier for alpha (0..1)
    - out_path: where to save the cropped composited image
    """
    base = Image.open(original_img_path).convert("RGB")
    W, H = base.size
    crop_box, side = center_square_crop_box(W, H)

    # Prepare heat alpha for the crop
    alpha_img = upsample_heatmap_to_size(heatmap_2d, side)
    if overlay_strength != 1.0:
        a = np.array(alpha_img, dtype=np.float32) * float(overlay_strength)
        a = np.clip(a, 0, 255).astype(np.uint8)
        alpha_img = Image.fromarray(a, mode="L")
    red_rgba = make_red_rgba_from_alpha(alpha_img)

    # Compose directly on the cropped region
    crop_rgba = base.crop(crop_box).convert("RGBA")
    out_crop = Image.alpha_composite(crop_rgba, red_rgba)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".jpg", ".jpeg", ".webp"):
        out_crop.convert("RGB").save(out_path, quality=95)
    else:
        out_crop.save(out_path)

def save_attention_heatmaps(attn_weights, save_dir, prefix="attention",
                            original_img_path=None, overlay_strength=1.0):
    """
    Generates and saves:
      1) transparent heatmap PNGs (as before)
      2) aligned overlays saved as the center-cropped image (if original_img_path is given)
    """
    os.makedirs(save_dir, exist_ok=True)

    print("attn_weights shape:", attn_weights.shape)
    if attn_weights.dim() == 4:  # (1, num_heads, B, 197) or similar
        B = attn_weights.shape[2]
        attn_weights = attn_weights.permute(1, 0, 2, 3)
        num_heads = attn_weights.shape[0]
    else:
        B = attn_weights.shape[1]
        num_heads = 1
        prefix = prefix + "_ave"

    cmap = LinearSegmentedColormap.from_list("transparent_red",
                                             [(1, 1, 1, 0), (1, 0, 0, 1)], N=256)

    for b in range(B):
        for head in range(num_heads):
            if num_heads > 1:
                heatmap = attn_weights[head, 0, b, :-1].clone()  # (196,)
            else:
                heatmap = attn_weights[0, b, :-1].clone()        # (196,)

            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()

            heatmap_2d = heatmap.reshape(14, 14).cpu().numpy()

            # standalone transparent heatmap
            plt.figure()
            plt.imshow(heatmap_2d, cmap=cmap, interpolation='bilinear')
            plt.axis('off')
            out_png = os.path.join(
                save_dir, f"{prefix}_caption_{b + 1}_tokens_head_{head + 1}.png"
            )
            plt.savefig(out_png, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

            # cropped overlay
            if original_img_path is not None:
                overlaid_path = os.path.join(
                    save_dir, f"{prefix}_caption_{b + 1}_head_{head + 1}_OVERLAY.png"
                )
                overlay_heatmap_on_original(
                    original_img_path, heatmap_2d, overlaid_path, overlay_strength=overlay_strength
                )

def save_similarity_heatmaps(sim_scores, save_dir, prefix="similarity",
                             original_img_path=None, overlay_strength=1.0):
    """
    Saves standalone heatmaps + aligned cropped overlays (if original_img_path is provided).
    """
    os.makedirs(save_dir, exist_ok=True)
    K = sim_scores.shape[1]

    cmap = LinearSegmentedColormap.from_list("transparent_red",
                                             [(1, 1, 1, 0), (1, 0, 0, 1)], N=256)

    for k in range(K):
        heatmap = sim_scores[:, k].clone()
        heatmap[heatmap < 0] = 0
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        heatmap_2d = heatmap.reshape(14, 14).cpu().numpy()

        # standalone transparent
        plt.figure()
        plt.imshow(heatmap_2d, cmap=cmap, interpolation='bilinear')
        plt.axis('off')
        out_png = os.path.join(save_dir, f"{prefix}_caption_{k + 1}.png")
        plt.savefig(out_png, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()

        # cropped overlay
        if original_img_path is not None:
            overlaid_path = os.path.join(save_dir, f"{prefix}_caption_{k + 1}_OVERLAY.png")
            overlay_heatmap_on_original(
                original_img_path, heatmap_2d, overlaid_path, overlay_strength=overlay_strength
            )

def main(args):
    mode = ''
    print('mode:', mode)
    print("begin main")
    use_original_transform = False
    args = parse_args(args)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_distributed_device(args)

    model_kwargs = {}
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        **model_kwargs,
    )
    model.to(device)
    model.eval()
    
    print("finished creating models")    

    autocast = get_autocast(args.precision)

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
        model = model.module

    # Configure your paths here
    input_image_path = "/p/project1/taco-vlm/xiao4/flair/assets/Arsenal.jpg"
    output_dir = "/p/project1/taco-vlm/xiao4/flair/vis"
    input_text = [
        "One of them is wearing a yellow shirt",
        "The other is wearing a blue shirt",
        "They appear to be engaged in a conversation or discussing something on the field"
    ]
    
    input_image_path = args.vis_input_image_path
    output_dir = args.vis_output_dir
    prefix = args.vis_prefix

    if args.vis_input_text is not None:
        # split on commas and strip whitespace
        input_text = [s.strip() for s in args.vis_input_text.split(",") if s.strip()]
    else:
        print("You MUST specify one or more input texts, separate by commas")
        return


    image = Image.open(input_image_path).convert('RGB')
    image = preprocess_val(image)
    if len(image.size()) == 3:
        image = image.unsqueeze(0)  # add batch dim manually

    tokenizer = get_tokenizer(args.model)
    text = tokenizer(input_text)

    image = image.to(device)
    text = text.to(device)

    with torch.no_grad():
        if args.visualize_patchwise_sim:
            print("visualizing token similarity...")
            with autocast():
                global_image_features, local_image_tokens = model.encode_image(image)
                global_image_features, local_image_tokens = model.image_post(global_image_features), model.image_post(local_image_tokens)
                global_text_features, _ = model.encode_text(text)
                global_text_features = model.text_post(global_text_features)

                B, L, D = local_image_tokens.size()  # B should be 1
                local_image_tokens = local_image_tokens.view(B * L, D).unsqueeze(1)  # (1*L, 1, D)

                expanded_query_features = global_text_features.unsqueeze(0).expand(B * L, -1, -1)  # (1*L, K, D)

                local_image_features = model.visual_proj(expanded_query_features, local_image_tokens, local_image_tokens)  # (B*L, K, D)

                local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)
                expanded_query_features = expanded_query_features / expanded_query_features.norm(dim=-1, keepdim=True)
                sim_scores = torch.einsum("ijk, ijk -> ij", local_image_features, expanded_query_features)  # (1*L, K)

                ts_prefix = prefix + "_ts"
                save_similarity_heatmaps(
                    sim_scores, save_dir=output_dir, prefix=ts_prefix,
                    original_img_path=input_image_path, overlay_strength=1.0
                )

        elif args.visualize_attn_maps:
            print("visualizing attention maps...")
            global_image_features, local_image_tokens = model.encode_image(image)  # (1, L, D)
            global_image_features, local_image_tokens = model.image_post(global_image_features), model.image_post(local_image_tokens)
            global_text_features, _ = model.encode_text(text)  # (B, D)
            global_text_features = model.text_post(global_text_features).unsqueeze(0)  # (1, B, D)
            _, ave_attn_weights = model.visual_proj(
                global_text_features, local_image_tokens, local_image_tokens,
                output_attn_weights=True, average_attn_weights=True
            )
            save_attention_heatmaps(
                ave_attn_weights, save_dir=output_dir, prefix=prefix,
                original_img_path=input_image_path, overlay_strength=1.0
            )

        else:
            print("Please either visualiaze the attention maps or patch-wise similarity")


if __name__ == "__main__":
    main(sys.argv[1:])
