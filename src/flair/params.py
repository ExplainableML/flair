"""
This code is adapted from OpenCLIP:
https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/params.py

The code integrates additional modifications and extensions to support the FLAIR models.
Original authors: ML Foundations.
"""
import argparse
import ast


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        )
    )
    parser.add_argument(
        "--data-root-dir",
        type=str,
        default='',
        help=(
            "Root directory to your dataset, especially the COCO dataset."
        )
    )
    parser.add_argument(
        "--cc3m-train-retrieval-dir",
        type=str,
        default='',
        help=(
            "Root directory to the train retrieval dataset subsampled from cc3m."
        )
    )
    parser.add_argument(
        "--sharegpt4v-retrieval-dir",
        type=str,
        default='',
        help=(
            "Root directory to the share4v dataset."
        )
    )
    parser.add_argument(
        "--dci-retrieval-dir",
        type=str,
        default='',
        help=(
            "Root directory to the train dci daatset."
        )
    )
    parser.add_argument(
        "--iiw-retrieval-dir",
        type=str,
        default='',
        help=(
            "Root directory to the image in words dataset."
        )
    )
    parser.add_argument(
        "--docci-retrieval-dir",
        type=str,
        default='',
        help=(
            "Root directory to fine-grained docci retrieval."
        )
    )
    parser.add_argument(
        "--urban-1k-retrieval-dir",
        type=str,
        default='',
        help=(
            "Root directory to fine-grained urban-1k retrieval."
        )
    )
    parser.add_argument(
        "--zeroshot-eval-datasets",
        type=str,
        default=None,
        help=(
            "Datasets that you want to do retrieval."
        )
    )
    parser.add_argument(
        "--coco-data-root-dir",
        type=str,
        default='',
        help=(
            "Root directory to the COCO dataset."
        )
    )
    parser.add_argument(
        "--flickr-data-root-dir",
        type=str,
        default='',
        help=(
            "Root directory to the flickr datasets (but we simply use the root of the whole dataset)."
        )
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to file(s) with validation data.",
    )
    parser.add_argument(
        "--dict-root-dir",
        type=str,
        default=None,
        help="Path to the preprocessed dictionaries to filter the dataset.",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input size for the zero-shot evaluation task",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output txt file for documenting the results",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--train-val-num-samples",
        type=int,
        default=None,
        help="Number of samples in train eval dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--train-eval-data",
        type=str,
        default=None,
        help="Path to evaluation set within train data with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto", "coco", "flickr"],
        default="coco",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--train-dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto", "coco", "flickr"],
        default="csv",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--val-dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto", "coco", "flickr"],
        default="coco",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--retrieval-dataset-type",
        choices=["coco", "flickr"],
        default="coco",
        help="Which type of dataset for the retrieval task."
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--inference-with-flair",
        default=False,
        action="store_true",
        help="If set to true, then we use FLAIR way of inference."
    )
    parser.add_argument(
        "--inference-with-flair-topk",
        default=False,
        action="store_true",
        help="If set to true, we use global matching to get the top-k similarity texts for one image, then we use FLAIR's way to re-rank the features."
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=128,
        help="K texts per image to keep when --inference-with-flair-topk is enabled."
    )
    parser.add_argument(
        "--direct-global-matching",
        default=False,
        action="store_true",
        help="If set to true, even in FLAIR, we directly match the global image and text features to do the retrieval (Original CLIP's way)"
    )
    parser.add_argument(
        "--fixed-merged-num",
        default=False,
        action="store_true",
        help="If true, then we fix the merging number in variable merge."
    )
    parser.add_argument(
        "--all-subsequent-merged",
        default=False,
        action="store_true",
        help="If enabled, then we only merge subsequent captions."
    )
    parser.add_argument(
        "--use-flair-loss",
        default=False,
        action="store_true",
        help="Whether to use the text-conditioned sigmoid loss or not."
    )
    parser.add_argument(
        "--add-mps-loss",
        default=False,
        action="store_true",
        help="Whether to add the multi-positive loss or not."
    )
    parser.add_argument(
        "--directly-use-attn-weights",
        default=False,
        action="store_true",
        help="Directly use attn weights for segmentation or not."
    )
    parser.add_argument(
        "--sampled-textcon-siglip-loss",
        default=False,
        action="store_true",
        help="Whether to use the sampled textcon siglip loss or not."
    )
    parser.add_argument(
        "--add-global-loss-textcon",
        default=False,
        action="store_true",
        help="Whether to add the global loss or not."
    )
    parser.add_argument(
        "--only-global-loss-attn-pool",
        default=False,
        action="store_true",
        help="Whether to add the global loss or not."
    )
    parser.add_argument(
        "--add-global-loss-textcon-with-attn-pool",
        default=False,
        action="store_true",
        help="Whether to add the global loss with extra attn pool or not."
    )
    parser.add_argument(
        "--pixelprose",
        default=False,
        action="store_true",
        help="Set to true to remind te webdataset to adapt to the pixelprose format."
    )
    parser.add_argument(
        "--datacomp",
        default=False,
        action="store_true",
        help="Set to true to remind te webdataset to adapt to the datacomp format."
    )
    parser.add_argument(
        "--add-global-loss",
        default=False,
        action="store_true",
        help="Whether to add the global loss implementation or not."
    )
    parser.add_argument(
        "--add-intra-sample-loss",
        default=False,
        action="store_true",
        help="Whether to add the intra sample loss or not."
    )
    parser.add_argument(
        "--cross-con",
        default=False,
        action="store_true",
        help="Using Cross conditioned model and loss or not."
    )
    parser.add_argument(
        "--text-con-with-down-proj",
        default=False,
        action="store_true",
        help="Using the new Text-conditioned model with down-proj or not."
    )
    parser.add_argument(
        "--use-csa",
        default=False,
        action="store_true",
        help='For segmentation evaluation use correlative self-attention by SCLIP.'
    )
    parser.add_argument(
        "--seg-model",
        type=str,
        default="",
        help=''' For segmentation evaluation, name of the openAI model otherwise evaluate from resume checkpoint
                    ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']'''
    )
    parser.add_argument(
        "--show-dir",
        type=str,
        default="",
        help=''' Directory for saving the visualizations for segmentation.'''
    )
    parser.add_argument(
        "--max-merged-num", type=int, default=3, help="Maximum number of merging."
    )
    parser.add_argument(
        "--cross-con-with-post-process",
        default=False,
        action="store_true",
        help="Using Cross conditioned model with post processing and loss or not."
    )
    parser.add_argument(
        "--add-global-loss-in-sampled-cross-con",
        default=False,
        action="store_true",
        help="Using Cross conditioned model with post processing and loss or not."
    )
    parser.add_argument(
        "--cross-con-with-down-proj",
        default=False,
        action="store_true",
        help="Whether to use the corss conditioned model with down-projected embed dim or not."
    )
    parser.add_argument(
        "--txt-con-attn-pool",
        default=False,
        action="store_true",
        help="To use text-conditioned attention pooling or not."
    )
    parser.add_argument(
        "--evaluate-as-original-clip",
        default=False,
        action="store_true",
        help="Though text-conditioned, still evaluate in the original CLIP way."
    )
    parser.add_argument(
        "--evaluate-as-text-conditioned",
        default=False,
        action="store_true",
        help="Using Text-conditioned and evaluated as text-conditioned way."
    )
    parser.add_argument(
        "--retrieval-coco",
        default=False,
        action="store_true",
        help="If true, then we enable the coco retrieval task.")

    parser.add_argument(
        "--retrieval-dci",
        default=False,
        action="store_true",
        help="If true, then we enable the dci retrieval task.")

    parser.add_argument(
        "--retrieval-iiw",
        default=False,
        action="store_true",
        help="If true, then we enable the iiw retrieval task.")

    parser.add_argument(
        "--use-finegrained-iiw",
        default=True,
        action="store_true",
        help="If set to true, under the condition that we enable iiw, we further use the fine-grained iiw mode.")

    parser.add_argument(
        "--retrieval-sharegpt4v-1k",
        default=False,
        action="store_true",
        help="If true, then we enable the sharegpt4v retrieval task with 1k data size.")

    parser.add_argument(
        "--retrieval-sharegpt4v-10k",
        default=False,
        action="store_true",
        help="If true, then we enable the sharegpt4v retrieval task with 10k data size.")

    parser.add_argument(
        "--retrieval-flickr",
        default=False,
        action="store_true",
        help="If true, then we enable the flickr retrieval task.")
    parser.add_argument(
        "--add-global-loss-cross-con",
        default=False,
        action="store_true",
        help="If true, then we add global loss to cross-condition setting.")
    parser.add_argument(
        "--add-global-loss-cross-con-mean",
        default=False,
        action="store_true",
        help="If true, then we add global loss to cross-condition mean setting.")
    parser.add_argument(
        "--add-pooled-global-loss-cross-con",
        default=False,
        action="store_true",
        help="If true, then we add pooled global loss to cross-condition setting.")
    parser.add_argument(
        "--txt-self-attn",
        default=False,
        action="store_true",
        help="If true, then we add pooled attn pooling also to generate the text embeddings.")

    parser.add_argument(
        "--retrieval-urban-1k",
        default=False,
        action="store_true",
        help="If true, then we enable the urban-1k retrieval task.")
    parser.add_argument(
        "--retrieval-data-cc3m-train",
        default=False,
        action="store_true",
        help="If true, then we enable the cc3m retrieval task.")
    parser.add_argument(
        "--retrieval-docci",
        default=False,
        action="store_true",
        help="If true, then we enable the DOCCI retrieval task.")

    parser.add_argument(
        "--use-original-openclip-csv-dataset",
        default=False,
        action="store_true",
        help="Whether to use the original openclip csv dataset or not, if false, then use new csv dataset."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--flickr-val-or-test",
        type=str,
        default='val',
        choices=['val', 'testing'],
        help="Which dataset to be used for inference, default choices are val or test.",
    )
    parser.add_argument(
        "--huggingface-model-name",
        type=str,
        default="",
        help="Name of the huggingface model."
    )
    parser.add_argument(
        "--huggingface-repo-name",
        type=str,
        default="",
        help="Name of the huggingface repo."
    )
    parser.add_argument(
        "--ablation-negative-type",
        type=str,
        default=None,
        choices=['ijj', 'iji', 'ijk', 'iij', 'intra'],
        help="Denote the ablation negative type for the abation study.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="Log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--random-select-text-tokens",
        action="store_true",
        default=False,
        help="To randomly select the text tokens for the img-con text tokens pooling or not.",
    )
    parser.add_argument(
        "--use-siglip",
        action="store_true",
        default=False,
        help="Whether to use the siglip loss for text conditioned model or not.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--epochs-cooldown", type=int, default=None,
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default='cosine',
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--lr-cooldown-end", type=float, default=0.0,
        help="End learning rate for cooldown schedule. Default: 0"
    )
    parser.add_argument(
        "--lr-cooldown-power", type=float, default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)"
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--negative-sampling-in-forward",
        action="store_true",
        default=False,
        help="Doing negative sampling in SampledCrossConSigLipLoss in the forward function.",
    )
    parser.add_argument(
        "--negative-sampling-in-gpu",
        action="store_true",
        default=False,
        help="Doing negative sampling in SampledCrossConSigLipLoss inside the GPU.",
    )
    parser.add_argument(
        "--merged-captions-num", type=int, default=1, help="Number of merged captions."
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="Path to latest checkpoint (default: none).",
    )
    parser.add_argument(
        "--coco-random-subset",
        default=None,
        type=int,
        help="Randomly subset how many k number of samples in COCO dataset."
    )
    parser.add_argument(
        "--coco-sliding-window",
        default=None,
        type=int,
        help="Number to specify the kth window to be used."
    )

    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--caption-sampling-mode",
        choices=["only_raw_caption", "raw_and_random", "random", "short", "dreamlip", "short-long-mixed-random", "diverse_sampling"],
        default="random",
        help="Floating point precision."
    )
    parser.add_argument(
        "--negative-type",
        choices=["ijj", "iji"],
        default="ijj",
        help="Main type of negatives in text conditioned pooling."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )

    parser.add_argument(
        "--target-model",
        type=str,
        default="RN50",
        help="Name of the target vision backbone to be copied.",
    )

    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--mixed-sampling-cross-con",
        default=False,
        action='store_true',
        help="Whether to use the mixed sampling mode for cross-con or not.",
    )
    parser.add_argument(
        "--use-dreamlip-loss",
        default=False,
        action='store_true',
        help="If true, then we use dreamlip loss.",
    )
    parser.add_argument(
        "--dreamlip-model",
        default=False,
        action='store_true',
        help="If true, then we use dreamlip model.",
    )
    parser.add_argument(
        "--normal-clip-with-multi-cap",
        default=False,
        action='store_true',
        help="If the CLIP model is using multi captions or not.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        "--text-conditioned-loss",
        default=False,
        action='store_true',
        help="Whether to use the text-conditioned loss for text conditioned CLIP model.",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset.')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset.')
    parser.add_argument(
        '--image-interpolation',
        default=None, type=str, choices=['bicubic', 'bilinear', 'random'],
        help="Override default image resize interpolation."
    )
    parser.add_argument(
        '--image-resize-mode',
        default=None, type=str, choices=['shortest', 'longest', 'squash'],
        help="Override default image resize (& crop) mode during inference."
    )
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="Calculate loss w/ local features @ global (instead of realizing full global @ global matrix)."
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="Enable full distributed gradient for feature gather."
    )
    parser.add_argument(
        '--force-image-size', type=int, nargs='+', default=None,
        help='Override default image size.'
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper.",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action='store_true',
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'.",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action='store_true',
        help="torch.compile() the model, requires pytorch 2.0 or later.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only.",
    )
    parser.add_argument(
        "--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps."
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb."
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='open-clip',
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there."
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--num-sampled-captions", type=int, default=8, help="Number of sampled captions per image."
    )
    parser.add_argument(
        "--num-sampled-long-captions", default=0, type=int,
        help="Number of sampled long captions per image. Set to 0 for the default setting."
    )

    parser.add_argument(
        "--merged-num", type=int, default=1, help="Merged number"
    )
    parser.add_argument(
        "--add-attn-pooling",
        default=False,
        action="store_true",
        help="To add attn-pooling in the end of vision encoder or not, note that this will set the original pooling to Identity."
    )
    parser.add_argument(
        "--text-con-attn-pool",
        default=False,
        action="store_true",
        help="Indicating whether the model is using text conditioning or not. Must be specified if using text-conditioned models."
    )
    parser.add_argument(
        "--visualize-patchwise-sim",
        default=False,
        action='store_true',
        help="If true, we visualize the patch-wise similarity",
    )
    parser.add_argument(
        "--visualize-attn-maps",
        default=False,
        action='store_true',
        help="If true, we visualize the attention maps",
    )
    parser.add_argument(
        "--vis-input-image-path",
        type=str,
        default=None,
        help="Input image path to create the attention or token similarity visualization",
    )
    parser.add_argument(
        "--vis-output-dir",
        type=str,
        default=None,
        help="Output directory that stores all the visualizations",
    )
    parser.add_argument(
        "--vis-input-text",
        type=str,
        default=None,
        help="Input Texts for attention or similarity visualization. If multiple texts, each should be splitted by comma.",
    )
    parser.add_argument(
        "--vis-prefix",
        type=str,
        default=None,
        help="A lovely prefix you can choose to distibuish between different attention maps from different images.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action='store_true',
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n text tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action='store_true',
        help="Freeze LayerNorm running stats in text tower for any locked layers.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--coca-caption-loss-weight",
        type=float,
        default=2.0,
        help="Weight assigned to caption loss in CoCa."
    )
    parser.add_argument(
        "--coca-contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight assigned to contrastive loss when training CoCa."
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        default=None,
        help="Optinoally sync with a remote path specified by this arg.",
    )
    parser.add_argument(
        "--remote-sync-frequency",
        type=int,
        default=300,
        help="How frequently to sync to a remote directly if --remote-sync is not None.",
    )
    parser.add_argument(
        "--remote-sync-protocol",
        choices=["s3", "fsspec"],
        default="s3",
        help="How to do the remote sync backup if --remote-sync is not None.",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one."
    )
    parser.add_argument(
        "--distill-model",
        default=None,
        help='Which model arch to distill from, if any.'
    )
    parser.add_argument(
        "--distill-pretrained",
        default=None,
        help='Which pre-trained weights to distill from, if any.'
    )
    parser.add_argument(
        "--use-bnb-linear",
        default=None,
        help='Replace the network linear layers from the bitsandbytes library. '
        'Allows int8 training/inference, etc.'
    )
    parser.add_argument(
        "--siglip",
        default=False,
        action="store_true",
        help='Use SigLip (sigmoid) loss.'
    )
    parser.add_argument(
        "--same-row",
        default=False,
        action="store_true",
        help='If same row, the use a different Text-conditioned SigLIP loss, it also means that you are using a different model.'
    )


    args = parser.parse_args(args)

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
