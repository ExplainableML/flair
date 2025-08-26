torchrun --nproc_per_node 1 -m visualization \
    --model ViT-B-16-FLAIR \
    --pretrained /p/project1/taco-vlm/xiao4/pretrained_models/flair-cc3m-recap.pt \
    --visualize-attn-maps \
    --vis-prefix Arsenal \
    --vis-input-image-path ../assets/Arsenal.jpg \
    --vis-output-dir ../vis \
    --vis-input-text "One of them is wearing a yellow shirt, The other is wearing a blue shirt, They appear to be engaged in a conversation or discussing something on the field" \
    --precision amp \
    --workers 4 \