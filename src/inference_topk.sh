torchrun --nproc_per_node 1 -m main \
    --model ViT-B-16-FLAIR \
    --huggingface-repo-name xiaorui638/flair \
    --huggingface-model-name flair-cc3m-recap.pt \
    --inference-with-flair-topk \
    --topk 128 \
    --coco-data-root-dir  ./datasets/coco \
    --flickr-data-root-dir  ./datasets/flickr30k-images \
    --iiw-retrieval-dir  ./datasets/imageinwords/ \
    --docci-retrieval-dir  ./datasets/docci \
    --urban-1k-retrieval-dir  ./datasets/Urban1k \
    --sharegpt4v-retrieval-dir ./datasets/share4v \
    --retrieval-coco \
    --retrieval-flickr \
    --retrieval-docci \
    --retrieval-iiw \
    --retrieval-urban-1k \
    --retrieval-sharegpt4v-1k \
    --retrieval-sharegpt4v-10k \
    --batch-size 128 \
    --precision amp \
    --workers 25 \
    


    



