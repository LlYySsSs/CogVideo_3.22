CUDA_VISIBLE_DEVICES=2 python inference/cli_demo.py \
--prompt="A curious tabby cat explores a backyard garden, sniffing at colorful flowers and investigating the base of a large, leafy tree." \
--model_path="/data1/yexiaoyu/cogvideox-2b" \
--seed=123
# --lora_path="/home/yexiaoyu/CogVideo/finetune/cogvideox-lora-single-node/pytorch_lora_weights.safetensors" \
# --lora_rank=128 \
