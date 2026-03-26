#   --backbone_weights /hdd/wangty/new_task/LLaMA-Factory/task/test_baseline/dinov3_vit/weight/model.safetensors \  /hdd/wangty/diffuser_workdir/dataset/origin_zjppt_c3-4.jsonl
CUDA_VISIBLE_DEVICES=6 python infer.py \
  --input_jsonl /hdd/wangty/diffuser_workdir/dataset/gen_zjppt_c3-4.jsonl \
  --output_jsonl /hdd/wangty/diffuser_workdir/result/gen_zjppt_c3-4.jsonl \
  --model dinov3_vit \
  --ckpt /hdd/wangty/new_task/LLaMA-Factory/task/baseline/dinov3_vit/zjppt_tra_b64/best.pth \
  --num_targets 1 \
  --num_classes 2 \
  --batch_size 64