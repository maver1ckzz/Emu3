#   --backbone_weights /hdd/wangty/new_task/LLaMA-Factory/task/test_baseline/dinov3_vit/weight/model.safetensors \  /hdd/wangty/diffuser_workdir/dataset/origin_zjppt_c3-4.jsonl
task="zjksjg_l"
data_type="gen"  #origin
input_data="/hdd/wangty/diffuser_workdir/dataset/${data_type}_${task}_c3-4.jsonl"
output_data="/hdd/wangty/diffuser_workdir/result/${data_type}_${task}_c3-4.jsonl"
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --input_jsonl $input_data \
  --output_jsonl $output_data \
  --model dinov3_vit \
  --ckpt /hdd/wangty/new_task/LLaMA-Factory/task/baseline/dinov3_vit/zjksjg_l_b128/best.pth \
  --num_targets 1 \
  --num_classes 2 \
  --batch_size 64