CUDA_VISIBLE_DEVICES=2,5 python evaluate.py \
  --data_root ../datasets/sa-1b_split/test \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
  --ckpt sam2_logs/adaptive_qat_toy_202511104_122400/checkpoints/checkpoint.pt \
  --eval_mode miou_point \
  --out_dir ./eval_results/SA1B/sam2.1_adaptive_qat \
  --viz_percent 5 \
  --viz_metric miou