python weight_clipping.py \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
  --ckpt ./checkpoints/sam2.1_hiera_base_plus.pt \
  --device cpu \
  --mode per_layer_percentile \
  --percentile 99 \
  --output ./checkpoints/sam2.1_hiera_base_plus.clipped_layer.pth