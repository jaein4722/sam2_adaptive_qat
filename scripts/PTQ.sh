python -m projects.layerwise_quantization.scripts.run_ptq \
    --config-name configs/sam2.1/sam2.1_hiera_b+.yaml \
    --checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
    --output sam2_quantized_low.pt \
    --bits 8 \
    --metadata "$(cat projects/layerwise_quantization/configs/ptq/bit_recommendation_v4_aggressive.json)"