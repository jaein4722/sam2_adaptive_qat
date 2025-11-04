CUDA_VISIBLE_DEVICES=4,5,6,7 python projects/sav_finetune/evaluate_sav.py \
  --checkpoint sam2_quantized_custom.pt \
  --video-root ../datasets/sa-v/sav_test/JPEGImages_24fps \
  --annotation-root ../datasets/sa-v/sav_test/Annotations_6fps \
  --video-list ../datasets/sa-v/sav_test/sav_test.txt \
  --output-root eval_results/SAV/sam2.1_PTQ_layerwise_low/sav_test_predictions \
  --metrics-json eval_results/SAV/sam2.1_PTQ_layerwise_low/sav_test_metrics.json