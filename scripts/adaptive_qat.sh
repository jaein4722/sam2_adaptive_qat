exp_dir=sam2_logs/adaptive_qat_toy_202511104_163300

CUDA_VISIBLE_DEVICES=3 python -m projects.adaptive_qat.train_adaptive_qat \
    --config projects/adaptive_qat/configs/toy.yaml \
    --experiment-dir $exp_dir

# CUDA_VISIBLE_DEVICES=2,3,5 python -m projects.sav_finetune.evaluate_sav \
#     --checkpoint $exp_dir/checkpoints/checkpoint.pt \
#     --video-root ../datasets/sa-v/sav_test/JPEGImages_24fps \
#     --annotation-root ../datasets/sa-v/sav_test/Annotations_6fps \
#     --video-list ../datasets/sa-v/sav_test/sav_test.txt \
#     --output-root $exp_dir/sav_test_predictions \
#     --metrics-json $exp_dir/sav_test_metrics.json