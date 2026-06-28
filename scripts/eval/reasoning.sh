export HF_ENDPOINT=https://hf-mirror.com

# Step 1: Format raw reason outputs
CUDA_VISIBLE_DEVICES=0 python scripts/eval/reason_formatted.py --data_json /data1/dongwen_data/CL-Anomaly/results/cl-anomaly-reason_metrics_mvtec_test.json
CUDA_VISIBLE_DEVICES=0 python scripts/eval/reason_formatted.py --data_json /data1/dongwen_data/CL-Anomaly/results/cl-anomaly-reason_metrics_bmad_test.json
CUDA_VISIBLE_DEVICES=0 python scripts/eval/reason_formatted.py --data_json /data1/dongwen_data/CL-Anomaly/results/cl-anomaly-reason_metrics_real3d_test.json

# Step 2: Evaluate formatted results
CUDA_VISIBLE_DEVICES=0 python scripts/eval/cal_metric_reasoning.py --data_json /data1/dongwen_data/CL-Anomaly/results/cl-anomaly-reason_metrics_mvtec_test_formatted.json
CUDA_VISIBLE_DEVICES=0 python scripts/eval/cal_metric_reasoning.py --data_json /data1/dongwen_data/CL-Anomaly/results/cl-anomaly-reason_metrics_bmad_test_formatted.json
CUDA_VISIBLE_DEVICES=0 python scripts/eval/cal_metric_reasoning.py --data_json /data1/dongwen_data/CL-Anomaly/results/cl-anomaly-reason_metrics_real3d_test_formatted.json
