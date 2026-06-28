export HF_ENDPOINT=https://hf-mirror.com

#A test case to evaluate the model's performance on various datasets at the final stage of continual learning.
#srtage8
CUDA_VISIBLE_DEVICES=0 python scripts/test/test_single_image.py --model_checkpoint /data1/dongwen_data/CL-Anomaly/checkpoints/qwen_llava7B_llm_cladmoe_anomaly_stage8 \
--data_dir /home/dongwen/CL-Anomaly/data/ --bench_json webad_stage1_test.json \
--model_base /data1/dongwen_data/Model/llava-onevision-qwen2-7b-ov

CUDA_VISIBLE_DEVICES=0 python scripts/test/test_single_image.py --model_checkpoint /data1/dongwen_data/CL-Anomaly/checkpoints/qwen_llava7B_llm_cladmoe_anomaly_stage8 \
--data_dir /home/dongwen/CL-Anomaly/data/ --bench_json webad_stage2_test.json \
--model_base /data1/dongwen_data/Model/llava-onevision-qwen2-7b-ov

CUDA_VISIBLE_DEVICES=0 python scripts/test/test_single_image.py --model_checkpoint /data1/dongwen_data/CL-Anomaly/checkpoints/qwen_llava7B_llm_cladmoe_anomaly_stage8 \
--data_dir /home/dongwen/CL-Anomaly/data/ --bench_json webad_stage3_test.json \
--model_base /data1/dongwen_data/Model/llava-onevision-qwen2-7b-ov

CUDA_VISIBLE_DEVICES=0 python scripts/test/test_single_image.py --model_checkpoint /data1/dongwen_data/CL-Anomaly/checkpoints/qwen_llava7B_llm_cladmoe_anomaly_stage8 \
--data_dir /home/dongwen/CL-Anomaly/data/ --bench_json mvtec_test.json \
--model_base /data1/dongwen_data/Model/llava-onevision-qwen2-7b-ov

CUDA_VISIBLE_DEVICES=0 python scripts/test/test_single_image.py --model_checkpoint /data1/dongwen_data/CL-Anomaly/checkpoints/qwen_llava7B_llm_cladmoe_anomaly_stage8 \
--data_dir /home/dongwen/CL-Anomaly/data/ --bench_json bmad_test.json \
--model_base /data1/dongwen_data/Model/llava-onevision-qwen2-7b-ov

CUDA_VISIBLE_DEVICES=0 python scripts/test/test_multi_image.py --model_checkpoint /data1/dongwen_data/CL-Anomaly/checkpoints/qwen_llava7B_llm_cladmoe_anomaly_stage8 \
--data_dir /home/dongwen/CL-Anomaly/data/ --bench_json mvtec3d_test.json \
--model_base /data1/dongwen_data/Model/llava-onevision-qwen2-7b-ov

CUDA_VISIBLE_DEVICES=0 python scripts/test/test_multi_image.py --model_checkpoint /data1/dongwen_data/CL-Anomaly/checkpoints/qwen_llava7B_llm_cladmoe_anomaly_stage8 \
--data_dir /home/dongwen/CL-Anomaly/data/ --bench_json real3d_test.json \
--model_base /data1/dongwen_data/Model/llava-onevision-qwen2-7b-ov

CUDA_VISIBLE_DEVICES=0 python scripts/test/test_multi_image.py --model_checkpoint /data1/dongwen_data/CL-Anomaly/checkpoints/qwen_llava7B_llm_cladmoe_anomaly_stage8 \
--data_dir /home/dongwen/CL-Anomaly/data/ --bench_json anomaly_shapenet_test.json \
--model_base /data1/dongwen_data/Model/llava-onevision-qwen2-7b-ov
