import json
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
import torch, json, time, os, argparse, warnings

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def cal_reasoning(data):
    rouge = Rouge()
    sbert = SentenceTransformer('paraphrase-mpnet-base-v2')

    rouge_scores, sbert_scores = [], []

    for rec in data:
        answers   = [a.replace('<image>', '') for a in rec['answers']]
        responses = [r.replace('<image>', '') for r in rec['response']]

        for ans, resp in zip(answers, responses):
            # ROUGE-L
            scores = rouge.get_scores(resp, ans)[0]['rouge-l']
            rouge_scores.append((scores['r'] + scores['f'] + scores['p']) / 3)

            # SBERT
            emb = sbert.encode([ans, resp], convert_to_tensor=True)
            sbert_scores.append(util.pytorch_cos_sim(emb[0], emb[1]).item())

    avg_r   = sum(rouge_scores) / len(rouge_scores)
    avg_s   = sum(sbert_scores) / len(sbert_scores)

    print(f'Global Mean ROUGE-L: {avg_r:.4f}')
    print(f'Global Mean SBERT:   {avg_s:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test detection performance of the model")
    parser.add_argument("--data_json", type=str, default='/data1/dongwen_data/CL-Anomaly/results/cl-anomaly-reason_metrics_real3d_test_formatted.json', help="Path to your data directory")
    args = parser.parse_args()

    data = load_data(args.data_json)

    print(args.data_json)
    cal_reasoning(data)
