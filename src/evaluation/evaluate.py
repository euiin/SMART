import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from grader import math_equal_process

from parser import parse_ground_truth, extract_answer
from utils import load_jsonl
from python_executor import PythonExecutor


def get_result(samples: list=None, file_path: str=None):
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]
        
    pred_keys = samples[0]['metrics']
    score_mat = np.array([s['correct'] for s in samples])
    col_means = score_mat.mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    result_json = {
        pred_key: score for pred_key, score in zip(pred_keys, mean_score)
    }

    return result_json



def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False, pred_keys: list=None):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    # parse gt
    for sample in samples:
        _, sample['gt'] = parse_ground_truth(sample, data_name)
        sample['metrics'] = pred_keys
        sample['preds'] = [extract_answer(sample[pred_key], data_name) for pred_key in pred_keys] # TODO: should be fixed
        

    # calculate scores for final prediction
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['preds']]

    scores = []
    timeout_cnt = 0 

    progress_bar = tqdm(total=len(params), desc="Extract preds for each completion")
    for idx, pred, gt in params:
        try:
            result = math_equal_process((idx, pred, gt))
            scores.append(result)
        except TimeoutError as error:
            print(error)
            scores.append(False)
            timeout_cnt += 1
        except Exception as error:
            print(error)
            exit()
        progress_bar.update(1)
    progress_bar.close()
    
    # calculate scores for each completions
    for sample in samples:
        sample['pred_completions'] = [extract_answer(completion, data_name) for completion in sample['completions']]
        
    # calculate scores for final prediction
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred_completions']]
    completion_scores = []
    timeout_cnt = 0 

    progress_bar = tqdm(total=len(samples), desc="Extract correctness for each completion")
    for idx, pred, gt in params:
        try:
            result = math_equal_process((idx, pred, gt))
            completion_scores.append(result)
        except TimeoutError as error:
            print(error)
            completion_scores.append(False)
            timeout_cnt += 1
        except Exception as error:
            print(error)
            exit()
        progress_bar.update(1)
    progress_bar.close()
    
        

    idx = 0
    score_mat = []
    for sample in samples:
        sample['correct'] = scores[idx: idx+len(sample['preds'])]
        assert len(sample['correct']) == len(sample['preds'])
        score_mat.append(sample['correct'])
        idx += len(sample['preds'])

    idx = 0
    for sample in samples:
        sample['correct_completions'] = completion_scores[idx: idx+len(sample['pred_completions'])]
        assert len(sample['correct_completions']) == len(sample['pred_completions'])
        idx += len(sample['pred_completions'])

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    # output mean of each column of scores
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['preds'][-1]]),
        "acc": {pred_key: score for pred_key, score in zip(pred_keys, mean_score)}
    }


    return samples, result_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tool-integrated")
    parser.add_argument("--file_path", type=str, default=None)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path,
             max_num_samples=args.max_num_samples, execute=args.execute)
