#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import random
import numpy as np

import torch
from vllm import LLM
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score
from sal.search import \
    best_of_n, \
    best_of_n_conf, \
    smart_best_of_n, \
    beam_search, \
    beam_search_conf, \
    smart_beam_search, \
    smart_beam_search_conf, \
    dvts
from datasets import Dataset
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

APPROACHES = {
    "beam_search":beam_search,
    "beam_search_smart": smart_beam_search,
    "beam_search_conf": beam_search_conf,
    "beam_search_smart_conf": smart_beam_search_conf,
    "dvts": dvts,
    "best_of_n": best_of_n,
    "best_of_n_smart": smart_best_of_n,
    "best_of_n_conf": best_of_n_conf,
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable optimizations for reproducibility

set_seed(42) 

def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    num_gpus = torch.cuda.device_count()
    print("="*20)
    print("The number of available GPUs:", num_gpus)
    
    # configure approach name
    approach_suffix = "_smart" if config.smart_search else ""
    approach_suffix += "_conf" if config.score_method == 'conf' else ""
    approach_name = config.approach + approach_suffix
    
    if approach_name not in APPROACHES:
        raise ValueError(f"Invalid score method: {config.score_method}")
    approach_fn = APPROACHES[approach_name]
    
    # log the search method and score method
    print("\nUsing " + \
        ("SMART" if config.smart_search else "Baseline") + \
        " search.\nUsing " + \
        ("Confidence" if config.score_method == 'conf' else "PRM") + \
        " based score.\n")
    if config.smart_search:
        print("Threshold:", config.threshold)
    print("N:", config.n)
    print("Beam width:", config.beam_width)
    print("="*20)
    
    if config.smart_search:                
        mp.set_start_method("spawn", force=True)
        slm = LLM(
            model=config.draft_model_path,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=True,
            seed=config.seed,
            tensor_parallel_size=num_gpus,
            max_model_len=2048,
        )
        
        llm = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
        
        if config.score_method == 'prm':
            prm = load_prm(config)

            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "slm": slm, "prm": prm, "llm": llm},
                desc="Running search",
                load_from_cache_file=False,
            )    
        elif config.score_method == 'conf':
            prm = load_prm(config)
            
            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "slm": slm, "prm": prm, "llm": llm},
                desc="Running search",
                load_from_cache_file=False,
            )
        else:
            raise ValueError(f"Invalid score method: {config.score_method}")
    else:
        llm = LLM(
            model=config.model_path,
            revision="main",
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=True,
            seed=config.seed,
            tensor_parallel_size=num_gpus,
            max_model_len=2048,
        )
        
        if config.score_method == 'prm':
            prm = load_prm(config)

            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "llm": llm, "prm": prm},
                desc="Running search",
                load_from_cache_file=False,
            )
        
        elif config.score_method == 'conf':
            prm = load_prm(config)
            
            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "llm": llm, "prm": prm},
                desc="Running search",
                load_from_cache_file=False,
            )    
        else: 
            raise ValueError(f"Invalid score method: {config.score_method}")

    dataset = score(dataset, config)
    save_dataset(dataset, config)
    
    import sys
    sys.path.append("src/evaluation")
    from evaluation.evaluate import evaluate
    if config.approach == "best_of_n" or config.approach == "beam_search":
        subsets = [2**i for i in range(config.n) if 2**i <= config.n]
        keys = []
        for n in subsets:
            keys.extend([f"pred_weighted@{n}", f"pred_maj@{n}", f"pred_naive@{n}"])
    else:
        keys = ["pred"]
        
    dataset, result = evaluate(data_name="math", prompt_type=None, samples=dataset, pred_keys=keys)
    dataset = Dataset.from_list([{k: v for k, v in dict(sample).items() if k != 'pred_completions'} for sample in dataset])
    
    save_dataset(dataset, config)
    
    logger.info(result)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
