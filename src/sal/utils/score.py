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


import math
from typing import Literal

from datasets import Dataset
from tqdm import tqdm
import numpy as np 

from sal.config import Config
from sal.utils.math import (
    compute_maj_pred,
    compute_naive_pred,
    compute_weighted_pred,
    extract_completion_answers,
    subsample_completions,
)

def calculate_confidence_score(answer_tokens_logprobs_list):
    """
    answer_tokens_logprobs_list에서 logprob 값을 합산하여 log-likelihood 및 likelihood를 계산하는 함수.

    Args:
        answer_tokens_logprobs_list (list of dict): [{token_id: Logprob(logprob=value, ...)}, {...}, ...]

    Returns:
        tuple: (log_likelihood, likelihood)
    * mean : likelihood(norm)
    * sum할때 -> answer_tokens_logprobs_list의 갯수를 뽑을 수 있는데 = T, 
    각 generation hyperparameter=e 
    e^T
    """
    log_likelihood_of_completion = sum(next(iter(logprob.values())).logprob for logprob in answer_tokens_logprobs_list)
    
    likelihood_score = np.exp(log_likelihood_of_completion)
    
    T = len(answer_tokens_logprobs_list) if len(answer_tokens_logprobs_list) > 0 else 1
    likelihood_mean_score = np.exp(log_likelihood_of_completion / T)
    
    probs_mean_score = np.mean([np.exp(next(iter(logprob.values())).logprob) for logprob in answer_tokens_logprobs_list])
    
    return [likelihood_score, likelihood_mean_score, probs_mean_score]


def aggregate_scores(
    scores: list[float], agg_strategy: Literal["min", "prod", "last"]
) -> float:
    if agg_strategy == "min":
        return min(scores)
    elif agg_strategy == "prod":
        return math.prod(scores)
    elif agg_strategy == "last":
        return scores[-1]
    else:
        raise ValueError(f"Invalid aggregation strategy: {agg_strategy}")


def score(dataset: Dataset, config: Config) -> Dataset:
    dataset = dataset.map(
        lambda x: {"agg_scores": [aggregate_scores(s, "last") for s in x["scores"]]}
    )
    subsets = [2**i for i in range(config.n) if 2**i <= config.n]
    for n in tqdm(subsets, desc="Computing majority & weighted predictions"):
        dataset = dataset.map(
            subsample_completions,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Subsample {n}",
        )
        dataset = dataset.map(
            extract_completion_answers,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Extract answers {n}",
        )
        dataset = dataset.map(
            compute_weighted_pred,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Compute weighted pred {n}",
        )
        dataset = dataset.map(
            compute_maj_pred,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Compute majority pred {n}",
        )
        dataset = dataset.map(
            compute_naive_pred,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Compute naive pred {n}",
        )
        # Nuke unused columns to keep dataset lean
        dataset = dataset.remove_columns(
            [f"completions@{n}", f"agg_scores@{n}", f"preds@{n}"]
        )
    return dataset
