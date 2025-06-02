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
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import (
    create_branch,
    list_repo_commits,
    repo_exists,
)

from sal.config import Config

logger = logging.getLogger()


def get_dataset(config: Config) -> Dataset:
    if config.dataset_name == "sampled_math500":
        dataset = pd.read_csv(config.dataset_name + ".csv")
        dataset = Dataset.from_pandas(dataset)
    elif config.dataset_name == "prm_math500":
        # load from jsonl file
        dataset = pd.read_json("train.jsonl", lines=True)
        dataset = Dataset.from_pandas(dataset)
    else:
        dataset = load_dataset(config.dataset_name, split=config.dataset_split, trust_remote_code=True)

    if config.dataset_start is not None and config.dataset_end is not None:
        dataset = dataset.select(range(config.dataset_start, config.dataset_end))
    if config.num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), config.num_samples)))

    return dataset


def save_dataset(dataset, config):
    if config.push_to_hub:
        # Since concurrent pushes can get rejected by the Hub, we make several attempts to push the dataset with try/except
        for _ in range(20):
            try:
                # Create branch from the repo's initial commit.
                # This is needed to avoid branching from a commit on main that already has data
                if repo_exists(config.hub_dataset_id, repo_type="dataset"):
                    initial_commit = list_repo_commits(
                        config.hub_dataset_id, repo_type="dataset"
                    )[-1]
                    create_branch(
                        repo_id=config.hub_dataset_id,
                        branch=config.revision,
                        revision=initial_commit.commit_id,
                        exist_ok=True,
                        repo_type="dataset",
                    )
                url = dataset.push_to_hub(
                    config.hub_dataset_id,
                    revision=config.revision,
                    split="train",
                    private=config.hub_dataset_private,
                    commit_message=f"Add {config.revision}",
                )
                break
            except Exception as e:
                logger.error(f"Error pushing dataset to the Hub: {e}")
                time.sleep(5)
        logger.info(f"Pushed dataset to {url}")
    else:
        if config.output_dir is None:
            config.output_dir = f"data/{config.model_path}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Name the folder based on the approach used
        if config.draft_model_path is not None:
            if config.score_method == 'prm':
                folder_name = "smart_prm"
            elif config.score_method == 'conf':
                folder_name = "smart_conf"
        else: 
            if config.score_method == 'prm':
                folder_name = "base_prm"
            elif config.score_method == 'conf':
                folder_name = "base_conf"
        
        # Name the appoarch in likelihood score
        if config.beam_width == 1:
            approach_fn = "best_of_n"
        else: 
            approach_fn = config.approach

        # Save the dataset to a jsonl file by splitting the dataset or not
        if config.dataset_start is not None and config.dataset_end is not None:
            dataset.to_json(
                f"{config.output_dir}/{folder_name}/{approach_fn}_completions_T-{config.temperature}--top_p-{config.top_p}--n-{config.n}--m-{config.beam_width}--iters-{config.num_iterations}--look-{config.lookahead}--seed-{config.seed}--agg_strategy--{config.agg_strategy}_{config.num_samples}_datasplit_{config.dataset_start}-{config.dataset_end}.jsonl", lines=True
            )
            logger.info(
                f"Saved completions to {config.output_dir}/{folder_name}/{approach_fn}_completions_T-{config.temperature}--top_p-{config.top_p}--n-{config.n}--m-{config.beam_width}--iters-{config.num_iterations}--look-{config.lookahead}--seed-{config.seed}--agg_strategy--{config.agg_strategy}_{config.num_samples}_datasplit_{config.dataset_start}-{config.dataset_end}.jsonl"
            )
        else:
            dataset.to_json(
                    f"{config.output_dir}/{folder_name}/{approach_fn}_completions_T-{config.temperature}--top_p-{config.top_p}--n-{config.n}--m-{config.beam_width}--iters-{config.num_iterations}--look-{config.lookahead}--seed-{config.seed}--agg_strategy--{config.agg_strategy}_threshold-{config.threshold}_{config.num_samples}.jsonl", lines=True
                )
            logger.info(
                f"Saved completions to {config.output_dir}/{folder_name}/{approach_fn}_completions_T-{config.temperature}--top_p-{config.top_p}--n-{config.n}--m-{config.beam_width}--iters-{config.num_iterations}--look-{config.lookahead}--seed-{config.seed}--agg_strategy--{config.agg_strategy}_threshold-{config.threshold}_{config.num_samples}.jsonl"
            )