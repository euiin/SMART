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
import copy
import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM

from .utils import Beam, build_conv, generate_k_steps, last, generate_k_steps_for_llm

logger = logging.getLogger()
from sal.utils.score import aggregate_scores

from transformers import AutoTokenizer

def _beam_search(batch_of_prompts, config: Config, slm: LLM, prm: PRM, llm: None) -> tuple[list[Beam], int]:
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )
    
    beams: list[Beam] = []
    for prompt in batch_of_prompts:
        for i in range(config.n):
            beams.append(
                Beam(
                    prompt=prompt,
                    index=i,
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
                    pruned=False,
                    completed=False,  # New flag to track completion
                    stop_reasons=None,
                    history=[],
                    best_scores=[],
                    all_scores=[],
                    previous_text=None,
                    completion_tokens=[],
                    smart_step=[],
                    prm_update=[],
                    gen_update=[],
                    llm_tokens=[],
                )
            )

    completed_beams: list[Beam] = []
    total_tokens = 0
    smart_done = False
    
    for iterate_idx in tqdm(range(config.num_iterations), desc="Beam search iterations"):
        if iterate_idx == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        # Duplicate active beams to ensure that we have config.n beams per iteration
        if len(active_beams) != config.n:
            repeats = (config.n // len(active_beams)) + 1
            logger.debug(
                f"Extending active_beams with {repeats} repetitions to reach size {config.n}"
            )
            extended_active_beams = [
                copy.deepcopy(b) for b in (active_beams * repeats)[: config.n]
            ]
            active_beams = extended_active_beams
            if len(active_beams) != config.n:
                raise ValueError(
                    f"Expected {config.n} active beams, but got {len(active_beams)}"
                )

        if iterate_idx == config.num_iterations - 1:
            # Last iteration, generate to EOS
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                n=1,
            )

        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in active_beams
        ]
        continue_final_message = iterate_idx > 0
        add_generation_prompt = iterate_idx == 0

        tokenizer = slm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        lookahead = 0 if iterate_idx == config.num_iterations - 1 else config.lookahead
        gen_results = generate_k_steps(
            templated_convs, lookahead, slm, sampling_params, 1
        )
        
        prev_active_beams = copy.deepcopy(active_beams)

        # copy the active beams to regenerate the beams with llm
        prompts, completions = [], []
        for beam, gen_result in zip(active_beams, gen_results, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.completion_tokens += gen_result.completion_tokens
            
            beam.current_text += beam.next_texts[0]
            beam.history.append(beam.next_texts[0])
            total_tokens += sum(gen_result.completion_tokens)

            history_text = " ".join(beam.history)
            if len(tokenizer.encode(history_text)) > 2048:
                beam.completed = True
                beam.stop_reasons = ["length"]
                completed_beams.append(beam)
            elif (
                beam.stop_reasons[0] == "EOS"
                or beam.stop_reasons[0] == "length"
                or beam.next_texts[0] == ""
            ):
                beam.completed = True
                completed_beams.append(beam)
            prompts.append(beam.prompt)
            completions.append([beam.current_text])

        scores = prm.score(prompts, completions)

        agg_scores = [
            [aggregate_scores(s, config.agg_strategy) for s in score]
            for score in scores
        ]

        for beam, score in zip(active_beams, scores, strict=True):
            beam.all_scores = score[0]

        # Now filter active_beams and agg_scores for beams that are completed
        agg_scores = [
            agg_scores[i] for i, b in enumerate(active_beams) if not b.completed
        ]
        
        prev_active_beams = [b for idx, b in enumerate(prev_active_beams) if not active_beams[idx].completed]
        active_beams = [b for b in active_beams if not b.completed]

        # Early stopping if all beams are completed
        if len(active_beams) == 0:
            break
        if not config.sort_completed and len(completed_beams) >= config.n:
            break

        # Filter duplicate active beams
        if config.filter_duplicates:
            # Create a dictionary to filter duplicates and retain order
            unique_beam_dict = {}
            for i, b in enumerate(active_beams):
                if b.current_text not in unique_beam_dict:
                    unique_beam_dict[b.current_text] = (
                        i  # Map the unique text to its index
                    )
            active_beams = [active_beams[i] for i in unique_beam_dict.values()]
            prev_active_beams = [prev_active_beams[i] for i in unique_beam_dict.values()]
            agg_scores = [agg_scores[i] for i in unique_beam_dict.values()]

        # Get indices for top (config.n / config.beam_width) completions
        top_indices = np.argsort(np.array(agg_scores).flatten())[
            -(config.n // config.beam_width) :
        ]

        for idx, beam in enumerate(active_beams):
            if idx not in top_indices:
                beam.pruned = True
                
        # SMART beam search implementation       
        # # filter the pruned beams with low scores
        # active_beams = [b for b in active_beams if not b.pruned]
        # agg_scores = [agg_scores[idx] for idx in top_indices]
        
        re_indices = [top_idx for top_idx in top_indices if agg_scores[top_idx][0] < config.threshold]
        if len(re_indices) == 0:
            continue
        
        smart_done = True
        re_beams = [prev_active_beams[idx] for idx in re_indices]          
        
        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in re_beams
        ]
        continue_final_message = iterate_idx > 0
        add_generation_prompt = iterate_idx == 0
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        lookahead = 0 if iterate_idx == config.num_iterations - 1 else config.lookahead
        gen_results = generate_k_steps_for_llm(
            tokenizer, templated_convs, lookahead, llm, config, 1
        )

        reprompts, recompletions = [], []
        for beam, gen_result in zip(re_beams, gen_results, strict=True):
            # update the beam
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.current_text += beam.next_texts[0]
            beam.history.append(beam.next_texts[0])

            if (
                beam.stop_reasons[0] == "EOS"
                or beam.stop_reasons[0] == "length"
                or beam.next_texts[0] == ""
            ):
                beam.completed = True
                completed_beams.append(beam)
            reprompts.append(beam.prompt)
            recompletions.append([beam.current_text])

        re_scores = prm.score(reprompts, recompletions)
        reagg_scores = [
            [aggregate_scores(s, config.agg_strategy) for s in score]
            for score in re_scores
        ]
        
        for beam, score in zip(re_beams, re_scores, strict=True):
            beam.all_scores = score[0]

        for i, (re_idx, beam) in enumerate(zip(re_indices, re_beams)):
            # log correction information
            beam.smart_step.append(iterate_idx)
            beam.gen_update.append((active_beams[re_idx].next_texts[0], beam.next_texts[0]))
            beam.prm_update.append((agg_scores[re_idx][0], reagg_scores[i][0]))
            beam.llm_tokens.append(len(tokenizer.encode(beam.next_texts[0])))
            total_tokens += len(tokenizer.encode(beam.next_texts[0]))
            active_beams[re_idx] = beam
        
    # Filter completed beams for those with top config.n scores
    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n]
    else:
        completed_beams = completed_beams[: config.n]

    if len(completed_beams) != config.n:
        # If we don't have enough completed_beams, duplicate until we reach config.n
        repeats = (config.n // len(completed_beams)) + 1
        logger.debug(
            f"Extending completed_beams with {repeats} repetitions to reach size {config.n}"
        )
        extended_completed_beams = [
            copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]
        ]
        completed_beams = extended_completed_beams

    # Print the problem information
    # for problem, info in problem_info.items():
    #     print(f"{{question: {problem}, generate_llm: {info['generate_llm']}, score_changed: {info['score_changed']}, text_changed: {info['text_changed']}}}")

            
    for beam in completed_beams:
        if len(beam.smart_step) == 0:
            beam.smart_step = [-1]
            beam.prm_update = [(-1.0, -1.0)]
            beam.gen_update = [('-1', '-1')]
            beam.llm_tokens = [-1]
    
    return completed_beams, total_tokens


def smart_beam_search(examples, config: Config, slm: LLM, prm: PRM, llm: None):
    problems = examples["problem"]
    beam_results, total_tokens = _beam_search(problems, config, slm, prm, llm)

    # Group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "pred": []}
    tokenizer = slm.get_tokenizer()

    for p in problems:
        beams = grouped_results[p]
        completions = [b.current_text for b in beams]
        pred = completions[np.argmax([
            aggregate_scores(b.all_scores, config.agg_strategy) for b in beams
        ])]
        results["completions"].append(completions)
        results["pred"].append(pred)
    return results
