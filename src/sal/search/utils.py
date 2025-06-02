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
from dataclasses import dataclass, field
from typing import Union
import torch.nn.functional as F

import numpy as np
from vllm import LLM, SamplingParams
from transformers.generation.stopping_criteria import StopStringCriteria
from transformers import GenerationConfig
from sal.config import Config

logger = logging.getLogger()


def build_conv(
    prompt: str, response: str | None, system_prompt: str
) -> list[dict[str, str]]:
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if response != "":
        conversation.append({"role": "assistant", "content": response})

    return conversation


def last(x):
    if len(x) == 0:
        logger.warning("empty list")
        return 0
    return x[-1]


def list_mean(x):
    if len(x) == 0:
        logger.warning("empty list")
        return 0
    return np.mean(x)


@dataclass
class Beam:
    prompt: str
    index: int
    current_text: str | None
    next_texts: list[str] | None
    lookahead_texts: list[str] | None
    stop_reasons: list[str | None] | None
    best_scores: list[float]  # the PRM scores
    all_scores: list[list[float]]  # all PRM scores
    previous_text: str | None
    pruned: False
    history: list[str] = field(default_factory=list) 
    completed: bool = False
    completion_tokens: list[int] = field(default_factory=list)
    smart_step: list[int] = field(default_factory=list)
    agg_score_update: list[tuple[float, float]] = field(default_factory=list)
    prm_update: list[tuple[float, float]] = field(default_factory=list)
    likelihood_update: list[tuple[float, float]] = field(default_factory=list)
    likelihood_mean_update: list[tuple[float, float]] = field(default_factory=list)
    tokenprobs_mean_update: list[tuple[float, float]] = field(default_factory=list)
    gen_update: list[tuple[list[str], list[str]]] = field(default_factory=list)
    llm_tokens: list[int] = field(default_factory=list)

@dataclass
class GenResult:
    index: int
    initial_prompt: str
    first_step_text: str
    first_step_stop_reason: str
    lookahead_text: str
    completion_tokens: int
    stop_reason: str | None


def generate_k_steps(
    templated_convs,
    lookahead_steps: int,
    llm: LLM,
    sampling_params: SamplingParams,
    beam_width: int,
) -> list[Beam]:
    gen_results = []
    for i, text in enumerate(templated_convs):
        for j in range(beam_width):
            gen_result = GenResult(
                index=i,
                initial_prompt=text,
                first_step_text="",
                lookahead_text="",
                completion_tokens=0,
                stop_reason=None,
                first_step_stop_reason=None,
            )
            gen_results.append(gen_result)

    gen_sampling_params = copy.deepcopy(sampling_params)

    for i in range(lookahead_steps + 1):
        if i == 1:
            gen_sampling_params.temperature = 0.0  # greedy for the rest of the steps
        # get all generations that did not finish with eos
        current_gen = [
            gen_results[i]
            for i in range(len(gen_results))
            if gen_results[i].stop_reason != "EOS"
        ]
        gen_prompts = [
            gen_result.initial_prompt + gen_result.lookahead_text
            for gen_result in current_gen
        ]
        llm_outputs = llm.generate(gen_prompts, gen_sampling_params, use_tqdm=False)
        for gen_result, output in zip(current_gen, llm_outputs):
            gen_text = output.outputs[0].text
            if i == 0:
                gen_result.first_step_text = gen_text
                gen_result.first_step_stop_reason = output.outputs[0].stop_reason
                if gen_result.first_step_stop_reason is None:
                    gen_result.first_step_stop_reason = "EOS"

            gen_result.lookahead_text = gen_result.lookahead_text + gen_text
            gen_result.completion_tokens = len(output.outputs[0].token_ids)
            gen_result.stop_reason = output.outputs[0].stop_reason
            if gen_result.stop_reason is None:
                gen_result.stop_reason = "EOS"

    outputs: list[Beam] = []

    counter = 0
    for i, text in enumerate(templated_convs):
        next_texts = []
        stop_reasons = []
        lookahead_texts = []
        num_completion_tokens = [] 
        for j in range(beam_width):
            gen_result = gen_results[counter]
            next_texts.append(gen_result.first_step_text)
            lookahead_texts.append(gen_result.lookahead_text)
            stop_reasons.append(gen_result.first_step_stop_reason)
            num_completion_tokens.append(gen_result.completion_tokens)
            counter += 1

        beam_result = Beam(
            prompt=text,
            index=i,
            current_text="",
            next_texts=next_texts,
            lookahead_texts=lookahead_texts,
            completion_tokens=num_completion_tokens,
            stop_reasons=stop_reasons,
            best_scores=[0.0],
            all_scores=[],
            previous_text=None,
            pruned=False,
            history=[],
        )
        outputs.append(beam_result)

    return outputs


def generate_k_steps_for_llm(
    tokenizer,
    templated_convs,
    lookahead_steps: int,
    llm: LLM,
    config: Config,
    beam_width: int,
) -> list[Beam]:
    gen_results = []
    for i, text in enumerate(templated_convs):
        for j in range(beam_width):
            gen_result = GenResult(
                index=i,
                initial_prompt=text,
                first_step_text="",
                lookahead_text="",
                completion_tokens=0,
                stop_reason=None,
                first_step_stop_reason=None,
            )
            gen_results.append(gen_result)
            
    stopping_criteria=StopStringCriteria(stop_strings="\n\n", tokenizer=tokenizer)
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=config.temperature,
        top_p=config.top_p,
        max_new_tokens=config.max_tokens,
    )

    for i in range(lookahead_steps + 1):
        # get all generations that did not finish with eos
        current_gen = [
            gen_results[i]
            for i in range(len(gen_results))
            if gen_results[i].stop_reason != "EOS"
        ]
        gen_prompts = [
            gen_result.initial_prompt + gen_result.lookahead_text
            for gen_result in current_gen
        ]

        decoded_outputs = []
        stop_reasons = []
        num_completion_tokens = [] 
        
        for gen_prompt in gen_prompts:
            input_ids = tokenizer(gen_prompt, return_tensors="pt").to(llm.device)
            # Generate just the next step using large LLM
            new_step = ""
            new_ids = llm.generate(
                **input_ids,
                stopping_criteria=[stopping_criteria],
                generation_config=generation_config,
            )[:, input_ids["input_ids"].shape[1]:]
            new_step = tokenizer.decode(new_ids[0]) #, skip_special_tokens=True)
            
            assert len(new_step) > 0 and new_step != "\n\n" and new_step != ""
            # stop reason logic
            stop_reason = None
            if new_step.endswith("\n\n"):
                stop_reason = '\n\n'
            elif len(new_step) > config.max_tokens:
                stop_reason = "length"
            else:
                stop_reason = "EOS"
            # elif tokenizer.eos_token_id == new_ids[0][-1] or new_step.endswith(tokenizer.eos_token):
            #     stop_reason = "EOS"


            decoded_outputs.append(new_step)
            stop_reasons.append(stop_reason)
            num_completion_tokens.append(len(new_ids[0]))
        
        for gen_result, output, completion_tokens, stop_reason in zip(current_gen, decoded_outputs, num_completion_tokens, stop_reasons):
            if i == 0:
                gen_result.first_step_text = output
                gen_result.first_step_stop_reason = stop_reason
                if gen_result.first_step_stop_reason is None:
                    gen_result.first_step_stop_reason = "EOS"

            gen_result.lookahead_text = gen_result.lookahead_text + output
            gen_result.completion_tokens = completion_tokens
            gen_result.stop_reason = stop_reason
            if gen_result.stop_reason is None:
                gen_result.stop_reason = "EOS"

    outputs: list[Beam] = []

    counter = 0
    for i, text in enumerate(templated_convs):
        next_texts = []
        stop_reasons = []
        lookahead_texts = []
        num_completion_tokens = [] 
        for j in range(beam_width):
            gen_result = gen_results[counter]
            next_texts.append(gen_result.first_step_text)
            lookahead_texts.append(gen_result.lookahead_text)
            stop_reasons.append(gen_result.first_step_stop_reason)
            num_completion_tokens.append(gen_result.completion_tokens)
            counter += 1

        beam_result = Beam(
            prompt=text,
            index=i,
            current_text="",
            next_texts=next_texts,
            lookahead_texts=lookahead_texts,
            completion_tokens= num_completion_tokens,
            stop_reasons=stop_reasons,
            best_scores=[0.0],
            all_scores=[],
            previous_text=None,
            pruned=False,
            history=[],
        )
        outputs.append(beam_result)

    return outputs


def generate_k_steps_with_responses(
    templated_convs,
    lookahead_steps: int,
    llm: LLM,
    sampling_params: SamplingParams,
    beam_width: int,
) -> list[Beam]:
    gen_results = []
    for i, text in enumerate(templated_convs):
        for j in range(beam_width):
            gen_result = GenResult(
                index=i,
                initial_prompt=text,
                first_step_text="",
                lookahead_text="",
                completion_tokens=[],
                stop_reason=None,
                first_step_stop_reason=None,
            )
            gen_results.append(gen_result)

    gen_sampling_params = copy.deepcopy(sampling_params)

    for i in range(lookahead_steps + 1):
        if i == 1:
            gen_sampling_params.temperature = 0.0  # greedy for the rest of the steps
        # get all generations that did not finish with eos
        current_gen = [
            gen_results[i]
            for i in range(len(gen_results))
            if gen_results[i].stop_reason != "EOS"
        ]
        gen_prompts = [
            gen_result.initial_prompt + gen_result.lookahead_text
            for gen_result in current_gen
        ]
        llm_outputs = llm.generate(gen_prompts, gen_sampling_params, use_tqdm=False)
        for gen_result, output in zip(current_gen, llm_outputs):
            gen_text = output.outputs[0].text
            if i == 0:
                gen_result.first_step_text = gen_text
                gen_result.first_step_stop_reason = output.outputs[0].stop_reason
                if gen_result.first_step_stop_reason is None:
                    gen_result.first_step_stop_reason = "EOS"

            gen_result.lookahead_text = gen_result.lookahead_text + gen_text
            gen_result.completion_tokens = len(output.outputs[0].token_ids)
            gen_result.stop_reason = output.outputs[0].stop_reason
            if gen_result.stop_reason is None:
                gen_result.stop_reason = "EOS"

    outputs: list[Beam] = []

    counter = 0
    for i, text in enumerate(templated_convs):
        next_texts = []
        stop_reasons = []
        lookahead_texts = []
        num_completion_tokens = [] 
        for j in range(beam_width):
            gen_result = gen_results[counter]
            next_texts.append(gen_result.first_step_text)
            lookahead_texts.append(gen_result.lookahead_text)
            stop_reasons.append(gen_result.first_step_stop_reason)
            num_completion_tokens.append(gen_result.completion_tokens)
            counter += 1

        beam_result = Beam(
            prompt=text,
            index=i,
            current_text="",
            next_texts=next_texts,
            lookahead_texts=lookahead_texts,
            completion_tokens= num_completion_tokens,
            stop_reasons=stop_reasons,
            best_scores=[0.0],
            all_scores=[],
            previous_text=None,
            pruned=False,
            history=[],
        )
        outputs.append(beam_result)

    return outputs, llm_outputs


def generate_k_steps_for_llm_with_responses(
    tokenizer,
    templated_convs,
    lookahead_steps: int,
    llm: LLM,
    config: Config,
    beam_width: int,
) -> list[Beam]:
    gen_results = []
    for i, text in enumerate(templated_convs):
        for j in range(beam_width):
            gen_result = GenResult(
                index=i,
                initial_prompt=text,
                first_step_text="",
                lookahead_text="",
                completion_tokens=0,
                stop_reason=None,
                first_step_stop_reason=None,
            )
            gen_results.append(gen_result)
            
    stopping_criteria=StopStringCriteria(stop_strings="\n\n", tokenizer=tokenizer)
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=config.temperature,
        top_p=config.top_p,
        max_new_tokens=config.max_tokens,
    )

    for i in range(lookahead_steps + 1):
        # get all generations that did not finish with eos
        current_gen = [
            gen_results[i]
            for i in range(len(gen_results))
            if gen_results[i].stop_reason != "EOS"
        ]
        gen_prompts = [
            gen_result.initial_prompt + gen_result.lookahead_text
            for gen_result in current_gen
        ]

        decoded_outputs = []
        stop_reasons = []
        num_completion_tokens = [] 
        responses_token_log_probs = []
        
        for gen_prompt in gen_prompts:
            input_ids = tokenizer(gen_prompt, return_tensors="pt").to(llm.device)
            # Generate just the next step using large LLM
            new_step = ""
            response = llm.generate(
                **input_ids,
                stopping_criteria=[stopping_criteria],
                generation_config=generation_config,
                output_scores=True,
                return_dict_in_generate=True
            )

            new_ids = response.sequences[:, input_ids["input_ids"].shape[-1]:]  #
            new_step = tokenizer.decode(new_ids[0]) #, skip_special_tokens=True)

            scores = response.scores
            log_probs = [F.log_softmax(score, dim=-1) for score in scores] 
            token_log_probs = [
                log_prob[0, tok].item()  # 배치 차원 제거 후 특정 토큰의 log probability 가져오기
                for log_prob, tok in zip(log_probs, new_ids[0])  # zip으로 길이 맞춤
            ]
            responses_token_log_probs.append(token_log_probs)

            
            assert len(new_step) > 0 and new_step != "\n\n" and new_step != ""
            # stop reason logic
            stop_reason = None
            if new_step.endswith("\n\n"):
                stop_reason = '\n\n'
            elif len(new_step) > config.max_tokens:
                stop_reason = "length"
            else:
                stop_reason = "EOS"
            # elif tokenizer.eos_token_id == new_ids[0][-1] or new_step.endswith(tokenizer.eos_token):
            #     stop_reason = "EOS"


            decoded_outputs.append(new_step)
            stop_reasons.append(stop_reason)
            num_completion_tokens.append(len(new_ids[0]))
        
        for gen_result, output, completion_tokens, stop_reason in zip(current_gen, decoded_outputs, num_completion_tokens, stop_reasons):
            if i == 0:
                gen_result.first_step_text = output
                gen_result.first_step_stop_reason = stop_reason
                if gen_result.first_step_stop_reason is None:
                    gen_result.first_step_stop_reason = "EOS"

            gen_result.lookahead_text = gen_result.lookahead_text + output
            gen_result.completion_tokens = completion_tokens
            gen_result.stop_reason = stop_reason
            if gen_result.stop_reason is None:
                gen_result.stop_reason = "EOS"

    outputs: list[Beam] = []

    counter = 0
    for i, text in enumerate(templated_convs):
        next_texts = []
        stop_reasons = []
        lookahead_texts = []
        num_completion_tokens = [] 
        for j in range(beam_width):
            gen_result = gen_results[counter]
            next_texts.append(gen_result.first_step_text)
            lookahead_texts.append(gen_result.lookahead_text)
            stop_reasons.append(gen_result.first_step_stop_reason)
            num_completion_tokens.append(gen_result.completion_tokens)
            counter += 1

        beam_result = Beam(
            prompt=text,
            index=i,
            current_text="",
            next_texts=next_texts,
            lookahead_texts=lookahead_texts,
            completion_tokens= num_completion_tokens,
            stop_reasons=stop_reasons,
            best_scores=[0.0],
            all_scores=[],
            previous_text=None,
            pruned=False,
            history=[],
        )
        outputs.append(beam_result)

    return outputs, responses_token_log_probs