# Unsloth Studio
# Copyright (C) 2025-present the Unsloth AI team. All rights reserved.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "calculate_mmlu",
]

import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
import functools
from tqdm import tqdm
import re
from openai import OpenAI, AsyncOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio
from concurrent.futures import ThreadPoolExecutor

@functools.lru_cache(2)
def tokenizer_adds_bos_token(tokenizer):
    if getattr(tokenizer, "bos_token", None) is not None:
        if tokenizer("1")["input_ids"][0] == tokenizer.bos_token_id:
            return True
    return False
pass


answer_finder = re.compile(r"(Answer\: (?:A|B|C|D))[\n]{0,}")
def to_chat(prompt, q, tokenizer):
    splits = answer_finder.split(prompt)
    if splits[-1] == "": splits.pop(-1)
    system_message, splits[0] = splits[0].split("\n\n", 1)
    messages = [{"role" : "system", "content" : system_message}] + [
        {
            "role" : "user" if i % 2 == 0 else "assistant",
            "content" : x
        }
        for i, x in enumerate(splits)
    ]
    if q.endswith("Answer:"): q = q[:-len("Answer:")]
    messages += [{"role" : "user", "content" : q}]
    message = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True,
    )
    message += "Answer:"

    # Remove first BOS token if tokenizer auto adds BOS tokens
    if tokenizer_adds_bos_token(tokenizer) and \
        message.startswith(tokenizer.bos_token):

        message = message[len(tokenizer.bos_token):]
    pass
    return message
pass


@functools.lru_cache(2)
def get_mmlu_dataset(
    tokenizer,
    n_samples = 200,
    random_state = 3407,
    add_space = False, # Adds space between question & answer
    apply_chat_template = False, # Use user:, assistant: format
):
    mmlu_dataset = load_dataset("unsloth/studio_mmlu", split = "train")
    if n_samples == -1 or n_samples is None:
        n_samples = len(mmlu_dataset)

    # Only used if tokenizer does not use _A, _B, _C, _D, and just uses
    # A, B, C, D. Llama 3.1 has _A.
    space = " " if add_space else ""

    def mmlu_tokenize(tokenizer):
        def _mmlu_tokenize(examples):
            all_shots = [
                examples["5_shot"],
                examples["4_shot"],
                examples["3_shot"],
                examples["2_shot"],
                examples["1_shot"],
            ]
            question = examples["Q"]
            prompts = []

            if apply_chat_template:
                for xx in all_shots:
                    for i, (x, q) in enumerate(zip(xx, question)):
                        xx[i] = to_chat(x, q, tokenizer)
            pass

            for shot, q in zip(all_shots[0], question):
                if apply_chat_template:
                    prompts.append(shot)
                else:
                    prompts.append(shot + q + space)
            pass
            input_ids = tokenizer(
                prompts,
                add_special_tokens = not apply_chat_template,
            ).input_ids

            # Original MMLU https://github.com/hendrycks/test
            # uses 2048 as maximum sequence length
            # then reduces 5 shot to 4 shot etc
            if not apply_chat_template:
                for j, input_ids_ in enumerate(input_ids):
                    if len(input_ids_) <= 2048: continue
                    shot = 0
                    while len(input_ids_) > 2048:
                        shot += 1
                        prompt = all_shots[shot][j] + q + space
                        input_ids_ = tokenizer(prompt).input_ids
                        prompts[j] = prompt
                    pass
                    input_ids[j] = input_ids_
                pass
            return { "input_ids" : input_ids, "prompts" : prompts, }
        pass
        return _mmlu_tokenize
    pass
    mmlu_dataset = mmlu_dataset.map(mmlu_tokenize(tokenizer), batched = True, num_proc = 4,)

    # We then sort by lengths of input_ids for faster processing
    mmlu_dataset = mmlu_dataset.map(
        lambda xs: {"lengths" : list(map(len, xs["input_ids"]))},
        batched = True, num_proc = 4,
    )
    mmlu_dataset = mmlu_dataset.to_pandas()

    subjects = mmlu_dataset["Subject"]
    counts = subjects.value_counts()

    # Sklearn type stratified sampling weights
    weights = len(subjects) / (len(counts) * counts)
    mmlu_dataset = mmlu_dataset.sample(n_samples, random_state = random_state, weights = subjects.map(weights))
    mmlu_dataset.reset_index(drop = True, inplace = True)

    sorted_indices = mmlu_dataset["input_ids"].str.len().sort_values().index.values

    return mmlu_dataset, sorted_indices
pass


@functools.lru_cache(2)
def prepare_calculate_mmlu(
    tokenizer,
    n_samples = 200,
    random_state = 3407,
    target_length = 4096,
    apply_chat_template = False,
):
    # Llama 3.1 tokenizer uses _A and not A
    # We shall check this for the specific tokenizer!
    A = tokenizer(" A", add_special_tokens = False).input_ids
    B = tokenizer(" B", add_special_tokens = False).input_ids
    C = tokenizer(" C", add_special_tokens = False).input_ids
    D = tokenizer(" D", add_special_tokens = False).input_ids
    if len(A) == 1 and len(B) == 1 and len(C) == 1 and len(D) == 1:
        answer_ids = torch.tensor([A[0], B[0], C[0], D[0]])
        answer_ids = answer_ids.to("cuda", non_blocking = True)
        mmlu_dataset, sorted_indices = get_mmlu_dataset(
            tokenizer = tokenizer,
            n_samples = n_samples,
            random_state = random_state,
            add_space = False,
            apply_chat_template = apply_chat_template,
        )
    else:
        A = tokenizer("A", add_special_tokens = False).input_ids
        B = tokenizer("B", add_special_tokens = False).input_ids
        C = tokenizer("C", add_special_tokens = False).input_ids
        D = tokenizer("D", add_special_tokens = False).input_ids
        answer_ids = torch.tensor([A[0], B[0], C[0], D[0]])
        answer_ids = answer_ids.to("cuda", non_blocking = True)
        # We need to add a space since _A does not exist!
        mmlu_dataset, sorted_indices = get_mmlu_dataset(
            tokenizer = tokenizer,
            n_samples = n_samples,
            random_state = random_state,
            add_space = True,
            apply_chat_template = apply_chat_template,
        )
    pass
    all_lengths = np.fromiter(mmlu_dataset["lengths"], dtype = int)

    return mmlu_dataset, sorted_indices, answer_ids, all_lengths
pass


@torch.inference_mode
def calculate_mmlu(
    model,
    tokenizer,
    n_samples = 200,
    random_state = 3407,
    target_length = 4096,
    apply_chat_template = False,
):
    """
    Calculates Macro MMLU Average with Stratified Random Sampling like in
    Tiny MMLU. Original MMLU from https://github.com/hendrycks/test.
    """
    answers_mapping = {"A" : 0, "B" : 1, "C" : 2, "D" : 3,}
    mmlu_dataset, sorted_indices, answer_ids, all_lengths = prepare_calculate_mmlu(
        tokenizer = tokenizer,
        n_samples = n_samples,
        random_state = random_state,
        target_length = target_length,
        apply_chat_template = apply_chat_template,
    )
    n_examples = len(mmlu_dataset)

    n_samples = 0
    i = 0
    device_start = model.get_input_embeddings ().weight.device
    device_end   = model.get_output_embeddings().weight.device
    total_correct = torch.zeros(1, dtype = int, device = device_end)
    answer_ids = answer_ids.to(device_end)

    with torch.amp.autocast(device_type = "cuda", dtype = model.config.torch_dtype), tqdm(total = n_examples) as progress_bar:
        while i < n_examples:
            # Find samples until (target_length = 4096) is reached
            length = 0
            left = i
            while length < target_length:
                index = sorted_indices[i]
                length += all_lengths[index]
                i += 1
                if i == n_examples: break
            right = i
            right = min(right, n_examples)
            i = right
            n_samples += (right-left)
            progress_bar.update(right-left)

            # Select batch
            indices = sorted_indices[left:right]
            current_batch = mmlu_dataset.iloc[indices]
            input_ids = current_batch["input_ids"]
            answers   = current_batch["A"]
            answers   = torch.from_numpy(
                np.fromiter((answers_mapping[x] for x in answers), dtype = int),
            ).to(device_end, non_blocking = True)

            # Pad to longest item
            lengths = [len(x) for x in input_ids]
            max_length = max(lengths)
            paddings = np.fromiter((max_length-len(x) for x in input_ids), dtype = int)
            indexing = max_length - paddings - 1
            input_ids = torch.from_numpy(
                np.vstack([np.hstack((input_ids_, np.zeros(padding, dtype = int))) \
                for input_ids_, padding in zip(input_ids, paddings)])
            ).to(device_start, non_blocking = True)

            logits = model(
                input_ids = input_ids,
                output_attentions = False,
                output_hidden_states = False,
                use_cache = False,
            ).logits

            # As per original MMLU, select A, B, C, D locations
            logits = logits[range(input_ids.shape[0]), indexing, :][:, answer_ids]

            # Original MMLU argmax - careful of ties!
            is_correct = logits.argmax(-1) == answers
            total_correct += is_correct.sum()

            progress_bar.set_description("MMLU Accuracy = {:.1f}%".format(total_correct.item() / n_samples * 100))
        pass
    pass
    return total_correct.item() / n_samples
pass

async def async_get_completion(openai_client, model_name, prompt):
    response = await openai_client.completions.create(
        model = model_name,
        prompt = prompt,
        logprobs = 5,
        temperature = 0.0,
        max_tokens = 1,
        n = 1,
    )
    return response.choices[0]
pass

def get_completion(openai_client, model_name, prompt):
    response = openai_client.completions.create(
        model = model_name,
        prompt = prompt,
        logprobs = 5,
        temperature = 0.0,
        max_tokens = 1,
        n = 1,
    )
    return response.choices[0]
pass

async def async_process_batch(openai_client, model_name, batch):
    return await asyncio.gather(*[
        async_get_completion(openai_client, model_name, x)
        for x in batch
    ])
pass


@torch.inference_mode
def calculate_mmlu_openai_server(
    model_name = "unsloth/Llama-3.1-8B-Instruct",
    base_url = "http://127.0.0.1:8080/v1",
    api_key = "sk-no-key-required",
    use_async = True,
    n_samples = None,
    random_state = 3407,
    target_length = 8192,
    apply_chat_template = False,
):
    """
    Calculates Macro MMLU Average with Stratified Random Sampling like in
    Tiny MMLU. Original MMLU from https://github.com/hendrycks/test.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    openai_client = (AsyncOpenAI if use_async else OpenAI)(
        base_url = base_url,
        api_key = api_key,
    )
    answers_mapping = {"A" : 0, "B" : 1, "C" : 2, "D" : 3,}
    mmlu_dataset, sorted_indices, answer_ids, all_lengths = prepare_calculate_mmlu(
        tokenizer = tokenizer,
        n_samples = n_samples,
        random_state = random_state,
        target_length = target_length,
        apply_chat_template = apply_chat_template,
    )
    answer_ids = answer_ids.tolist()
    set_answer_ids = frozenset(answer_ids)
    print("Answer IDs =", answer_ids)
    n_examples = len(mmlu_dataset)

    n_samples = 0
    i = 0
    total_correct = 0

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    with ThreadPoolExecutor(1) as pool, \
        tqdm(total = n_examples) as progress_bar:

        while i < n_examples:
            if use_async:
                # Find samples until (target_length = 4096) is reached
                length = 0
                left = i
                while length < target_length:
                    index = sorted_indices[i]
                    length += all_lengths[index]
                    i += 1
                    if i == n_examples: break
                right = i
                right = min(right, n_examples)
                i = right
                n_samples += (right-left)
                progress_bar.update(right-left)

                current_batch = mmlu_dataset.iloc[left:right]
                prompt  = current_batch["prompts"].values
                answers = current_batch["A"]
                t = async_process_batch(openai_client, model_name, prompt)
                if loop is not None:
                    completions = pool.submit(lambda: asyncio.run(t))
                    completions = completions.result()
                else:
                    completions = asyncio.run(t)
            else:
                progress_bar.update(1)
                n_samples += 1
                current_batch = mmlu_dataset.iloc[i]
                prompt  = current_batch["prompts"]
                answers = [current_batch["A"]]
                t = get_completion(openai_client, model_name, prompt)
                completions = [t]
                i += 1
            pass

            for completion, answer in zip(completions, answers):
                top_logprobs = completion.logprobs\
                    .content[0]["top_logprobs"]

                # As per original MMLU, select A, B, C, D locations
                selected_answers = [x["token"] for x in top_logprobs if x["id"] in set_answer_ids]

                # Original MMLU argmax - careful of ties!
                # Must do strip since _A == A
                is_correct = (len(answer) != 0 and selected_answers[0].strip() == answer)
                total_correct += is_correct
            pass
            progress_bar.set_description("MMLU Accuracy = {:.1f}%".format(total_correct / n_samples * 100))
        pass
    pass
    return total_correct / n_samples
pass


def test_mmlu(
    model_name = "unsloth/Llama-3.1-8B-Instruct",
    base_url = "http://127.0.0.1:8080/v1",
    api_key = "sk-no-key-required",
    n_samples = None,
    random_state = 3407,
    apply_chat_template = False,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = torch.bfloat16,
        device_map = {"" : 0},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    calculate_mmlu(
        model,
        tokenizer,
        n_samples = n_samples,
        apply_chat_template = apply_chat_template,
        target_length = 1024 * 4,
    )
    calculate_mmlu_openai_server(
        model_name = model_name,
        base_url = base_url,
        n_samples = None,
        apply_chat_template = apply_chat_template,
        use_async = True,
        target_length = 1024 * 32,
    )
pass
