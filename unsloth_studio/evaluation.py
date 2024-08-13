# Unsloth Studio
# Copyright (C) 2024-present the Unsloth AI team. All rights reserved.

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


@functools.lru_cache(2)
def get_mmlu_dataset(
    n_samples = 200,
    random_state = 3407,
    add_space = False, # Adds space between question & answer
):
    mmlu_dataset = load_dataset("unsloth/studio_mmlu", split = "train")

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
            for shot, q in zip(all_shots[0], question):
                prompts.append(shot + space + q)
            input_ids = tokenizer(prompts).input_ids

            # Original MMLU https://github.com/hendrycks/test
            # uses 2048 as maximum sequence length
            # then reduces 5 shot to 4 shot etc
            for j, input_ids_ in enumerate(input_ids):
                if len(input_ids_) <= 2048: continue
                shot = 0
                while len(input_ids_) > 2048:
                    shot += 1
                    prompt = all_shots[shot][j] + q
                    input_ids_ = tokenizer(prompt).input_ids
                pass
                input_ids[j] = input_ids_
            pass
            return { "input_ids" : input_ids, }
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
        mmlu_dataset, sorted_indices = get_mmlu_dataset(n_samples = n_samples, random_state = random_state, add_space = False)
    else:
        A = tokenizer("A", add_special_tokens = False).input_ids
        B = tokenizer("B", add_special_tokens = False).input_ids
        C = tokenizer("C", add_special_tokens = False).input_ids
        D = tokenizer("D", add_special_tokens = False).input_ids
        # We need to add a space since _A does not exist!
        mmlu_dataset, sorted_indices = get_mmlu_dataset(n_samples = n_samples, random_state = random_state, add_space = True)
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
    )
    n_examples = len(mmlu_dataset)

    n_samples = 0
    i = 0
    total_correct = torch.zeros(1, dtype = int, device = "cuda")

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
            ).to("cuda", non_blocking = True)

            # Pad to longest item
            lengths = [len(x) for x in input_ids]
            max_length = max(lengths)
            paddings = np.fromiter((max_length-len(x) for x in input_ids), dtype = int)
            indexing = max_length - paddings - 1
            input_ids = torch.from_numpy(
                np.vstack([np.hstack((input_ids_, np.zeros(padding, dtype = int))) \
                for input_ids_, padding in zip(input_ids, paddings)])
            ).to("cuda", non_blocking = True)

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
