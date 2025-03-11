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
    "UnslothEfficientLoss",
    "unsloth_efficient_ce_loss",
]

import torch
from .utils import *

UNSLOTH_COMPILE_ENABLE = True
class UnslothEfficientLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        _input : torch.Tensor,
        weight : torch.Tensor,
        target : torch.Tensor,
        bias  : Optional[torch.Tensor] = None,
        shift : bool = True,
        reduction : str = "mean",
        logit_scale : Optional[float] = None,
        logit_softcapping : Optional[float] = None,
        loss_function : Callable = torch.nn.CrossEntropyLoss,
        ignore_index : int = -100,
        chunk_size : int = 8192,
        attention_mask : Optional[torch.Tensor] = None,
    ):
        # All Unsloth Studio code licensed under AGPLv3
        device = weight.device
        dtype = _input.dtype
        vocab_size, hd = weight.shape
        _loss_function = loss_function(
            reduction = "sum",
            ignore_index = ignore_index,
        )

        has_grad_weight = weight.requires_grad
        has_grad_input  = _input.requires_grad
        has_grad_bias   = False

        # Mixed precision downcasts from float32 to float16
        weight = weight.to(dtype)
        if bias is not None:
            has_grad_bias = bias.requires_grad
            bias = bias.to(dtype)
        pass

        def process_labels(target, attention_mask = None):
            if shift:
                shift_target = torch.empty_like(target, device = device, dtype = torch.int64)
                shift_target[..., :-1] = target[..., 1:]
                shift_target[..., -1] = ignore_index
                shift_target = shift_target.view(-1)

                # For VLMs like Paligemma, Idefics - used to mask tokens like <image> out
                # if attention_mask is not None:
                #     shift_attention_mask = torch.empty_like(attention_mask, device = device, dtype = torch.bool)
                #     shift_attention_mask[..., :-1] = attention_mask[..., 1:] != 0
                #     shift_attention_mask[..., -1]  = False
                #     shift_attention_mask = shift_attention_mask.view(-1)
                # else:
                #     shift_attention_mask = None
                shift_attention_mask = None
            else:
                shift_target = target.view(-1)
                # if attention_mask is not None:
                #     shift_attention_mask = attention_mask.view(-1)
                # else:
                #     shift_attention_mask = None
                shift_attention_mask = None
            return (
                shift_target,
                (shift_target != ignore_index).sum() if reduction == "mean" else 1.0,
                shift_attention_mask,
            )
        pass
        # if UNSLOTH_COMPILE_ENABLE:
        #     process_labels = torch.compile(
        #         process_labels,
        #         dynamic = None,
        #         options = torch_compile_options,
        #     )
        #     mark_dynamic(target, 1)
        #     # if attention_mask is not None:
        #     #     mark_dynamic(attention_mask, 1)
        # pass
        target, n_labels, attention_mask = process_labels(target, attention_mask)
        if reduction == "sum": n_labels = 1.0

        def inner_compute_loss(input_chunk, weight, bias, target):#, mask = None):
            input_chunk = input_chunk.to(weight.device)
            # if mask is not None:
            #     # Only calculate loss on good attention parts for VLMs
            #     input_chunk = input_chunk[mask]
            #     target = target[mask]
            # pass
            if bias is not None:
                logits = torch.addmm(bias, input_chunk, weight.t())
            else:
                logits = torch.matmul(input_chunk, weight.t())
            
            # Logit Scaling like in Cohere, Granite
            if logit_scale is not None:
                logits = logits * logit_scale

            # Softcapping like in Gemma 2
            if logit_softcapping is not None:
                logits = logits / logit_softcapping
                logits = torch.tanh(logits)
                logits = logits * logit_softcapping
            
            # Upcast and loss_function -> force float32 upcast
            with torch.autocast(device_type = "cuda", enabled = False):
                logits = logits.float()
                loss = _loss_function(logits, target)
            print("loss", loss)
            return loss / n_labels
        pass

        grad_weight = torch.zeros_like(weight, dtype = torch.float32, device = device) if has_grad_weight else None
        grad_bias   = torch.zeros_like(bias, dtype = torch.float32, device = device) if has_grad_bias else None
        total_loss  = torch.zeros((), dtype = torch.float32, device = device)
        grad_input  = torch.zeros_like(_input, device = device) if has_grad_input else None
        _input      = _input.view(-1, hd)

        # if > 50%, then make a new chunk
        n_chunks = int(round(_input.shape[0] / chunk_size))
        if n_chunks == 0: n_chunks = 1

        def accumulate_chunk(input_chunk, target_chunk, grad_input_chunk):#, mask_chunk = None):
            chunk_grad_weight = None
            chunk_grad_bias = None
            print("input_chunk", input_chunk)
            print("target_chunk", target_chunk)
            print("grad_input_chunk", grad_input_chunk)
            print("inner_compute_loss", inner_compute_loss, type(inner_compute_loss))
            if has_grad_weight and has_grad_bias and has_grad_input:
                (chunk_grad_input, chunk_grad_weight, chunk_grad_bias,), chunk_loss = torch.func.grad_and_value(
                    inner_compute_loss, argnums = (0, 1, 2,))(
                    input_chunk, weight, bias, target_chunk, #mask_chunk,
                )
            elif not has_grad_weight and not has_grad_bias and has_grad_input:
                (chunk_grad_input,), chunk_loss = torch.func.grad_and_value(
                    inner_compute_loss, argnums = (0,))(
                    input_chunk, weight, bias, target_chunk, #mask_chunk,
                )
            elif has_grad_weight and not has_grad_bias and has_grad_input:
                (chunk_grad_input, chunk_grad_weight,), chunk_loss = torch.func.grad_and_value(
                    inner_compute_loss, argnums = (0, 1,))(
                    input_chunk, weight, bias, target_chunk, #mask_chunk,
                )
            elif not has_grad_weight and has_grad_bias and has_grad_input:
                (chunk_grad_input, chunk_grad_bias,), chunk_loss = torch.func.grad_and_value(
                    inner_compute_loss, argnums = (0, 2,))(
                    input_chunk, weight, bias, target_chunk, #mask_chunk,
                )
            else:
                raise RuntimeError(
                    f"Unsloth: Loss gradients not supported: {has_grad_weight}, {has_grad_bias}, {has_grad_input}"
                )
                
            if grad_weight      is not None: grad_weight     .add_ (chunk_grad_weight)
            if grad_bias        is not None: grad_bias       .add_ (chunk_grad_bias)
            if grad_input_chunk is not None: grad_input_chunk.copy_(chunk_grad_input)
            total_loss.add_(chunk_loss)
        pass
        if UNSLOTH_COMPILE_ENABLE:
            accumulate_chunk = torch.compile(
                accumulate_chunk,
                dynamic = None,
                options = torch_compile_options,
            )
        pass

        input_chunks  = torch.chunk(_input, n_chunks, dim = 0)
        target_chunks = torch.chunk(target, n_chunks, dim = 0)
        grad_input_chunks = torch.chunk(grad_input.view(-1, hd), n_chunks, dim = 0) \
            if has_grad_input else [None] * n_chunks
        mask_chunks = torch.chunk(attention_mask, n_chunks, dim = 0) \
            if attention_mask is not None else [None] * n_chunks

        for input_chunk, target_chunk, grad_input_chunk, mask_chunk in \
            zip(input_chunks, target_chunks, grad_input_chunks, mask_chunks):
            
            if UNSLOTH_COMPILE_ENABLE: 
                mark_dynamic(input_chunk,      0)
                mark_dynamic(target_chunk,     0)
                mark_dynamic(grad_input_chunk, 0)
                # if mask_chunk is not None:
                #     mark_dynamic(mask_chunk,   0)
            accumulate_chunk(input_chunk, target_chunk, grad_input_chunk)#, mask_chunk)
        pass

        ctx.save_for_backward(
            grad_input,
            grad_weight,
            grad_bias,
        )
        return total_loss
    pass

    @staticmethod
    def backward(ctx, grad_output):
        # All Unsloth Studio code licensed under AGPLv3
        (grad_input, grad_weight, grad_bias) = ctx.saved_tensors
        # Multiply by upstream gradients
        if grad_output != 1.0:
            if grad_input  is not None: grad_input .mul_(grad_output)
            if grad_weight is not None: grad_weight.mul_(grad_output)
            if grad_bias   is not None: grad_bias  .mul_(grad_output)
        pass
        return (
            grad_input, grad_weight, None, grad_bias,
            None, None, None, None, None, None, None, None,
        )
    pass
pass


def unsloth_efficient_ce_loss(
    hidden_states : torch.Tensor,
    lm_head : torch.nn.Linear,
    labels : torch.Tensor,
    shift : bool = True,
    reduction : str = "mean",
    logit_scale : Optional[float] = None,
    logit_softcapping : Optional[float] = None,
    ignore_index : int = -100,
    chunk_size : int = 1024, # Around 512MB per 1024 for 128K vocab
    attention_mask : Optional[torch.Tensor] = None, # For VLMs Paligemma, Idefics
):
    # All Unsloth Studio code licensed under AGPLv3
    assert(type(hidden_states) is torch.Tensor)
    assert(type(lm_head) is torch.nn.Linear)
    assert(type(labels) is torch.Tensor)
    assert(type(shift) is bool)
    assert(reduction == "mean" or reduction == "sum")
    assert(logit_scale is None or type(logit_scale) is float)
    assert(logit_softcapping is None or type(logit_softcapping) is float)
    assert(type(ignore_index) is int)
    assert(type(chunk_size) is int)
    # if attention_mask is not None:
    #     assert(type(attention_mask) is torch.Tensor)
    #     assert(attention_mask.shape == labels.shape)
    assert(attention_mask is None)

    # Dynamic chunk size
    # Smaller ones have less chunks, larger ones more chunks
    vocab_size = lm_head.out_features
    chunk_size = int(chunk_size * ((128 * 1024) / vocab_size))

    print("hidden_states", hidden_states)
    print("lm_head", lm_head)
    print("labels", labels)
    print("shift", lm_head)
    print("reduction", reduction)
    print("logit_scale", logit_scale)
    print("logit_softcapping", logit_softcapping)
    print("ignore_index", ignore_index)
    print("chunk_size", chunk_size)
    print("attention_mask", attention_mask)

    return UnslothEfficientLoss.apply(
        hidden_states,
        lm_head.weight,
        labels,
        lm_head.bias,
        shift,
        reduction,
        logit_scale,
        logit_softcapping,
        torch.nn.CrossEntropyLoss,
        ignore_index,
        chunk_size,
        attention_mask,
    )
pass
