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
    "test_efficient_ce_loss",
]
from ..utils import *
from ..losses import (
    UnslothEfficientLoss,
    unsloth_efficient_ce_loss,
)
import torch


def _test_efficient_ce_loss(
    bsz = 4,
    qlen = 2009,
    hd = 4096,
    vocab_size = 16 * 1024,
    dtype = torch.float16,
    reduction = "sum",
    logit_scale = None,
    logit_softcapping = None,
    random_state = 3407,
    has_bias = True,
    weight_requires_grad = True,
    bias_requires_grad   = True,
    ignore_index = -100,
    device = "cuda",
):
    # All Unsloth Studio code licensed under AGPLv3
    torch.cuda.manual_seed(random_state)
    torch.manual_seed(random_state)
    if not has_bias: bias_requires_grad = False

    # Increase atol / rtol for float16 due to float16_scaler being 65536
    # https://pytorch.org/docs/stable/testing.html#torch.testing.assert_close
    if dtype == torch.float16:
        rtol = 1e-3 * 65536 / 100
        atol = 1e-5 * 65536 / 10
        print(rtol, atol)
    else:
        rtol = 1.6e-2
        atol = 1e-5

    # Use grad scaling for float16 as well!
    scaler = torch.amp.GradScaler() if dtype == torch.float16 else None
    hidden_states = torch.randn((bsz, qlen, hd), dtype = dtype, requires_grad = True, device = device)
    if weight_requires_grad or bias_requires_grad:
        lm_head = torch.nn.Linear(hd, vocab_size, bias = has_bias, device = device, dtype = torch.float32)
    else:
        # Downcast to bfloat16
        lm_head = torch.nn.Linear(hd, vocab_size, bias = has_bias, device = device, dtype = dtype)
    lm_head.weight.requires_grad_(weight_requires_grad)
    if has_bias: lm_head.bias.requires_grad_(bias_requires_grad)
    labels = torch.randint(0, vocab_size, (bsz, qlen), device = device).to(torch.int64)
    padding = torch.randint(0, int(qlen * 0.2), (bsz, ), device = device)
    for i in range(bsz):
        labels[i, padding[i]:] = ignore_index

    n_items = (labels != ignore_index).sum()

    # Get original CE Loss
    hidden_states.grad = None
    lm_head.weight.grad = None
    if has_bias: lm_head.bias.grad = None

    with torch.amp.autocast(device_type = "cuda", dtype = dtype):
        logits = lm_head(hidden_states)

    with torch.autocast(device_type = "cuda", enabled = False):
        logits = logits.float()
        # Logit Scaling like in Cohere, Granite
        if logit_scale is not None:
            logits = logits * logit_scale
        # Softcapping like in Gemma 2
        if logit_softcapping is not None:
            logits = logits / logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * logit_softcapping

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(
            reduction = reduction,
            ignore_index = ignore_index,
        )
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss / n_items
        old_loss = loss.detach()
    pass
    if scaler is not None: loss = scaler.scale(loss)
    print(loss)
    loss.backward()
    old_hidden_states_grad = hidden_states.grad.detach()

    old_weight_grad = lm_head.weight.grad if lm_head.weight is not None else None
    old_bias_grad = lm_head.bias.grad if lm_head.bias is not None else None
    if old_weight_grad is not None: old_weight_grad = old_weight_grad.detach()
    if old_bias_grad is not None: old_bias_grad = old_bias_grad.detach()
    print(
        torch.amax(old_hidden_states_grad),
        torch.amax(old_weight_grad) if old_weight_grad is not None else None,
        torch.amax(old_bias_grad) if old_bias_grad is not None else None
    )

    # Get new CE Loss
    hidden_states.grad = None
    lm_head.weight.grad = None
    if has_bias: lm_head.bias.grad = None

    with torch.amp.autocast(device_type = "cuda", dtype = dtype):

        loss = unsloth_efficient_ce_loss(
            hidden_states = hidden_states,
            lm_head = lm_head,
            labels = labels,
            shift = True,
            reduction = reduction,
            logit_scale = logit_scale,
            logit_softcapping = logit_softcapping,
            ignore_index = ignore_index,
            chunk_size = 512,
        )
        loss = loss / n_items
        new_loss = loss.detach()
    pass
    if scaler is not None: loss = scaler.scale(loss)
    print(loss)
    loss.backward()
    torch.testing.assert_close(new_loss, old_loss, atol = 0.1, rtol = 1e-2)
    new_hidden_states_grad = hidden_states.grad.detach()
    new_weight_grad = lm_head.weight.grad if lm_head.weight is not None else None
    new_bias_grad = lm_head.bias.grad if lm_head.bias is not None else None
    if new_weight_grad is not None: new_weight_grad = new_weight_grad.detach()
    if new_bias_grad is not None: new_bias_grad = new_bias_grad.detach()
    print(
        torch.amax(new_hidden_states_grad),
        torch.amax(new_weight_grad) if new_weight_grad is not None else None,
        torch.amax(new_bias_grad) if new_bias_grad is not None else None
    )

    torch.testing.assert_close(new_hidden_states_grad, old_hidden_states_grad, atol = atol, rtol = rtol)
    if weight_requires_grad:
        assert(new_weight_grad is not None and old_weight_grad is not None)
        torch.testing.assert_close(new_weight_grad, old_weight_grad, atol = atol, rtol = rtol)
    else:
        assert(new_weight_grad is None and old_weight_grad is None)

    if bias_requires_grad:
        assert(new_bias_grad is not None and old_bias_grad is not None)
        torch.testing.assert_close(new_bias_grad, old_bias_grad, atol = atol / 100, rtol = rtol / 100)
    else:
        assert(new_bias_grad is None and old_bias_grad is None)
    torch.cuda.empty_cache()
pass


def test_efficient_ce_loss():
    _test_efficient_ce_loss(
        bsz = 4,
        qlen = 2048,
        hd = 4096,
        vocab_size = 16 * 1024,
        dtype = torch.float16,
        reduction = "sum",
        logit_scale = None,
        logit_softcapping = None,
        random_state = 3407,
        has_bias = True,
        weight_requires_grad = True,
        bias_requires_grad   = True,
        ignore_index = -100,
        device = "cuda",
    )
    _test_efficient_ce_loss(
        bsz = 3,
        qlen = 2047,
        hd = 4097,
        vocab_size = 64 * 1024,
        dtype = torch.bfloat16,
        reduction = "sum",
        logit_scale = None,
        logit_softcapping = 50.0,
        random_state = 3409,
        has_bias = False,
        weight_requires_grad = False,
        bias_requires_grad   = False,
        ignore_index = -101,
        device = "cuda",
    )
    _test_efficient_ce_loss(
        bsz = 1,
        qlen = 1023,
        hd = 2048,
        vocab_size = 128 * 1024,
        dtype = torch.float16,
        reduction = "mean",
        logit_scale = 0.125,
        logit_softcapping = None,
        random_state = 3410,
        has_bias = False,
        weight_requires_grad = True,
        bias_requires_grad   = True,
        ignore_index = -200,
        device = "cuda",
    )
    _test_efficient_ce_loss(
        bsz = 1,
        qlen = 512,
        hd = 2000,
        vocab_size = 256 * 1024,
        dtype = torch.bfloat16,
        reduction = "mean",
        logit_scale = 0.125,
        logit_softcapping = 50.0,
        random_state = 3411,
        has_bias = True,
        weight_requires_grad = True,
        bias_requires_grad   = True,
        ignore_index = -10,
        device = "cuda",
    )
pass
