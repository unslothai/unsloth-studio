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
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
from transformers.models.llama4.modeling_llama4 import (
    Llama4Config,
    ACT2FN,
)
from peft.utils.integrations import dequantize_module_weight

class Llama4TextExperts(nn.Module):
    def __init__(self, config: Llama4Config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_proj = nn.Linear(
            self.expert_dim,
            self.num_experts * self.hidden_size,
            bias = False,
        )
        self.up_proj = nn.Linear(
            self.expert_dim,
            self.num_experts * self.hidden_size,
            bias = False,
        )
        self.down_proj = nn.Linear(
            self.hidden_size,
            self.num_experts * self.expert_dim,
            bias = False,
        )
        self.act_fn = ACT2FN[config.hidden_act]
    pass

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        This should really not be run on a single machine, as we are reaching compute bound:
        - the inputs are expected to be "sorted" per expert already.
        - the weights are viewed with another dim, to match num_expert, 1, shape * num_tokens, shape

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            selected_experts (torch.Tensor): (batch_size * token_num, top_k)
            routing_weights (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor
        """
        hidden_states = hidden_states.view(
            self.num_experts,
            -1,
            self.hidden_size,
        )

        gate_proj = dequantize_module_weight(self.gate_proj)
        gate_proj = gate_proj.view(
            self.num_experts,
            self.hidden_size,
            self.expert_dim,
        )

        up_proj = dequantize_module_weight(self.up_proj)
        up_proj = up_proj.view(
            self.num_experts,
            self.hidden_size,
            self.expert_dim,
        )

        down_proj = dequantize_module_weight(self.down_proj)
        down_proj = down_proj.view(
            self.num_experts,
            self.expert_dim,
            self.hidden_size,
        )

        gate = torch.bmm(hidden_states, gate_proj)
        up   = torch.bmm(hidden_states,   up_proj)
        # gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
        next_states = torch.bmm((up * self.act_fn(gate)), down_proj)
        next_states = next_states.view(-1, self.hidden_size)
        return next_states
    pass
pass


def patch_llama4():
    import transformers.models.llama4.modeling_llama4
    transformers.models.llama4.modeling_llama4.Llama4TextExperts = Llama4TextExperts
    return
pass
