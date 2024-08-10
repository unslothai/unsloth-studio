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
    "merge_and_overwrite_lora",
]

from huggingface_hub import (
    HfFileSystem,
    snapshot_download,
    hf_hub_download,
    HfApi,
    whoami,
)
from safetensors import safe_open
from safetensors.torch import save_file
from collections import OrderedDict
from tqdm import tqdm as ProgressBar
import os, shutil
from unsloth.kernels import get_lora_parameters
from unsloth.save import create_huggingface_repo
from unsloth.models.loader import get_model_name


def _merge_and_overwrite_lora(save_location, filename, lora_weights, lora_scaling,):
    # Merges LoRA and overwrites the safetensors file it was merged to
    filename = os.path.join(save_location, filename)
    tensors = OrderedDict()
    with safe_open(filename, framework = "pt", device = "cpu") as file:
        for key in file.keys():
            W = file.get_tensor(key)
            if key in lora_weights:
                A, B = lora_weights[key]
                old_dtype = W.dtype
                W = W.to("cuda", dtype = torch.float32, non_blocking = True)

                W = W.addmm_(B.to(torch.float32), A.to(torch.float32), alpha = lora_scaling)

                maximum_element = torch.max(W.min().abs(), W.max())
                if not torch.isfinite(maximum_element).item():
                    raise ValueError(f"Unsloth: Merge failed.\n{key} has some elements = infinity.")
                W = W.to(old_dtype)
            pass
            tensors[key] = W
        pass
    pass
    save_file(tensors, filename, metadata = {"format": "pt"})
pass


def merge_and_overwrite_lora(
    model,
    save_location = "model",
    push_to_hub = False,
    token = None,
    upload_location = None,
    low_disk_space_usage = True,
    private = False,
):
    ignore_files = [
        "*.gitattributes",
        "*.md",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
    ]
    model_name = get_model_name(model.config._name_or_path, load_in_4bit = False)
    print(f"Unsloth: Merging QLoRA weights directly to the 16bit version of {model_name}.")

    if push_to_hub and upload_location is None:
        raise RuntimeError(
            "Unsloth: You're trying to upload to a HuggingFace repo, but did not provide an `upload_location`. Please do!"
        )
    pass

    if upload_location is not None:
        upload_location, hf_api = create_huggingface_repo(
            model = model,
            save_directory = upload_location,
            token = token,
            private = private,
        )
    pass

    # Find all LoRA A and B matrices
    lora_weights = {}
    for name, param in model.named_parameters():
        if "lora_A" in name:
            name = name[name.find("model.layers"):]
            name = name.replace(".lora_A.default", "")
            lora_weights[name] = [param, None,]
        elif "lora_B" in name:
            name = name[name.find("model.layers"):]
            name = name.replace(".lora_B.default", "")
            lora_weights[name][1] = param
        pass
    pass
    # Get LoRA scaling factor
    lora_scaling = get_lora_parameters(model.model.model.layers[0].self_attn.q_proj)[-1]

    # Only enable low_disk_space_usage for uploading
    if upload_location is not None and low_disk_space_usage:
        file_list = HfFileSystem().ls(model_name, detail = False)
        file_list = [x for x in file_list if x.endswith(".safetensors")]
        file_list = [x[len(model_name):].strip("/\\") for x in file_list if x.startswith(model_name)]

        # Download other items that are not .safetensors
        snapshot_download(
            repo_id = model_name,
            local_dir = save_location,
            ignore_patterns = ["*.safetensors"] + ignore_files,
        )

        for filename in ProgressBar(file_list):
            hf_hub_download(
                repo_id = model_name,
                filename = filename,
                repo_type = "model",
                local_dir = save_location,
            )
            _merge_and_overwrite_lora(
                save_location = save_location,
                filename = filename,
                lora_weights = lora_weights,
                lora_scaling = lora_scaling,
            )

            if upload_location is not None:
                location_to_file = os.path.join(save_location, filename)
                hf_api.upload_file(
                    path_or_fileobj = location_to_file,
                    path_in_repo = filename,
                    repo_id = upload_location,
                    repo_type = "model",
                    commit_message  = "(Trained with Unsloth)",
                )
                # Remove safetensors file
                os.remove(location_to_file)
            pass
        pass

        # Upload rest of files that are not safetensors
        if upload_location is not None:
            hf_api.upload_folder(
                folder_path = save_location,
                repo_id = upload_location,
                repo_type = "model",
                commit_message  = "(Trained with Unsloth)",
                ignore_patterns = ["*.safetensors"] + ignore_files,
            )
            # Delete entire repo at the end!
            shutil.rmtree(save_location, ignore_errors = True)
        pass
    else:
        # Download entire repo in 1 call
        snapshot_download(
            repo_id = model_name,
            local_dir = save_location,
            ignore_patterns = ignore_files,
        )

        file_list = os.listdir(save_location)
        file_list = [x for x in file_list if x.endswith(".safetensors")]
        for filename in ProgressBar(file_list):
            _merge_and_overwrite_lora(
                save_location = save_location,
                filename = filename,
                lora_weights = lora_weights,
                lora_scaling = lora_scaling,
            )
        pass

        # Upload repo
        if upload_location is not None:
            hf_api.upload_folder(
                folder_path = save_location,
                repo_id = upload_location,
                repo_type = "model",
                commit_message  = "(Trained with Unsloth)",
                ignore_patterns = ignore_files,
            )
        pass
    pass
pass
