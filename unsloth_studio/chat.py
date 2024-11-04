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

from IPython.display import clear_output
import subprocess
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

print("Installing packages for 🦥 Unsloth Studio ... Please wait 5 minutes ...")

install_first = [
    "pip", "install",
    "unsloth"
]

install_first = subprocess.Popen(install_first)
install_first.wait()

uninstall_install_first = [
    "pip", "uninstall", "-y",
    "unsloth"
]

uninstall_install_first = subprocess.Popen(uninstall_install_first)
uninstall_install_first.wait()

install_second = [
    "pip", "install",
    "gradio==4.44.1",
    "unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git",
]
install_second = subprocess.Popen(install_second)
install_second.wait()

upgrade_huggingface_hub = [
    "pip", "install", 
    "huggingface-hub",
    "-U",
    "--force-reinstall"
]

upgrade_huggingface_hub = subprocess.Popen(upgrade_huggingface_hub)
upgrade_huggingface_hub.wait()


from huggingface_hub import snapshot_download
import warnings
warnings.filterwarnings(action = "ignore", category = UserWarning,    module = "torch")
warnings.filterwarnings(action = "ignore", category = UserWarning,    module = "huggingface_hub")
warnings.filterwarnings(action = "ignore", category = FutureWarning,  module = "huggingface_hub")
warnings.filterwarnings(action = "ignore", category = RuntimeWarning, module = "subprocess")
warnings.filterwarnings(action = "ignore", category = UserWarning,    module = "transformers")
warnings.filterwarnings(action = "ignore", category = FutureWarning,  module = "accelerate")
warnings.filterwarnings(action = "ignore", category = RuntimeWarning, module = "multiprocessing")
warnings.filterwarnings(action = "ignore", category = RuntimeWarning, module = "multiprocess")

from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()
snapshot_download(repo_id = MODEL_NAME, repo_type = "model")

clear_output()


from contextlib import redirect_stdout
import io
import logging
logging.getLogger("transformers.utils.hub").setLevel(logging.CRITICAL+1)

print("Loading model ... Please wait 1 more minute! ...")

with redirect_stdout(io.StringIO()):
    from unsloth import FastLanguageModel
    import torch
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = None,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
pass
clear_output()

import gradio
gradio.strings.en["SHARE_LINK_DISPLAY"] = ""
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from threading import Thread

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = tuple(set(stop_token_ids))
    pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1].item() in self.stop_token_ids
    pass
pass

def async_process_chatbot(message, history):
    eos_token = tokenizer.eos_token
    stop_on_tokens = StopOnTokens([eos_token,])
    text_streamer  = TextIteratorStreamer(tokenizer, skip_prompt = True)

    # From https://www.gradio.app/guides/creating-a-chatbot-fast
    history_transformer_format = history + [[message, ""]]
    messages = []
    for item in history_transformer_format:
        messages.append({"role": "user",      "content": item[0]})
        messages.append({"role": "assistant", "content": item[1]})
    pass
    # Remove last assistant and instead use add_generation_prompt
    messages.pop(-1)

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda", non_blocking = True)

    # Add stopping criteria - will not output EOS / EOT
    generation_kwargs = dict(
        input_ids = input_ids,
        streamer = text_streamer,
        max_new_tokens = 1024,
        stopping_criteria = StoppingCriteriaList([stop_on_tokens,]),
        temperature = 0.7,
        do_sample = True,
    )
    thread = Thread(target = model.generate, kwargs = generation_kwargs)
    thread.start()

    # Yield will save the output to history!
    generated_text = ""
    for new_text in text_streamer:
        if new_text.endswith(eos_token):
            new_text = new_text[:len(new_text) - len(eos_token)]
        generated_text += new_text
        yield generated_text
    pass
pass

studio_theme = gradio.themes.Soft(
    primary_hue = "teal",
)

scene = gradio.ChatInterface(
    async_process_chatbot,
    chatbot = gradio.Chatbot(
        height = 325,
        label = "Unsloth Studio Chat",
    ),
    textbox = gradio.Textbox(
        placeholder = "Message Unsloth Chat",
        container = False,
    ),
    title = None,
    theme = studio_theme,
    examples = None,
    cache_examples = False,
    retry_btn = None,
    undo_btn = "Remove Previous Message",
    clear_btn = "Restart Entire Chat",
)

scene.launch(quiet = True)
