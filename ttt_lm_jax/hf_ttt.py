# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Xinyang Geng
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

# This script converts LLaMA model checkpoint trained by EsayLM to the
# HuggingFace transformers LLaMA PyTorch format, which can then be loaded
# by HuggingFace transformers.
"""
python convert_to_hf.py --load_checkpoint='trainstate_params::/nlp/scr/yusun/data/jiarui/easylm_ckpts/LLAMA-1B/06_30_TTT_Linear_1B/step_50000/streaming_train_state_50000' --output_dir=/nlp/scr/yusun/data/jiarui/easylm_to_hf_ckpts_release_karan/LLAMA-1B/06_30_TTT_Linear_1B/hf_50000 --tokenizer_path=meta-llama/Llama-2-7b-hf --model_size 1b-TTT
python convert_to_hf.py --load_checkpoint='trainstate_params::/nlp/scr/yusun/data/jiarui/easylm_ckpts/LLAMA-1B/06_30_TTT_MLP_1B/step_50000/streaming_train_state_50000' --output_dir=/nlp/scr/yusun/data/jiarui/easylm_to_hf_ckpts_release_karan/LLAMA-1B/06_30_TTT_MLP_1B/hf_50000 --tokenizer_path=meta-llama/Llama-2-7b-hf --model_size 1b-TTT --update_model_config="dict(seq_modeling_block='ttt_mlp', ttt_base_lr=0.1, ttt_base_lr_init=0.01, ttt_base_lr_warmup=5000)"
"""

import gc
import json
import math
import os
import shutil

import numpy as np
import mlxu
import jax
import jax.numpy as jnp
import flax
from flax.traverse_util import flatten_dict
import torch
from transformers import AutoTokenizer
import sys

from ttt.infra.checkpoint import StreamingCheckpointer
from ttt.infra.jax_utils import float_tensor_to_dtype
from ttt.models.model import CONFIGS, ModelConfig
from tttM import TTTForCausalLM, TTTConfig, TTT_STANDARD_CONFIGS

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    load_checkpoint='',
    tokenizer_path='',
    model_size='125m-TTT',
    output_dir='',
    update_model_config='',
)


def match_keywords(string, positives, negatives):
    for positive in positives:
        if positive not in string:
            return False
    for negative in negatives:
        if negative in string:
            return False
    return True


def transpose_last_two_dims(tensor):
    if tensor.ndim < 2:
        raise ValueError("Input tensor must have at least two dimensions")

    # Generate a list of axes in the original order
    axes = list(range(tensor.ndim))

    # Swap the last two axes
    axes[-2], axes[-1] = axes[-1], axes[-2]

    # Transpose the tensor according to the new axes order
    return np.transpose(tensor, axes=axes)


def load_and_convert_checkpoint(path):
    _, flax_params = StreamingCheckpointer.load_trainstate_checkpoint(path)
    flax_params = flatten_dict(flax_params['params'], sep='.')
    torch_params = {}
    for key, tensor in flax_params.items():
        if match_keywords(key, ["kernel"], ["norm", 'ln_f']):
            if tensor.ndim > 2:
                if 'conv' in key:
                    tensor = np.transpose(tensor, (2, 1, 0))
                else:
                    tensor = transpose_last_two_dims(tensor)
            else:
                tensor = tensor.T
        torch_params[key] = torch.tensor(
            float_tensor_to_dtype(tensor, 'fp32'), dtype=torch.float32
        )
    print('Loaded and converted the checkpoint.')
    print(torch_params.keys())
    return torch_params


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


NAME_MAP = {
    'ttt_linear_base': 'linear',
    'ttt_mlp_base': 'mlp',
    'ttt_linear': 'linear',
    'ttt_mlp': 'mlp',
}


def write_model(loaded, model_path, config, input_tokenizer_path):
    # Skip tokenizer loading completely
    # tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_path)
    # tokenizer.save_pretrained(model_path)
    
    print("Skipping tokenizer loading - focusing only on model conversion")

    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    # Continue with the rest of your function to convert model weights
    n_layers = config.num_hidden_layers
    

    param_count = 0
    index_dict = {"weight_map": {}}
    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        state_dict = {
            f"model.layers.{layer_i}.seq_modeling_block.q_proj.weight": loaded[f"model.h.{layer_i}.seq_modeling_block.wq.kernel"],
            # f"model.layers.{layer_i}.self_attn.k_proj.weight": loaded[f"model.h.{layer_i}.seq_modeling_block.wk.kernel"],
            f"model.layers.{layer_i}.seq_modeling_block.v_proj.weight": loaded[f"model.h.{layer_i}.seq_modeling_block.wv.kernel"],
            f"model.layers.{layer_i}.seq_modeling_block.o_proj.weight": loaded[f"model.h.{layer_i}.seq_modeling_block.wo.kernel"],

            f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded[f"model.h.{layer_i}.feed_forward.w1.kernel"],
            f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"model.h.{layer_i}.feed_forward.w2.kernel"],
            f"model.layers.{layer_i}.mlp.up_proj.weight": loaded[f"model.h.{layer_i}.feed_forward.w3.kernel"],

            f"model.layers.{layer_i}.seq_norm.weight": loaded[f"model.h.{layer_i}.seq_norm.kernel"],
            f"model.layers.{layer_i}.ffn_norm.weight": loaded[f"model.h.{layer_i}.ffn_norm.kernel"],
        }
        if 'base' not in config.seq_modeling_block:
            state_dict[f"model.layers.{layer_i}.seq_modeling_block.conv_q.weight"] = loaded[f"model.h.{layer_i}.seq_modeling_block.conv_q.kernel"]
            state_dict[f"model.layers.{layer_i}.seq_modeling_block.conv_q.bias"] = loaded[f"model.h.{layer_i}.seq_modeling_block.conv_q.bias"]
            state_dict[f"model.layers.{layer_i}.seq_modeling_block.conv_k.weight"] = loaded[f"model.h.{layer_i}.seq_modeling_block.conv_k.kernel"]
            state_dict[f"model.layers.{layer_i}.seq_modeling_block.conv_k.bias"] = loaded[f"model.h.{layer_i}.seq_modeling_block.conv_k.bias"]
        else:
            state_dict[f"model.layers.{layer_i}.seq_modeling_block.k_proj.weight"] = loaded[f"model.h.{layer_i}.seq_modeling_block.wk.kernel"]
        state_dict[f"model.layers.{layer_i}.seq_modeling_block.ttt_norm_weight"] = loaded[f"model.h.{layer_i}.seq_modeling_block.ttt_norm/scale"]
        state_dict[f"model.layers.{layer_i}.seq_modeling_block.ttt_norm_bias"] = loaded[f"model.h.{layer_i}.seq_modeling_block.ttt_norm/bias"]
        state_dict[f"model.layers.{layer_i}.seq_modeling_block.learnable_ttt_lr_weight"] = loaded[f"model.h.{layer_i}.seq_modeling_block.learnable_ttt_lr/kernel"]
        state_dict[f"model.layers.{layer_i}.seq_modeling_block.learnable_ttt_lr_bias"] = loaded[f"model.h.{layer_i}.seq_modeling_block.learnable_ttt_lr/bias"]
        state_dict[f"model.layers.{layer_i}.seq_modeling_block.W1"] = loaded[f"model.h.{layer_i}.seq_modeling_block.ttt_dense_0" ]
        state_dict[f"model.layers.{layer_i}.seq_modeling_block.b1"] = loaded[f"model.h.{layer_i}.seq_modeling_block.ttt_bias_0" ]
        if NAME_MAP[config.seq_modeling_block] == 'mlp':
            state_dict[f"model.layers.{layer_i}.seq_modeling_block.W2"] = loaded[f"model.h.{layer_i}.seq_modeling_block.ttt_dense_1" ]
            state_dict[f"model.layers.{layer_i}.seq_modeling_block.b2"] = loaded[f"model.h.{layer_i}.seq_modeling_block.ttt_bias_1" ]
        if config.pre_conv:
            state_dict[f"model.layers.{layer_i}.conv.norm.weight"] = loaded[f"model.h.{layer_i}.conv.conv_norm.kernel"]
            state_dict[f"model.layers.{layer_i}.conv.conv.weight"] = loaded[f"model.h.{layer_i}.conv.conv1.kernel"]
            state_dict[f"model.layers.{layer_i}.conv.conv.bias"] = loaded[f"model.h.{layer_i}.conv.conv1.bias"]
        if 'base' not in config.seq_modeling_block:
            state_dict[f"model.layers.{layer_i}.seq_modeling_block.g_proj.weight"] = loaded[f"model.h.{layer_i}.seq_modeling_block.wg.kernel"]
        state_dict[f"model.layers.{layer_i}.seq_modeling_block.post_norm.weight"] = loaded[f"model.h.{layer_i}.seq_modeling_block.post_norm.scale"]
        state_dict[f"model.layers.{layer_i}.seq_modeling_block.post_norm.bias"] = loaded[f"model.h.{layer_i}.seq_modeling_block.post_norm.bias"]

        state_dict[f"model.layers.{layer_i}.seq_modeling_block.learnable_token_idx"] = loaded[f"model.h.{layer_i}.seq_modeling_block.learnable_token_idx"]

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    # Unsharded
    state_dict = {
        "model.embed_tokens.weight": loaded["model.wte.embedding"],
        "model.norm.weight": loaded["model.ln_f.kernel"],
        # "lm_head.weight": loaded["lm_head.kernel"],
    }
    if not config.tie_word_embeddings:
        state_dict["lm_head.weight"] = loaded["lm_head.kernel"]

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))
    ttt_config = TTTConfig(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=config.num_hidden_layers,
        tie_word_embeddings=config.tie_word_embeddings,
        share_qk="base" not in config.seq_modeling_block,
        use_gate="base" not in config.seq_modeling_block,
        ttt_layer_type=NAME_MAP[config.seq_modeling_block],
        ttt_base_lr=config.ttt_base_lr,
        mini_batch_size=config.mini_batch_size,
        pre_conv=config.pre_conv,
        conv_kernel=config.conv_width,
    )
    ttt_config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()

    print("Loading the checkpoint in a Ttt model.")
    model = TTTForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.float32, vocab_size=16384, use_cache=False)
    # Avoid saving this as part of the config.
    del model.config._name_or_path

    print("Saving in the HF Transformers format.")
    model.save_pretrained(model_path)
    shutil.rmtree(tmp_model_path)

def main(argv):
    assert FLAGS.load_checkpoint != "" and FLAGS.output_dir != "" and FLAGS.tokenizer_path != ""
    assert FLAGS.model_size in CONFIGS
    model_config = ModelConfig.from_dict(CONFIGS[FLAGS.model_size])
    if FLAGS.update_model_config != '':
        model_config_update = FLAGS.update_model_config
        update_dic = dict(eval(model_config_update))
        # update_dic has to overlap with model_config
        update_keys = set(update_dic.keys())
        original_keys = set(model_config.__dict__.keys())
        assert update_keys.issubset(original_keys), f"Update keys {update_keys-original_keys} not in model_config"
        model_config.update(update_dic)

    write_model(
        load_and_convert_checkpoint(FLAGS.load_checkpoint),
        model_path=FLAGS.output_dir,
        config=model_config,
        input_tokenizer_path=FLAGS.tokenizer_path,
    )


if __name__ == "__main__":
    mlxu.run(main)