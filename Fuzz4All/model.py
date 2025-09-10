import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
import openai

def is_ollama_model(model_name: str) -> bool:
    return model_name.startswith("ollama/")


def get_ollama_model_name(model_name: str) -> str:
    if is_ollama_model(model_name):
        return model_name.split("/", 1)[1]
    return model_name


def make_model(eos: list, model_name: str, device: str, max_length: int, vllm_server_config: dict = None):
    if vllm_server_config:
        return RemoteStarCoder(vllm_server_config["api_url"],vllm_server_config["api_key"],vllm_server_config["vllm_model_name"],eos, max_length)
    elif is_ollama_model(model_name):
        return None
    else:
        return StarCoder(model_name, device, eos, max_length)


torch.cuda.empty_cache()
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable warning
EOF_STRINGS = ["<|endoftext|>", "###"]


class EndOfFunctionCriteria(StoppingCriteria):
    def __init__(self, start_length, eos, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_length = start_length
        self.eos = eos
        self.tokenizer = tokenizer
        self.end_length = {}

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for index, decoded_generation in enumerate(decoded_generations):
            finished = any(
                [stop_string in decoded_generation for stop_string in self.eos]
            )
            if (
                finished and index not in self.end_length
            ):  # ensures first time we see it
                for stop_string in self.eos:
                    if stop_string in decoded_generation:
                        self.end_length[index] = len(
                            input_ids[
                                index,  # get length of actual generation
                                self.start_length : -len(
                                    self.tokenizer.encode(
                                        stop_string,
                                        add_special_tokens=False,
                                        return_tensors="pt",
                                    )[0]
                                ),
                            ]
                        )
            done.append(finished)
        return all(done)


class StarCoder:
    def __init__(
        self, model_name: str, device: str, eos: List, max_length: int
    ) -> None:
        checkpoint = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                checkpoint,
            )
            .to(torch.bfloat16)
            .to(device)
        )
        self.eos = EOF_STRINGS + eos
        self.max_length = max_length
        self.prefix_token = "<fim_prefix>"
        self.suffix_token = "<fim_suffix><fim_middle>"
        self.skip_special_tokens = False

    @torch.inference_mode()
    def generate(
        self, prompt, batch_size=10, temperature=1.0, max_length=512
    ) -> List[str]:
        input_str = self.prefix_token + prompt + self.suffix_token
        input_tokens = self.tokenizer.encode(input_str, return_tensors="pt").to(
            self.device
        )

        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )

        raw_outputs = self.model.generate(
            input_tokens,
            max_length=min(self.max_length, len(input_tokens[0]) + max_length),
            do_sample=True,
            top_p=1.0,
            temperature=max(temperature, 1e-2),
            num_return_sequences=batch_size,
            stopping_criteria=scores,
            output_scores=True,
            return_dict_in_generate=True,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs

# ===== 公共部分 =====
EOF_STRINGS = ["<|endoftext|>", "<|eot_id|>"]

class BaseStarCoder:
    def __init__(self, eos: List[str], max_length: int):
        self.eos = EOF_STRINGS + eos
        self.max_length = max_length
        self.prefix_token = "<fim_prefix>"
        self.suffix_token = "<fim_suffix><fim_middle>"
        self.skip_special_tokens = False

    def _build_prompt(self, prompt: str) -> str:
        """构造输入 prompt，添加 prefix/suffix。"""
        return self.prefix_token + prompt + self.suffix_token

    def _post_process(self, texts: List[str]) -> List[str]:
        """处理生成的结果，去掉 eos token。"""
        outputs = []
        for text in texts:
            min_index = len(text)
            for eos in self.eos:
                if eos in text:
                    min_index = min(min_index, text.index(eos))
            outputs.append(text[:min_index])
        return outputs

    def generate(
        self, prompt: str, batch_size: int = 1, temperature: float = 1.0, max_length: int = 512
    ) -> List[str]:
        raise NotImplementedError("Subclasses must implement this method.")


# ===== 本地模型推理实现 =====
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList

class LocalStarCoder(BaseStarCoder):
    def __init__(self, model_name: str, device: str, eos: List[str], max_length: int):
        super().__init__(eos, max_length)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name)
            .to(torch.bfloat16)
            .to(device)
        )

    @torch.inference_mode()
    def generate(
        self, prompt: str, batch_size: int = 1, temperature: float = 1.0, max_length: int = 512
    ) -> List[str]:
        input_str = self._build_prompt(prompt)
        input_tokens = self.tokenizer.encode(input_str, return_tensors="pt").to(self.device)

        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )

        raw_outputs = self.model.generate(
            input_tokens,
            max_length=min(self.max_length, len(input_tokens[0]) + max_length),
            do_sample=True,
            top_p=1.0,
            temperature=max(temperature, 1e-2),
            num_return_sequences=batch_size,
            stopping_criteria=scores,
            output_scores=True,
            return_dict_in_generate=True,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(gen_seqs, skip_special_tokens=self.skip_special_tokens)
        return self._post_process(gen_strs)


class RemoteStarCoder(BaseStarCoder):
    def __init__(self, api_url: str, api_key: str, model_name: str, eos: List[str], max_length: int):
        super().__init__(eos, max_length)
        self.openai = openai.OpenAI(api_key=api_key, base_url=api_url)
        # openai.api_key = api_key
        # openai.base_url = api_url.rstrip("/") + "/v1/"
        self.model = model_name

    def generate(
        self, prompt: str, batch_size: int = 1, temperature: float = 1.0, max_length: int = 512
    ) -> List[str]:
        input_str = self._build_prompt(prompt)

        response = self.openai.completions.create(
            model=self.model,
            prompt=input_str,
            max_tokens=min(self.max_length, max_length),
            temperature=max(temperature, 1e-2),
            n=batch_size,
            stop=self.eos,
        )
        texts = [choice.text for choice in response.choices]
        return self._post_process(texts)
