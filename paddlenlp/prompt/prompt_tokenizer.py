# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import itertools
from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np

from paddlenlp.utils.log import logger

__all__ = ["MLMPromptTokenizer"]


class MLMPromptTokenizer(object):
    """
    还要看个特殊的分词器
    """

    omask_token = "[O-MASK]"

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, inputs: List[Dict[str, Any]]):
        """
        主要看看被调用时做了什么
        """
        part_do_truncate = [part["do_truncate"] for part in inputs]

        # 这是最后返回的结果
        encoded_inputs = defaultdict(list)
        option_length = None
        # 位置编码从 1 开始，因为 0 是 [CLS]
        last_position = 1  # Id 0 denotes special token '[CLS]'.
        last_token_type = 0
        orig_input_ids = []
        for index, part in enumerate(inputs):
            # Create input_ids.
            soft_token_ids = part.get("soft_tokens", None)
            if soft_token_ids is None or len(soft_token_ids) == 1 and soft_token_ids[0] == 0:
                # 如果没有 soft_tokens, 或者 soft_tokens 只有一个 0
                orig_input_ids.append(
                    # 就直接用分词器编码 part["text"] 的值
                    self.tokenizer.encode(part["text"], add_special_tokens=False, return_token_type_ids=False)[
                        "input_ids"
                    ]
                )
            else:
                # 直接用 soft_token_ids
                orig_input_ids.append(soft_token_ids)
        # max_lengths 是个数组, 记录了每个部分的可用长度. None 表示不截断
        max_lengths = self._create_max_lengths_from_do_truncate(orig_input_ids, part_do_truncate)

        # 遍历输入的每个部分
        for index, part in enumerate(inputs):
            # Create input_ids.
            soft_token_ids = part.get("soft_tokens", None)
            if soft_token_ids is None or len(soft_token_ids) == 1 and soft_token_ids[0] == 0:
                # 截断到当前部分可用的最大长度
                if self.tokenizer.truncation_side == "left":
                    input_ids = orig_input_ids[index][-max_lengths[index] :]
                else:
                    input_ids = orig_input_ids[index][: max_lengths[index]]
                # 即使没有 soft_token_ids, 也要用 0 填充
                encoded_inputs["soft_token_ids"].append([0] * len(input_ids))
            else:
                input_ids = soft_token_ids
                encoded_inputs["soft_token_ids"].append(soft_token_ids)
            # 然后是添加 input_ids, 并记录当前部分的长度
            encoded_inputs["input_ids"].append(input_ids)
            part_length = len(input_ids)

            # Create position_ids.
            position_ids, last_position = self._create_position_ids_from_part(input_ids, part, last_position)
            encoded_inputs["position_ids"].append(position_ids)

            # Create token_type_ids.
            if "token_types" in part:
                last_token_type = part["token_types"]
            encoded_inputs["token_type_ids"].append([last_token_type] * part_length)

            # Create other features like encoder_ids.
            for name in part:
                if name not in ["text", "soft_tokens", "positions", "token_types"]:
                    encoded_inputs[name].append([part[name]] * part_length)

            # Record the length of options if exists. 记录选项的长度
            if self.omask_token in part["text"]:
                option_length = len(input_ids)

        encoded_inputs.pop("do_truncate")
        encoded_inputs = self.join(encoded_inputs)
        encoded_inputs = self.add_special_tokens(encoded_inputs)
        attention_mask = self._create_attention_mask(encoded_inputs["input_ids"], option_length)
        if attention_mask is not None:
            encoded_inputs["attention_mask"] = attention_mask
        masked_positions = self._create_masked_positions(encoded_inputs["input_ids"], encoded_inputs["soft_token_ids"])
        if masked_positions is not None:
            encoded_inputs["masked_positions"] = masked_positions
        return encoded_inputs

    def _create_position_ids_from_part(self, input_ids: List[int], part: Dict[str, Any], last_position: int):
        """
        Create position ids from prompt for each part.
        创建每个部分的位置编码
        """
        part_length = len(input_ids)
        # 如果当前部分有 positions, 就用 positions
        if "positions" in part and part["positions"] >= 0:
            last_position = part["positions"]
        # 如果输入文本中有 omask_token
        if self.omask_token in part["text"]:
            omask_id = self.tokenizer.convert_tokens_to_ids(self.omask_token)
            # 找到 omask_token 的索引位置
            omask_index = [x for x in range(part_length) if input_ids[x] == omask_id]
            omask_index = [0] + omask_index
            position_ids = []
            max_index = 0  # 记录的是标签的最大长度
            # 前后两个 omask_token 之间的
            for start_id, end_id in zip(omask_index[:-1], omask_index[1:]):
                position_ids.extend(list(range(last_position, last_position + end_id - start_id)))
                max_index = max(end_id - start_id, max_index)
            # 如果数量不够, 就再来一次
            if len(position_ids) < part_length:
                difference = part_length - len(position_ids)
                position_ids.extend(range(last_position, last_position + difference))
                max_index = max(difference, max_index)
            last_position += max_index
        else:
            # 这种情况下, 就是简单的顺序编码
            position_ids = list(range(last_position, last_position + part_length))
            last_position += part_length
        # 返回当前部分的位置编码, 和最后一个位置
        return position_ids, last_position

    def _create_max_lengths_from_do_truncate(self, part_text: List[str], part_do_truncate: List[bool]):
        """
        Create the max sequence length of each part, where the longest part is truncated first.
        创建每个部分的最大序列长度.
        """
        # 当前总长度
        text_length = sum([len(x) for x in part_text])
        # 特殊 token 的数量
        num_special_token = self.tokenizer.num_special_tokens_to_add()
        # 当前可用的最大长度
        max_length = self.max_length - num_special_token
        if text_length <= max_length:
            # 如果长度够用, 就直接返回
            return [None] * len(part_text)
        max_lengths = [None for _ in range(len(part_text))]
        # 将是否截断从 bool 转为 int
        do_truncate = [int(x) for x in part_do_truncate]

        # Remove parts that can not be truncated.
        for index, part in enumerate(part_text):
            # 如果当前部分是不可截断的, 就直接减去最大可用长度
            if not part_do_truncate[index]:
                max_length -= len(part)
            else:
                # 记录当前部分的长度
                max_lengths[index] = len(part)
        # 如果所有的部分都不可截断, 就直接返回
        if sum(do_truncate) == 0:
            logger.warning(
                f"Can not truncate the sequence with length {text_length}. Set more `truncate` attributes as True."
            )
            return max_lengths

        # Remove parts whose length is less than average maximum length of parts to truncate.
        has_short = True
        while has_short:
            # 是否有更短的可以调整
            has_short = False
            # 定义平均长度
            avg_max_length = max_length // sum(do_truncate)
            for index, part in enumerate(part_text):
                # 那些可以截断的, 且长度小于平均长度的.
                # 留下的就是可以截断的, 且长度大于平均长度的
                if do_truncate[index] == 1 and len(part) <= avg_max_length:
                    # 设置成不能截断
                    do_truncate[index] = 0
                    max_lengths[index] = len(part)
                    max_length -= len(part)
                    has_short = True
        # 这个 while 循环跳出的额时候, 剩下的就是那些平均长度大于 max_length // sum(do_truncate) 的. 然后就截断它们
        # 如果长度不足, 就报错
        if max_length < 0:
            raise AssertionError("Actual length has exceeded the maximum length. Check the implementation.")
        avg_max_length = max_length // sum(do_truncate)
        for index in range(len(part_text)):
            if do_truncate[index] == 1:
                # 设置当前部分的最大长度
                max_lengths[index] = min(avg_max_length, max_length)
                max_length -= max_lengths[index]
                if max_length < 0:
                    raise AssertionError("Actual length has exceeded the maximum length. Check the implementation.")
        return max_lengths

    def _create_attention_mask(self, input_ids: List[int], option_length: Union[int, None]):
        if option_length is None:
            return None
        omask_id = self.tokenizer.convert_tokens_to_ids(self.omask_token)
        input_ids = np.array(input_ids)
        attention_mask = np.ones([len(input_ids), len(input_ids)])
        omask_index = np.where(input_ids == omask_id)[0].tolist()
        cls_indices = np.where(input_ids == self.tokenizer.cls_token_id)[0]
        sep_indices = np.where(input_ids == self.tokenizer.sep_token_id)[0]
        cls_index = len(input_ids)
        for idx in cls_indices:
            if idx > omask_index[-1]:
                cls_index = idx
                break
        sep_index = len(input_ids)
        for idx in sep_indices:
            if idx > omask_index[-1]:
                sep_index = idx
                break
        opt_begin = omask_index[0]
        opt_end = min(cls_index, sep_index)
        attention_mask[opt_begin:opt_end, opt_begin:opt_end] = 0
        omask_index.append(opt_end)
        for opt_begin, opt_end in zip(omask_index[:-1], omask_index[1:]):
            attention_mask[opt_begin:opt_end, opt_begin:opt_end] = 1
        attention_mask = (attention_mask - 1) * 1e4
        return attention_mask

    def _create_masked_positions(self, input_ids: List[int], soft_token_ids: List[int]):
        non_soft_ids = np.array(input_ids) * (np.array(soft_token_ids) == 0)
        mask_id = self.tokenizer.mask_token_id

        masked_positions = np.where(non_soft_ids == mask_id)[0]
        if masked_positions.shape[0] == 0:
            return None
        return masked_positions.tolist()

    def add_special_tokens(self, input_dict: Dict[str, Any]):
        """
        添加特殊的 token
        """
        for key in input_dict:
            new_inputs = self.tokenizer.build_inputs_with_special_tokens(input_dict[key])
            if key != "input_ids":
                # 获取特殊 token 的标记
                special_mask = np.array(self.tokenizer.get_special_tokens_mask(input_dict[key]))
                new_inputs = np.array(new_inputs)
                # TODO (Huijuan): Use different ids according to specific keyword.
                # 将是特殊 token 的部分设置为 0
                new_inputs[special_mask == 1] = 0
                new_inputs = new_inputs.tolist()
            input_dict[key] = new_inputs
        return input_dict

    @staticmethod
    def join(input_dict):
        """
        因为 input_dict 的值是由很多部分组成的, 所以需要将嵌套的列表展平
        """
        for key in input_dict:
            # 将嵌套的列表展平
            input_dict[key] = list(itertools.chain(*input_dict[key]))
        return input_dict
