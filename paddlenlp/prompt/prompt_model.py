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


from typing import Any, Dict, Optional

import paddle
from paddle.static import InputSpec

from ..transformers.model_outputs import (
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    SequenceClassifierOutput,
)
from .prompt_utils import signature
from .template import PrefixTemplate, Template
from .verbalizer import Verbalizer


class PromptModelForSequenceClassification(paddle.nn.Layer):
    """
    用于分类任务的 PromptModel
    PromptModel for classification tasks.
    """

    def __init__(
        self,
        model: paddle.nn.Layer,
        template: Template,
        verbalizer: Optional[Verbalizer] = None,
        freeze_plm: bool = False,
        freeze_dropout: bool = False,
    ):
        super(PromptModelForSequenceClassification, self).__init__()
        self.plm = model
        self.template = template
        self.verbalizer = verbalizer
        self.freeze_plm = freeze_plm
        self.freeze_dropout = freeze_dropout
        # 冻结参数
        if self.freeze_plm:
            for param in self.plm.parameters():
                param.stop_gradient = True
            if self.freeze_dropout:
                self.plm.eval()
        # 获取参数名字
        self.forward_keys = signature(self.plm.forward)
        self._mask_token_id = self.template.tokenizer.mask_token_id
        self._pad_token_id = self.template.tokenizer.pad_token_id
        # 前缀模板特殊处理
        if isinstance(self.template, PrefixTemplate):
            self.plm = self.template.process_model(self.plm)
            self.forward_keys.append("past_key_values")

    def forward(
        self,
        input_ids: paddle.Tensor,
        token_type_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        masked_positions: Optional[paddle.Tensor] = None,
        soft_token_ids: Optional[paddle.Tensor] = None,
        encoder_ids: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Dict[str, Any]
    ):
        """
        主要还是看看模型的前向传播过程
        """
        return_dict = return_dict if return_dict is not None else False
        return_hidden_states = kwargs.get("return_hidden_states", False)
        input_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "masked_positions": masked_positions,
            "soft_token_ids": soft_token_ids,
            "attention_mask": attention_mask,
            "encoder_ids": encoder_ids,
            **kwargs,
        }
        # 批量处理数据, 很多模板类下这个都是空的, 即原样返回
        input_dict = self.template.process_batch(input_dict)
        # 又合并了一遍
        input_dict = {**input_dict, **kwargs}
        # 只需要模型的输入字段
        model_inputs = {k: input_dict[k] for k in input_dict if k in self.forward_keys}
        # masked_positions 字段不能用
        if "masked_positions" in model_inputs:
            model_inputs.pop("masked_positions")
        model_outputs = self.plm(**model_inputs, return_dict=True)
        if isinstance(model_outputs, MaskedLMOutput):
            if self.verbalizer is not None:
                logits = self.verbalizer.process_outputs(model_outputs.logits, input_dict["masked_positions"])
                num_labels = len(self.verbalizer.label_words)
            else:
                raise Exception("Verbalizer is required when model uses the MaskedLM head")
        elif isinstance(model_outputs, SequenceClassifierOutput):
            logits = model_outputs.logits
            num_labels = self.plm.num_labels if self.plm.num_labels is not None else self.plm.num_labels
        elif isinstance(model_outputs, MultipleChoiceModelOutput):
            # UTC 模型是这里
            logits = model_outputs.logits
            num_labels = -1
        else:
            raise Exception(f"Model type not support yet: {type(model_outputs)}")

        loss = None
        if labels is not None:
            if num_labels == 1:
                loss_fct = paddle.nn.MSELoss()
                loss = loss_fct(logits, labels)
            elif num_labels > 0 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int32):
                loss_fct = paddle.nn.CrossEntropyLoss()
                loss = loss_fct(logits.reshape((-1, num_labels)), labels.reshape((-1,)))
            else:
                # UTC 模型选这里
                loss_fct = paddle.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 第一个字段是 logits
        if not return_dict:
            output = (logits,)
            if return_hidden_states:
                output = output + (model_outputs.logits,)
            if loss is not None:
                return (loss,) + output
            if isinstance(output, (list, tuple)) and len(output) == 1:
                output = output[0]
            return output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=model_outputs.logits,
        )

    def prompt_parameters(self):
        """
        获取模板和 verbalizer 的参数. verbalizer 是标签的映射
        Get the parameters of template and verbalizer.
        """
        params = [p for p in self.template.parameters()]
        if self.verbalizer is not None:
            params += [p for p in self.verbalizer.parameters()]
        return params

    def get_input_spec(self):
        """
        获取输入字段定义, 返回一个列表
        """
        template_keywords = self.template.extract_template_keywords(self.template.prompt)
        # 四个固定的输入字段
        input_spec = [
            InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="position_ids"),
            InputSpec(shape=[None, None, None, None], dtype="float32", name="attention_mask"),
        ]
        if "mask" in template_keywords:
            input_spec.append(InputSpec(shape=[None], dtype="int64", name="masked_positions"))
        if "soft" in template_keywords:
            # Add placeholder for argument `masked_positions` if not exists.
            if "mask" not in template_keywords:
                input_spec.append(None)
            input_spec.append(InputSpec(shape=[None, None], dtype="int64", name="soft_token_ids"))
            if "encoder" in template_keywords:
                input_spec.append(InputSpec(shape=[None, None], dtype="int64", name="encoder_ids"))
        return input_spec
