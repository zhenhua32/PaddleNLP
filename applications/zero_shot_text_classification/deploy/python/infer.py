# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import distutils.util
import os
from typing import Any, Dict, List, Union

import fastdeploy as fd
import numpy as np

from paddlenlp.prompt import PromptDataCollatorWithPadding, UTCTemplate
from paddlenlp.transformers import AutoTokenizer


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="The directory of model.")
    parser.add_argument("--vocab_path", type=str, default="", help="The path of tokenizer vocab.")
    parser.add_argument("--model_prefix", type=str, default="model", help="The model and params file prefix.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["gpu", "cpu"],
        help="Type of inference device, support 'cpu' or 'gpu'.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="paddle",
        choices=["onnx_runtime", "paddle", "tensorrt", "paddle_tensorrt"],
        help="The inference runtime backend.",
    )
    parser.add_argument(
        "--pred_threshold",
        default=0.5,
        type=float,
        help="Probability threshold for start/end index probabiliry.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size of data.")
    parser.add_argument("--max_length", type=int, default=128, help="The max length of sequence.")
    parser.add_argument("--num_omask_tokens", type=int, default=64, help="The max length of sequence.")
    parser.add_argument("--log_interval", type=int, default=10, help="The interval of logging.")
    parser.add_argument("--use_fp16", type=distutils.util.strtobool, default=False, help="Wheter to use FP16 mode")
    parser.add_argument("--cpu_threads", type=int, default=1, help="Number of threads to predict when using cpu.")
    parser.add_argument("--device_id", type=int, default=0, help="Select which gpu device to train model.")
    return parser.parse_args()


class Predictor(object):
    def __init__(self, args, schema: list = None):
        self.set_schema(schema)
        self.tokenizer = AutoTokenizer.from_pretrained(args.vocab_path)
        print("self.tokenizer", self.tokenizer)
        self.runtime = self.create_fd_runtime(args)
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.template = UTCTemplate(self.tokenizer, self.max_length)
        self.collator = PromptDataCollatorWithPadding(self.tokenizer, return_tensors="np")
        self.pred_threshold = args.pred_threshold

    def set_schema(self, schema):
        """
        设置 question 和 choices
        """
        if schema is None:
            self._question = None
            self._choices = None
        elif isinstance(schema, list):
            self._question = ""
            self._choices = schema
        elif isinstance(schema, dict) and len(schema) == 1:
            for key, value in schema.items():
                self._question = key
                self._choices = value
        else:
            raise ValueError(f"Invalid schema: {schema}.")

    def _check_input_text(self, inputs) -> list:
        """
        检查输入文本
        """
        # 将输入转换成 list
        if isinstance(inputs, str) or isinstance(inputs, dict):
            inputs = [inputs]

        if isinstance(inputs, list):
            input_list = []
            for example in inputs:
                # 构建一个样本
                data = {"text_a": "", "text_b": "", "choices": self._choices, "question": self._question}
                if isinstance(example, dict):
                    # 替换掉默认的值
                    for k, v in example.items():
                        if k in data:
                            data[k] = example[k]
                elif isinstance(example, str):
                    # 就是纯文本
                    data["text_a"] = example
                    data["text_b"] = ""
                elif isinstance(example, list):
                    # 取第一个作为 text_a，剩下的作为 text_b
                    for x in example:
                        if not isinstance(x, str):
                            raise ValueError("Invalid inputs, input text should be strings.")
                    data["text_a"] = example[0]
                    data["text_b"] = "".join(example[1:]) if len(example) > 1 else ""
                else:
                    raise ValueError(
                        "Invalid inputs, the input should be {'text_a': a, 'text_b': b}, a text or a list of text."
                    )

                # 检查输入字段
                if len(data["text_a"]) < 1 and len(data["text_b"]) < 1:
                    raise ValueError("Invalid inputs, input `text_a` and `text_b` are both missing or empty.")
                if not isinstance(data["choices"], list) or len(data["choices"]) < 2:
                    raise ValueError("Invalid inputs, label candidates should be a list with length >= 2.")
                if not isinstance(data["question"], str):
                    raise ValueError("Invalid inputs, prompt question should be a string.")
                input_list.append(data)
        else:
            raise TypeError("Invalid input format!")
        return input_list

    def create_fd_runtime(self, args):
        """
        初始化 fastdeploy runtime 并返回
        """
        option = fd.RuntimeOption()
        model_path = os.path.join(args.model_dir, args.model_prefix + ".pdmodel")
        params_path = os.path.join(args.model_dir, args.model_prefix + ".pdiparams")
        option.set_model_path(model_path, params_path)

        # 支持 CPU 和 GPU 部署
        if args.device == "cpu":
            option.use_cpu()
            option.set_cpu_thread_num(args.cpu_threads)
        else:
            option.use_gpu(args.device_id)

        # 支持多种后端
        if args.backend == "paddle":
            option.use_paddle_infer_backend()
        elif args.backend == "onnx_runtime":
            option.use_ort_backend()
        elif args.backend == "openvino":
            option.use_openvino_backend()
        else:
            # trt 的最复杂, 没看
            option.use_trt_backend()
            if args.backend == "paddle_tensorrt":
                option.use_paddle_infer_backend()
                option.paddle_infer_option.collect_trt_shape = True
                option.paddle_infer_option.enable_trt = True
            trt_file = os.path.join(args.model_dir, "model.trt")
            option.trt_option.set_shape(
                "input_ids", [1, 1], [args.batch_size, args.max_length], [args.batch_size, args.max_length]
            )
            option.trt_option.set_shape(
                "token_type_ids", [1, 1], [args.batch_size, args.max_length], [args.batch_size, args.max_length]
            )
            option.trt_option.set_shape(
                "position_ids", [1, 1], [args.batch_size, args.max_length], [args.batch_size, args.max_length]
            )
            option.trt_option.set_shape(
                "attention_mask",
                [1, 1, 1, 1],
                [args.batch_size, 1, args.max_length, args.max_length],
                [args.batch_size, 1, args.max_length, args.max_length],
            )
            option.trt_option.set_shape(
                "omask_positions",
                [1, 1],
                [args.batch_size, args.num_omask_tokens],
                [args.batch_size, args.num_omask_tokens],
            )
            option.trt_option.set_shape("cls_positions", [1], [args.batch_size], [args.batch_size])
            if args.use_fp16:
                option.trt_option.enable_fp16 = True
                trt_file = trt_file + ".fp16"
            option.trt_option.serialize_file = trt_file
        return fd.Runtime(option)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def preprocess(self, inputs: Union[str, List[str]]) -> Dict[str, Any]:
        """
        将文本转换成模型需要的输入
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        # Get the config from the kwargs
        # 使用模板对输入进行编码
        tokenized_inputs = [self.template(i) for i in inputs]
        # 按照 batch_size 切分
        batches = [
            tokenized_inputs[idx : idx + self.batch_size] for idx in range(0, len(tokenized_inputs), self.batch_size)
        ]
        outputs = {}
        outputs["text"] = inputs
        # 调用填充
        outputs["batches"] = [self.collator(batch) for batch in batches]

        return outputs

    def infer(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行推理
        """
        outputs = {}
        outputs["text"] = inputs["text"]
        outputs["batch_logits"] = []
        # 定义数据类型
        dtype_list = ["int64", "int64", "int64", "float32", "int64", "int64"]
        for batch in inputs["batches"]:
            batch = dict(batch)
            for i in range(self.runtime.num_inputs()):
                input_name = self.runtime.get_input_info(i).name
                # 转换类型
                batch[input_name] = batch[input_name].astype(dtype_list[i])
            del batch["soft_token_ids"]
            # 推理, 获取 logits
            logits = self.runtime.infer(batch)[0]
            outputs["batch_logits"].append(logits)
        return outputs

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        后处理
        """
        outputs = []
        for logits in inputs["batch_logits"]:
            # 这是只有多标签的, 没写单标签的 softmax
            # TODO: 支持单标签
            scores = self.sigmoid(np.array(logits))
            output = {}
            output["predictions"] = []
            for i, class_score in enumerate(scores[0]):
                # 添加超过阈值的
                if class_score > self.pred_threshold:
                    output["predictions"].append({"label": i, "score": class_score})
            outputs.append(output)

        for i, output in enumerate(outputs):
            if len(inputs["text"][i]["text_a"]) > 0:
                output["text_a"] = inputs["text"][i]["text_a"]
            if len(inputs["text"][i]["text_b"]) > 0:
                output["text_b"] = inputs["text"][i]["text_b"]
            for j, pred in enumerate(output["predictions"]):
                output["predictions"][j] = {
                    # 从 index 转换成 label 文本
                    "label": inputs["text"][i]["choices"][pred["label"]],
                    "score": pred["score"],
                }

        return outputs

    def predict(self, texts):
        """
        预测的入口函数
        """
        inputs = self.preprocess(texts)
        outputs = self.infer(inputs)
        results = self.postprocess(outputs)
        return results


if __name__ == "__main__":
    args = parse_arguments()
    # schema 是选项
    # predictor = Predictor(args, schema=["这是一条差评", "这是一条好评"])
    schema = ['故事', '文化', '娱乐', '体育', '财经', '房产', '汽车', '教育', '科技', '军事', '旅游', '国际', '股票', '农业', '电竞']
    predictor = Predictor(args, schema=schema)
    results = predictor.predict("农村依然很重视土葬")
    print(results)

"""
python infer.py --model_dir G:\code\github\PaddleNLP\outputs\tnews\export_model --vocab_path G:\code\github\PaddleNLP\outputs\tnews --device cpu
python infer.py --model_dir G:\code\github\PaddleNLP\outputs\tnews\export_model --vocab_path G:\code\github\PaddleNLP\outputs\tnews --device gpu
"""
