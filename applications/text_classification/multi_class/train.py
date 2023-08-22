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

import functools
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import paddle
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from utils import log_metrics_debug, preprocess_function, read_local_dataset

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import (
    CompressionArguments,
    EarlyStoppingCallback,
    PdArgumentParser,
    Trainer,
)
from paddlenlp.transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    export_model,
)
from paddlenlp.utils.log import logger

# 可使用的模型列表
SUPPORTED_MODELS = [
    "ernie-1.0-large-zh-cw",
    "ernie-1.0-base-zh-cw",
    "ernie-3.0-xbase-zh",
    "ernie-3.0-base-zh",
    "ernie-3.0-medium-zh",
    "ernie-3.0-micro-zh",
    "ernie-3.0-mini-zh",
    "ernie-3.0-nano-zh",
    "ernie-3.0-tiny-base-v2-zh",
    "ernie-3.0-tiny-medium-v2-zh",
    "ernie-3.0-tiny-micro-v2-zh",
    "ernie-3.0-tiny-mini-v2-zh",
    "ernie-3.0-tiny-nano-v2-zh",
    "ernie-3.0-tiny-pico-v2-zh",
    "ernie-2.0-large-en",
    "ernie-2.0-base-en",
    "ernie-3.0-tiny-mini-v2-en",
    "ernie-m-base",
    "ernie-m-large",
]


# yapf: disable
@dataclass
class DataArguments:
    """
    数据集相关的参数
    """
    max_length: int = field(default=128, metadata={"help": "Maximum number of tokens for the model."})
    early_stopping: bool = field(default=False, metadata={"help": "Whether apply early stopping strategy."})
    early_stopping_patience: int = field(default=4, metadata={"help": "Stop training when the specified metric worsens for early_stopping_patience evaluation calls"})
    debug: bool = field(default=False, metadata={"help": "Whether choose debug mode."})
    train_path: str = field(default='./data/train.txt', metadata={"help": "Train dataset file path."})
    dev_path: str = field(default='./data/dev.txt', metadata={"help": "Dev dataset file path."})
    test_path: str = field(default='./data/dev.txt', metadata={"help": "Test dataset file path."})
    label_path: str = field(default='./data/label.txt', metadata={"help": "Label file path."})
    bad_case_path: str = field(default='./data/bad_case.txt', metadata={"help": "Bad case file path."})


@dataclass
class ModelArguments:
    """
    模型相关的参数
    """
    model_name_or_path: str = field(default="ernie-3.0-tiny-medium-v2-zh", metadata={"help": "Build-in pretrained model name or the path to local model."})
    export_model_dir: Optional[str] = field(default=None, metadata={"help": "Path to directory to store the exported inference model."})
# yapf: enable


def main():
    """
    Training a binary or multi classification model
    训练流程
    """

    parser = PdArgumentParser((ModelArguments, DataArguments, CompressionArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.do_compress:
        training_args.strategy = "dynabert"
    if training_args.do_train or training_args.do_compress:
        training_args.print_config(model_args, "Model")
        training_args.print_config(data_args, "Data")
    paddle.set_device(training_args.device)

    # Define id2label
    id2label = {}
    label2id = {}
    with open(data_args.label_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            l = line.strip()
            id2label[i] = l
            label2id[l] = i

    # Define model & tokenizer
    if os.path.isdir(model_args.model_name_or_path):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, label2id=label2id, id2label=id2label
        )
    elif model_args.model_name_or_path in SUPPORTED_MODELS:
        # 需要定义下类别数
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, num_classes=len(label2id), label2id=label2id, id2label=id2label
        )
    else:
        raise ValueError(
            f"{model_args.model_name_or_path} is not a supported model type. Either use a local model path or select a model from {SUPPORTED_MODELS}"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # load and preprocess dataset
    train_ds = load_dataset(read_local_dataset, path=data_args.train_path, label2id=label2id, lazy=False)
    dev_ds = load_dataset(read_local_dataset, path=data_args.dev_path, label2id=label2id, lazy=False)
    trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_length=data_args.max_length)
    train_ds = train_ds.map(trans_func)
    dev_ds = dev_ds.map(trans_func)

    # debug 模式会加载 test 数据集
    if data_args.debug:
        test_ds = load_dataset(read_local_dataset, path=data_args.test_path, label2id=label2id, lazy=False)
        test_ds = test_ds.map(trans_func)

    # Define the metric function.
    def compute_metrics(eval_preds):
        """
        计算指标
        """
        pred_ids = np.argmax(eval_preds.predictions, axis=-1)
        metrics = {}
        metrics["accuracy"] = accuracy_score(y_true=eval_preds.label_ids, y_pred=pred_ids)
        for average in ["micro", "macro"]:
            # 原来能一次性计算所有
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=eval_preds.label_ids, y_pred=pred_ids, average=average
            )
            metrics[f"{average}_precision"] = precision
            metrics[f"{average}_recall"] = recall
            metrics[f"{average}_f1"] = f1
        return metrics

    def compute_metrics_debug(eval_preds):
        """
        debug 模式下会计算分类报告
        """
        pred_ids = np.argmax(eval_preds.predictions, axis=-1)
        metrics = classification_report(eval_preds.label_ids, pred_ids, output_dict=True)
        return metrics

    # Define the early-stopping callback.
    if data_args.early_stopping:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=data_args.early_stopping_patience)]
    else:
        callbacks = None

    # Define Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        criterion=paddle.nn.loss.CrossEntropyLoss(),
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        callbacks=callbacks,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics_debug if data_args.debug else compute_metrics,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        # 这里删除了所有的检查点
        for checkpoint_path in Path(training_args.output_dir).glob("checkpoint-*"):
            shutil.rmtree(checkpoint_path)

    # Evaluate and tests model
    if training_args.do_eval:
        if data_args.debug:
            output = trainer.predict(test_ds)
            log_metrics_debug(output, id2label, test_ds, data_args.bad_case_path)
        else:
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)

    # export inference model 导出静态模型
    if training_args.do_export:
        if model.init_config["init_class"] in ["ErnieMForSequenceClassification"]:
            input_spec = [paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids")]
        else:
            input_spec = [
                paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
                paddle.static.InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            ]
        if model_args.export_model_dir is None:
            model_args.export_model_dir = os.path.join(training_args.output_dir, "export")
        export_model(model=trainer.model, input_spec=input_spec, path=model_args.export_model_dir)
        # 保存分词器和标签
        tokenizer.save_pretrained(model_args.export_model_dir)
        id2label_file = os.path.join(model_args.export_model_dir, "id2label.json")
        with open(id2label_file, "w", encoding="utf-8") as f:
            json.dump(id2label, f, ensure_ascii=False)
            logger.info(f"id2label file saved in {id2label_file}")

    # compress 模型裁剪
    if training_args.do_compress:
        trainer.compress()
        for width_mult in training_args.width_mult_list:
            pruned_infer_model_dir = os.path.join(training_args.output_dir, "width_mult_" + str(round(width_mult, 2)))
            tokenizer.save_pretrained(pruned_infer_model_dir)
            id2label_file = os.path.join(pruned_infer_model_dir, "id2label.json")
            with open(id2label_file, "w", encoding="utf-8") as f:
                json.dump(id2label, f, ensure_ascii=False)
                logger.info(f"id2label file saved in {id2label_file}")

    # 这里删除了所有的训练日志. 这个还是要保留下来的
    # for path in Path(training_args.output_dir).glob("runs"):
    #     shutil.rmtree(path)


if __name__ == "__main__":
    main()
