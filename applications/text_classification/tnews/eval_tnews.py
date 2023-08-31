import json
import os
import subprocess
import sys

import pandas as pd


# 设置缓存目录
os.environ["PPNLP_HOME"] = r"G:\code\pretrain_model_dir\_paddlenlp"

# 尝试覆盖掉已经安装的 paddlenlp
sys.path.insert(0, r"G:\code\github\PaddleNLP")
os.environ["PYTHONPATH"] = r"G:\code\github\PaddleNLP"

cur_dir = os.path.dirname(__file__)
data_dir = r"G:\dataset\text_classify\tnews_public\paddlenlp"

# 输入的训练文件是固定的
train_file = os.path.join(data_dir, "train.txt")
dev_file = os.path.join(data_dir, "dev.txt")
test_file = os.path.join(data_dir, "dev.txt")  # 假的, 没这个文件, 临时代替下
label_file = os.path.join(data_dir, "label.txt")


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
    "ernie-m-base",
    "ernie-m-large",
]
base_output_dir = r"G:\code\github\PaddleNLP\outputs\tnews_public"


def eval_one_model(model_dir: str):
    bad_case_file = os.path.join(model_dir, "bad_case.txt")

    cmd_list = [
        sys.executable,
        os.path.join(cur_dir, "../multi_class/train.py"),
        "--do_eval",
        "--debug=True",
        f"--model_name_or_path={model_dir}",
        f"--output_dir={model_dir}",
        "--per_device_eval_batch_size=64",
        "--max_length=64",
        f"--train_path={train_file}",
        f"--dev_path={dev_file}",
        f"--test_path={test_file}",
        f"--label_path={label_file}",
        f"--bad_case_path={bad_case_file}",
    ]
    print(cmd_list)
    subprocess.run(cmd_list)


def get_all_models():
    """
    获取所有待评估的模型目录和模型名字
    """
    for name in os.listdir(base_output_dir):
        model_dir = os.path.join(base_output_dir, name)
        if os.path.isdir(model_dir) and name.startswith("ernie-3.0-base-zh_lr"):
            yield model_dir, name


def extract_metric(output_excel="metric.xlsx"):
    """
    从所有的模型中提取指标, 并输出成 excel 文件
    """
    data_list = []

    for model_dir, model in get_all_models():
        bad_case_file = os.path.join(model_dir, "bad_case.txt")

        with open(bad_case_file, "r", encoding="utf8") as f:
            line = f.readline()
            line = json.loads(line)

        data_list.append(
            {
                "model": model,
                "precision": line["test_macro avg"]["precision"],
                "recall": line["test_macro avg"]["recall"],
                "f1": line["test_macro avg"]["f1-score"],
                "support": line["test_macro avg"]["support"],
                "samples_per_second": line["test_samples_per_second"],
            }
        )

    df = pd.DataFrame(data_list)
    df.to_excel(os.path.join(base_output_dir, output_excel))


def main():
    for model_dir, name in get_all_models():
        print("run eval on model: ", model_dir)
        eval_one_model(model_dir)
    extract_metric(output_excel="metric.xlsx")


if __name__ == "__main__":
    main()
