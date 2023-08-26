"""
学习率搜索, 手动搜索
"""

import os
import subprocess
import sys

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


def train_one_model(model_dir: str, output_dir: str, lr: str):
    """
    使用不同的预训练模型进行训练
    """
    bad_case_file = os.path.join(output_dir, "bad_case.txt")

    cmd_list = [
        sys.executable,
        os.path.join(cur_dir, "../multi_class/train.py"),
        "--do_train",
        "--do_eval",
        "--do_export",
        "--debug",
        f"--model_name_or_path={model_dir}",
        f"--output_dir={output_dir}",
        f"--train_path={train_file}",
        f"--dev_path={dev_file}",
        f"--test_path={test_file}",
        f"--label_path={label_file}",
        f"--bad_case_path={bad_case_file}",
        "--device=gpu",
        "--num_train_epochs=100",
        "--early_stopping=True",
        "--early_stopping_patience=5",
        f"--learning_rate={lr}",
        "--max_length=64",
        "--per_device_eval_batch_size=256",
        "--per_device_train_batch_size=256",
        "--metric_for_best_model=accuracy",
        "--load_best_model_at_end",
        "--logging_steps=5",
        "--evaluation_strategy=epoch",
        "--save_strategy=epoch",
        "--save_total_limit=1",
        "--fp16",
    ]
    print(cmd_list)
    subprocess.run(cmd_list)


def main():
    model = "ernie-3.0-xbase-zh"
    model = "ernie-3.0-base-zh"

    search_space = {
        "learning_rate": ["1e-4", "1e-5", "3e-5", "5e-5", "7e-5", "1e-6"],
    }

    base_output_dir = r"G:\code\github\PaddleNLP\outputs\tnews_public"
    os.makedirs(base_output_dir, exist_ok=True)

    for lr in search_space["learning_rate"]:
        print(f"=========={model}, lr={lr}==========")
        model_dir = model
        output_dir = os.path.join(base_output_dir, f"{model}_lr={lr}")
        train_one_model(model_dir, output_dir, lr)


if __name__ == "__main__":
    main()
