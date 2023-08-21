import os
import subprocess
import sys

# 设置缓存目录
os.environ["PPNLP_HOME"] = r"G:\code\pretrain_model_dir\_paddlenlp"

# 尝试覆盖掉已经安装的 paddlenlp
sys.path.insert(0, r"G:\code\github\PaddleNLP")
os.environ["PYTHONPATH"] = r"G:\code\github\PaddleNLP"

cur_dir = os.path.dirname(__file__)
model_dir = "ernie-3.0-tiny-medium-v2-zh"
data_dir = r"G:\dataset\text_classify\KUAKE_QIC"
output_dir = r"G:\code\github\PaddleNLP\outputs\KUAKE_QIC"

train_file = os.path.join(data_dir, "train.txt")
dev_file = os.path.join(data_dir, "dev.txt")
test_file = os.path.join(data_dir, "dev.txt")  # 假的, 没这个文件, 临时代替下
label_file = os.path.join(data_dir, "label.txt")
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
    "--learning_rate=3e-5",
    "--max_length=128",
    "--per_device_eval_batch_size=32",
    "--per_device_train_batch_size=32",
    "--metric_for_best_model=accuracy",
    "--load_best_model_at_end",
    "--logging_steps=5",
    "--evaluation_strategy=epoch",
    "--save_strategy=epoch",
    "--save_total_limit=1",
]
print(cmd_list)
subprocess.run(cmd_list)
