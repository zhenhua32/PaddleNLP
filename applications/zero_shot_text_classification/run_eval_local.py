import os
import subprocess
import sys

# 尝试覆盖掉已经安装的 paddlenlp
# sys.path.insert(0, r"G:\code\github\PaddleNLP")
os.environ["PYTHONPATH"] = r"G:\code\github\PaddleNLP"

cur_dir = os.path.dirname(__file__)
# model_dir = "utc-base"
data_dir = r"G:\dataset\text_classify\baidu_utc_medical\raw"
train_output_dir = r"G:\code\github\PaddleNLP\outputs"
# 这里需要的是训练输出目录, 而不是模型导出目录
model_dir = os.path.join(train_output_dir, "")
test_path = os.path.join(data_dir, "test.txt")
output_dir = os.path.join(train_output_dir, "eval_results")

cmd_list = [
    sys.executable,
    os.path.join(cur_dir, "run_eval.py"),
    "--device=gpu",
    # 如果将 model_path 注释掉, 就是 zero-shot
    f"--model_path={model_dir}",
    f"--test_path={test_path}",
    "--per_device_eval_batch_size=2",
    "--max_seq_len=512",
    f"--output_dir={output_dir}",
]
print(cmd_list)
subprocess.run(cmd_list)
