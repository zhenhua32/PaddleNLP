import os
import subprocess
import sys

# 尝试覆盖掉已经安装的 paddlenlp
sys.path.insert(0, r"G:\code\github\PaddleNLP")
os.environ["PYTHONPATH"] = r"G:\code\github\PaddleNLP"

cur_dir = os.path.dirname(__file__)
model_dir = "utc-base"
data_dir = r"G:\dataset\text_classify\tnews\paddlenlp"
output_dir = r"G:\code\github\PaddleNLP\outputs\tnews"

cmd_list = [
    sys.executable,
    os.path.join(cur_dir, "../run_train.py"),
    "--device=gpu",
    "--single_label=True",
    "--logging_steps=10",
    "--save_steps=100",
    "--eval_steps=100",
    "--seed=1000",
    f"--model_name_or_path={model_dir}",
    f"--output_dir={output_dir}",
    f"--dataset_path={data_dir}",
    "--max_seq_length=512",
    "--per_device_train_batch_size=16",
    "--per_device_eval_batch_size=16",
    "--gradient_accumulation_steps=1",
    "--num_train_epochs=100",
    "--learning_rate=1e-5",
    "--do_train",
    "--do_eval",
    "--do_export",
    f"--export_model_dir={output_dir}/export_model",
    "--overwrite_output_dir",
    "--disable_tqdm=True",
    "--metric_for_best_model=accuracy",
    "--load_best_model_at_end=True",
    "--save_total_limit=1",
    "--save_plm",
]
print(cmd_list)
subprocess.run(cmd_list)
