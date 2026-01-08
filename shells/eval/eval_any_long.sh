#!/bin/bash

# 指定脚本的解释器为 bash

# 创建一个名为 debug_logs 的文件夹，用于存放运行过程中的调试日志或中间输出
# 如果文件夹已存在，mkdir 不会报错（除非没权限）
mkdir debug_logs

# 检查环境变量 NO_INFER 是否不等于 1
# 这提供了一个开关：如果你只想重新计算分数而不想跑模型推理，
# 可以在运行前设置 export NO_INFER=1
if [ "$NO_INFER" != "1" ]; then

    # 运行预测脚本 (pred.py)
    # --model my_model: 指定使用的模型名称
    # --cfg "$1": 传入第一个参数，即配置文件路径 (如 configs/example.json)
    # --ws "$2": 传入第二个参数，通常指 World Size 或 GPU 数量 (你传入的 1)
    python benchmark/long_bench/pred.py --model my_model --cfg "$1" --ws "$2"
    
fi # 结束 if 判断

# 运行评估脚本 (eval.py)
# 在模型生成预测结果后，对比预测内容与标准答案，计算 ROUGE、F1 等指标
# 它使用同样的配置文件 "$1" 来定位预测结果的存放路径
python benchmark/long_bench/eval.py --model my_model --cfg "$1"