# GPU / LLM Infra Lab

面向 **GPU 集群调度、网络通信、大模型训练工程** 实习/校招的 **可运行** 小项目：在单卡（例如笔记本 **RTX 2060 6GB**）上也能完整跑通训练、基准测试与简单量化推理对比。代码刻意保持可读，便于你在面试里讲清楚 **混合精度、显存换计算、分布式扩展思路**。

## 本机实测记录（可复现）

以下数据在 **2026-03-23** 于本仓库目录下 **真实跑通**（命令见各小节）。硬件信息来自 `nvidia-smi`，设备为 **RTX 2060 (6GB)**，驱动 **591.59**。

| 项目 | 结果 |
|------|------|
| GPU（查询） | NVIDIA GeForce RTX 2060，驱动 **591.59**，显存 **6144 MiB** |
| Python / PyTorch | **3.13.8** / CUDA 可用（`bench_gpu` 已识别 `NVIDIA GeForce RTX 2060`） |
| `bench_gpu`（GPU） | `4096×4096` FP16：**~7.49 ms** / **~18.36 TFLOP/s**；FP32：**~27.13 ms** / **~5.07 TFLOP/s**；MLP fp32：**~7.18 ms**，AMP fp16：**~2.67 ms** |
| `bench_collectives`（单进程） | 约 **50M** float 元素本地归约模拟：**~66.8 ms/iter**（通信脚本单进程 baseline） |
| `scheduler_sim --gpus 4` | FIFO makespan **2.11 h**，按显存贪心 **2.00 h** |
| `train`（`--max-iters 120`，GPU） | iter **0** loss **3.84**；iter **50** loss **~1.96**；iter **100** loss **~1.18**；吞吐约 **65k~67k tokens/s** |
| `export_onnx` | **`artifacts/tiny_gpt.onnx`**，`onnx.checker` **通过**；文件约 **12.4 MiB**（`artifacts/` 已 gitignore，本地生成） |
| `infer_quant`（CPU，`--steps 100`） | FP32 **~5.73 ms/forward**；动态量化 Linear **~7.05 ms/forward**（小模型量化开销可见） |

一键复现（项目根目录，已 `pip install -e . onnx`）：

```bash
python -u -m gpu_llm_infra_lab.train --max-iters 120 --config configs/default.yaml --out_dir runs/gpu_run
python -m gpu_llm_infra_lab.export_onnx --ckpt runs/gpu_run/ckpt_final.pt --out artifacts/tiny_gpt.onnx
python -m gpu_llm_infra_lab.infer_quant --ckpt runs/gpu_run/ckpt_final.pt --steps 100
python -m gpu_llm_infra_lab.bench_gpu
python -m gpu_llm_infra_lab.bench_collectives
python -m gpu_llm_infra_lab.scheduler_sim --gpus 4
```

## 和岗位 JD 的对应关系

| JD 方向 | 本仓库里对应内容 |
|--------|------------------|
| GPU 利用率 / 混合精度 / 量化 | `train.py` 中 **AMP (FP16)**、`infer_quant.py` 中 **动态量化 (CPU)**、`bench_gpu.py` 中 **算子与 AMP 对比**；**ONNX → TensorRT** 见下文 |
| 网络 / 集合通信 | `bench_collectives.py`：**all_reduce** 延迟（单机多卡可用 `torchrun`；单卡给 baseline + 说明 NCCL/MPI 场景） |
| 大模型训练框架 / 显存优化 | **梯度检查点** (`torch.utils.checkpoint`)、**梯度累积**、**AdamW + warmup + cosine**；模型为精简 **Decoder-only Transformer**（字符级 LM） |
| 调度与资源 | `scheduler_sim.py`：极简 **FIFO vs 按显存贪心** 的 makespan 对比（讨论利用率与尾延迟） |

## 环境

- Python **3.10+**
- PyTorch **CUDA 版**（与显卡驱动匹配），安装见 [pytorch.org](https://pytorch.org/get-started/locally/)

```bash
cd gpu-llm-infra-lab
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
# 笔记本 2060 请安装带 CUDA 的 PyTorch（示例，按官网命令为准）:
# pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .
pip install onnx   # 导出 ONNX 时需要
```

## ONNX 与 TensorRT（笔记本 RTX 2060 / Turing）

**2060 Laptop** 为 **Turing、算力 7.5**，与本仓库默认 **FP16** 推理路径一致。CUDA / TensorRT 请从 NVIDIA 官网按驱动版本自选安装；这里只固定命令形状。

1. 训练得到 `runs/.../ckpt_final.pt` 后导出 ONNX（项目根目录）：

```bash
python -m gpu_llm_infra_lab.export_onnx --ckpt runs/exp1/ckpt_final.pt --out artifacts/tiny_gpt.onnx
# 若 trtexec 对 dynamic shape 报错，可先试固定形状：
python -m gpu_llm_infra_lab.export_onnx --ckpt runs/exp1/ckpt_final.pt --out artifacts/tiny_gpt_static.onnx --static
```

- 输入：**`input_ids`**，`int64`，形状 `[batch, seq]`，`seq ≤ config 里的 block_size`。
- 输出：**`logits`**，`float32`，`[batch, seq, vocab_size]`。
- 导出使用 **`torch.onnx.export(..., dynamo=False)`**（TorchScript 路径），避免默认 Dynamo 导出对 `onnxscript` 的依赖，和 **TensorRT / trtexec** 更常见。

2. 使用 **TensorRT** 自带的 `trtexec` 生成 engine（示例，路径按你本机 TensorRT 安装修改）：

```bash
trtexec --onnx=artifacts/tiny_gpt.onnx --saveEngine=artifacts/tiny_gpt_fp16.plan --fp16 --memPoolSize=workspace:1024
```

`INT8` 需要校准数据与额外 flag；小 demo 用 **FP16** 即可在面试里讲清「图优化、kernel 融合、与 PyTorch eager 对比」。

3. **Triton Inference Server**：将 ONNX 或 TensorRT plan 放入 model repository（`config.pbtxt` 里声明输入输出名与 dtype），用 `onnxruntime` 或 `tensorrt` backend 加载；官方文档：<https://github.com/triton-inference-server/server>。

## 训练（2060 默认配置在 `configs/default.yaml`）

在项目根目录执行（保证相对路径 `configs/`、`data/` 正确）：

```bash
python -m gpu_llm_infra_lab.train --config configs/default.yaml --out_dir runs/exp1
```

快速冒烟（少步数）：

```bash
python -m gpu_llm_infra_lab.train --max-iters 30 --config configs/default.yaml --out_dir runs/smoke
```

可选：Linux + 多卡时（本机需正确安装 NCCL 环境）：

```bash
torchrun --standalone --nproc_per_node=2 -m gpu_llm_infra_lab.train --config configs/default.yaml --out_dir runs/ddp
```

Windows 单卡无需 `torchrun`；多卡/分布式以你本机 PyTorch 文档为准。

## 其它脚本

```bash
python -m gpu_llm_infra_lab.bench_gpu
python -m gpu_llm_infra_lab.bench_collectives
python -m gpu_llm_infra_lab.scheduler_sim --gpus 4
python -m gpu_llm_infra_lab.infer_quant --ckpt runs/exp1/ckpt_final.pt
python -m gpu_llm_infra_lab.export_onnx --ckpt runs/exp1/ckpt_final.pt --out artifacts/tiny_gpt.onnx
```

说明：在 **极小模型 + CPU** 上，动态量化有时会比 FP32 慢（量化/反量化开销）；在 **更大模型或 x86 INT8 内核更友好** 的场景下通常更划算。面试里可以主动讲清这个 trade-off。

## 你可以接着做的「加分」方向（与 JD 一致）

1. **推理**：在 README「ONNX 与 TensorRT」基础上，把 **trtexec 延迟数字** 或 **Triton 目录结构** 截图放进仓库。
2. **数据**：把 `data/sample_corpus.txt` 换成公开小语料，报告 **loss / tokens/s** 曲线截图放 README。
3. **通信**：在实验室双机或多卡上复现 `bench_collectives`，对比 **NVLink / PCIe / RDMA**（有环境再写结论）。
4. **训练**：接入 **DeepSpeed ZeRO-1** 或 **FSDP** 只包一层 wrapper，对比显存与吞吐（说明工程化能力即可）。

## 上传到 GitHub

本机未检测到 **GitHub CLI (`gh`)**，需手动在网页创建空仓库后推送（将 `YOUR_USER` 换成你的用户名）：

```bash
cd gpu-llm-infra-lab
git init
git add .
git commit -m "Initial commit: GPU/LLM infra lab with benchmarks"
git branch -M main
git remote add origin https://github.com/YOUR_USER/gpu-llm-infra-lab.git
git push -u origin main
```

若已安装 `gh` 且已登录，可改为：`gh repo create gpu-llm-infra-lab --public --source=. --remote=origin --push`。

## License

见 [LICENSE](LICENSE)（MIT）。
