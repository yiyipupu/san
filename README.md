> 原始项目：[SAN](https://github.com/MendelXu/SAN) · [论文](https://arxiv.org/abs/2302.12242) · [原始 README 备份](README.upstream.md)

# SAN 环境配置过程与 4090 迁移

## 1. 当前结果（2026-11-12）

**RTX 3060 上 SAN ViT-L/14 已完成单图 GPU 前向，并在阻断 Internet socket 连接后重复成功。**

- 工作区：`/home/pupu/insPart/SAN`，部署分支 `deployment/san-cu118`。复制自 external/SAN，原目录保留。
- 官方 SAN 基线提交：`e19262d1a23992fc2cef934d6756b89ff50fe5c3`。
- Ubuntu 20.04、GCC 9.4、RTX 3060 12GB、驱动 550.90.07。
- 已有 CUDA Toolkit 11.8，路径 `/usr/local/cuda-11.8`；没有修改驱动或安装系统工具链。
- Conda 环境：`/home/pupu/miniconda3/envs/instructpart-san`，Python 3.10.22。
- torch 2.1.0+cu118，torchvision 0.16.0+cu118，Detectron2 0.6（本机 CUDA 编译），open_clip_torch 2.16.0，timm 0.6.13。
- Detectron2 v0.6 精确提交：`d1e04565d3bec8719335b88be9e9b961bf3ec464`。
- Git 提交身份 pupu / p91865122@gmail.com，仅在本仓库设置；没有远程推送。

验收结果：GPU 小计算含前向/反向通过；`pip check` 通过；SAN 导入通过；在线和离线模型前向均退出码0；无核心权重 missing/unmatched 警告。输入 fruit.jpg，文本 watermelon，短边512，输出477×634，前景36287像素，峰值已分配显存2323.7 MiB。两次结果一致，输出图片已人工查看。这仅证明部署流程可用，不代表分割精度或训练验证。

## 2. 存储与日志

硬盘挂载点 `/media/pupu/PUPU`，NTFS，宿主机可读写；先前受限执行环境显示只读不代表宿主实际状态。

| 内容 | 路径 | 实测空间 |
|---|---|---|
| Conda 运行环境 | `/home/pupu/miniconda3/envs/instructpart-san` | 约5.2 GiB |
| 权重、wheel、模型及pip缓存 | `/media/pupu/PUPU/insPart/SAN` | 约4.9 GiB |
| 本机pip缓存 | `/home/pupu/.cache/pip` | 约474 MiB |
| Conda共享包缓存（含base，非SAN独占） | `/home/pupu/miniconda3/pkgs` | 约1.2 GiB |

硬盘下 weights/ 保存 SAN-L，cache/clip/ 保存 OpenAI CLIP，wheels/ 保存 torch/torchvision。环境和编译产物在本机 ext4。运行中不要拔出硬盘。

```bash
tail -f /home/pupu/insPart/SAN/logs/progress.log
```

进度监控在配置期间每10秒更新，完成后停止。详细日志：dependencies-install.log、environment-install.log、san-forward.log、san-forward-offline.log；第一次失败保留 san-forward-initial-failure.log。日志和输出不纳入 Git；摘要与校验值在 deployment/validation-results.json 和 artifacts.json。

## 3. 实际安装过程与问题记录

1. 从 external/SAN 复制含历史的官方仓库至 SAN，创建部署分支；建立此文档并逐阶段提交。
2. 通过已配置的清华 Conda 源创建环境：`conda create -n instructpart-san python=3.10 pip -y`。
3. 安装 `pip<25`、`setuptools<70`、wheel、ninja，普通Python依赖使用 `https://pypi.tuna.tsinghua.edu.cn/simple`。
4. 探测国内wheel源：南京大学索引实际指向官方存储；阿里和交大实际托管wheel。16MiB测试交大约9.8秒、阿里约61秒。主体选择交大，最后少量慢速分段切换阿里续传。
5. SAN模型国内 hf-mirror 地址重定向回 Hugging Face，未提供独立加速；因此模型使用可连通的官方源。通过 deployment/download_file.py 分段续传，严格检查 HTTP 206、Content-Range、字节数，完成后核对SHA256。
6. 原清单未锁定SciPy等版本，导致NumPy 1.22.4兼容性回溯。固定 scipy 1.10.1、scikit-image 0.20.0、h5py 3.10.0、matplotlib 3.7.5；固定旧timm接口、HF hub 0.17.3、OpenCV 4.8.1.78。推理路径未使用mmcv，未安装；未安装torchaudio。
7. open_clip需要 protobuf<4，固定protobuf 3.20.3、sentencepiece 0.1.99；Detectron2固定iopath 0.1.9等依赖。所有最终精确版本见 deployment/requirements.lock.txt。
8. 从移动硬盘本地wheel安装torch及torchvision，先执行 deployment/probe_gpu.py，确认本环境CUDA计算正常。
9. 获取官方Detectron2 v0.6源码归档，在本机 `/tmp/detectron2-0.6` 使用 CUDA_HOME=/usr/local/cuda-11.8、MAX_JOBS=2、`pip install --no-build-isolation` 编译成功。
10. 首次SAN前向失败：OpenAI CLIP文件是TorchScript archive，直接传给pretrained文件路径接口会报 RecursiveScriptModule.items。修复为保留 pretrained=openai，增加 cache_dir=SAN_CLIP_CACHE，使用open_clip原生加载流程。没有转换或重复下载权重。
11. SAN-L前向成功，再由 deployment/offline_run.py 阻断 AF_INET/AF_INET6 socket 连接重复成功。模型及tokenizer资源齐全。
12. 导出环境时清除ROS的PYTHONPATH，避免pip freeze混入系统ROS包。运行环境脚本也会清除此变量。

## 4. 本机复现

```bash
cd /home/pupu/insPart/SAN
conda activate instructpart-san
source deployment/env.sh
python -m pip check
python deployment/probe_gpu.py
bash deployment/run_forward.sh
```

默认运行配置为 SAN-L，单卡、FP32、eval/no_grad、短边512。输出在 outputs/san_forward/：mask.png、overlay.png、result.json。可通过 SAN_IMAGE 指定图片；默认文本为watermelon，改任务文本时修改运行脚本。若显存不足可将短边依次降至384、256，必须记录改动。当前无需降尺寸。

离线验证命令：

```bash
python deployment/offline_run.py deployment/smoke_forward.py \
  --method san --repo "$SAN_ROOT" \
  --weights "$SAN_STORAGE/weights/san_vit_large_14.pth" \
  --image deployment/assets/fruit.jpg --text watermelon --short-side 512 \
  --output outputs/san_forward_offline
```

示例图片来自 microsoft/X-Decoder 官方 inference/images/fruit.jpg，随仓库保存，运行无需依赖外部X-Decoder工作区。

## 5. 迁移至 4090：重建环境后验收

1. 复制或clone本仓库及部署分支，保留Git提交记录；复制硬盘中的 weights/、cache/、wheels/。按 deployment/artifacts.json 核对文件。SAN、CLIP、torch SHA256与发布方一致；torchvision清单值为本机下载文件的迁移校验参考。
2. 检查目标机NVIDIA驱动、磁盘、CUDA 11.8 Toolkit。nvidia-smi显示的是驱动能力，不是已安装Toolkit版本。无需照搬本机驱动版本，只要满足运行时要求。
3. 通过 `conda env create -f deployment/environment.yml` 创建Python3.10环境并激活。若同名环境已存在，先检查，不自动覆盖。deployment/conda-linux64.explicit.txt 保存本机Conda精确包记录，供同平台审计。
4. 设置目标路径后运行重建脚本：

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export SAN_STORAGE=/实际权重存储位置/SAN
bash deployment/rebuild.sh
```

重建脚本先装已缓存torch/torchvision wheel，再装精确Python依赖清单，然后从固定Git提交重新编译Detectron2。**不要复制3060上编译的Detectron2 wheel或扩展到4090。** 保持目标GPU可见，编译会探测目标架构；4090为8.9。如离线重建，需要预先下载精确依赖清单中的所有wheel和Detectron2源码；现有缓存仅证明模型推理离线可用，不保证安装过程离线可用。

5. 修改 deployment/env.sh 的默认路径或设置 SAN_ROOT、SAN_STORAGE、SAN_MOUNT；SAN_MOUNT必须为实际存储挂载点。另设 SAN_PYTHON 指向目标环境Python，再运行 `bash deployment/run_forward.sh`。路径不一致时不可直接复制本机默认命令。
6. 目标机重复pip check、GPU probe、模型前向和离线前向；保留版本、日志、显存及输出。先完成单卡验收，再进行训练。

requirements.lock.txt 为实际成功环境的精确Python版本，刻意排除torch、torchvision、Detectron2，三者由重建脚本单独安装。pip-freeze.raw.txt保留原始安装来源，其中本机绝对路径不能直接用于迁移安装。

## 6. 4090训练准备（尚未实测）

当前未下载完整数据集、未执行训练。训练需按官方README准备COCO Stuff等图像和标注、匹配数据注册路径。官方配置默认batch=32、60000次迭代、640随机裁剪及AMP，不能直接视为单张4090已验证参数。

训练入口示例：

```bash
python train_net.py --config-file configs/san_clip_vit_large_res4_coco.yaml \
  --num-gpus 1 OUTPUT_DIR /训练输出位置 SOLVER.IMS_PER_BATCH 1
```

在实际训练时，应先以小批量确认loss有限、反向、可训练参数变化及checkpoint保存/恢复，再决定正式batch、学习率、周期与梯度累积策略。从SAN权重继续训练时明确设置MODEL.WEIGHTS；恢复中断训练使用Detectron2 resume机制保留优化器/调度器状态。默认DATASETS.TEST涉及多个数据集，只准备COCO时必须覆盖测试列表及评测策略。当前前向通过不构成训练或论文精度通过。
