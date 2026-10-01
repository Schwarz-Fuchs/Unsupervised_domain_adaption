# 无监督领域自适应（Unsupervised Domain Adaptation）

本仓库整理了图像分类任务中的无监督领域自适应实验代码，包含两条不同方向的研究/实验流程：

1. **输入空间适配**：通过 CycleGAN 及其改造版本在图像空间转换源域/目标域外观，并结合分类器或 ADDA 做下游适配。
2. **输出空间适配**：使用源域监督训练与目标域伪标签自训练（ST、CBST、CRST），以及 CLIP/Tip-Adapter 风格的 few-shot 适配。

两个子项目的代码、数据格式和训练入口相互独立。建议先选择一个子项目复现，不要把两边的 checkpoint 或数据目录混用。

## 仓库结构

```text
Unsupervised_domain_adaption/
├── UDA_inputspace/                 # CycleGAN/CyCADA/ADDA 输入空间适配
│   ├── train_cyclegan.py           # CycleGAN 系列训练入口
│   ├── test_cyclegan.py            # 图像转换测试与 HTML 结果页
│   ├── train_resnet_18.py          # ResNet-18 分类器训练
│   ├── train_adda_net.py           # ADDA 特征对齐训练
│   ├── cycle_gan_generate.py       # 用生成器转换数据集图像
│   ├── UDA_test.py                 # 多种分类 checkpoint 的目标域评估示例
│   ├── UDA_test_batchjob.py        # 批量运行实验/任务的脚本
│   ├── options/                    # CycleGAN 的基础、训练和测试参数
│   ├── models/                     # CycleGAN、语义/频域变体、ResNet/ADDA 模型
│   ├── datasets/                   # CycleGAN 数据加载及数据集准备脚本
│   ├── util/、cycada/              # 可视化、图像缓存和 ADDA 辅助工具
│   ├── docs/                       # CycleGAN 代码说明、数据和容器提示
│   └── tc2run_*.sh                 # 集群作业提交脚本示例
└── UDA_outputspace/                # 自训练与 CLIP Adapter 输出空间适配
    ├── ST.py                       # ST、CBST、CRST 训练入口
    ├── Adapter.py                  # CLIP/Tip-Adapter few-shot 适配
    ├── ST_batch_run.py             # 批量运行多组自训练实验
    ├── original_datasets/          # 数据下载来源说明
    ├── offical codes/              # 参考的官方/原始代码片段
    └── README.md                   # 输出空间适配说明与实验注意事项
```

## 环境

子项目文档记录了 Python 3.10 和 PyTorch 2.8 的实验环境，但仓库根目录没有统一的依赖清单。请分别为 `UDA_inputspace` 和 `UDA_outputspace` 建立环境，根据脚本实际导入内容安装 PyTorch、torchvision、NumPy、Pillow、scikit-learn、tqdm、OpenCV、pandas、scipy 等依赖。`UDA_outputspace/Adapter.py` 还依赖 CLIP Python 包及其预训练权重。

CycleGAN 代码基于 `junyanz/pytorch-CycleGAN-and-pix2pix` 改造，数据加载和训练参数沿用该项目的结构。新机器上请根据 CUDA 驱动选择匹配的 PyTorch wheel，首次运行前确认目标机器能访问预训练权重下载地址，或事先缓存相应模型。

请勿直接假设上述所有包必须安装；优先从目标实验入口导入内容确认实际依赖，并在验证后保存精简的 requirements 文件。训练主要面向 NVIDIA GPU；输入空间项目的一些分类/ADDA 脚本还直接使用 CUDA，CPU 支持并不统一。

## 数据集格式

### 输入空间适配：图像转换数据

CycleGAN 常用的非配对数据结构如下：

```text
datasets/<数据集名>/
├── trainA/      # 训练源域 A 图像
├── trainB/      # 训练目标域 B 图像
├── testA/       # 测试源域图像
└── testB/        # 测试目标域图像
```

`UDA_inputspace/datasets/generate_UDA_dataset.py` 可以从两个按类别组织的图像目录递归收集图片，并随机划分、复制成 `trainA/testA/trainB/testB`。运行前修改脚本末尾的 `source_a`、`source_b`、`output_root` 和 `split_ratio`。当前实现没有设置随机种子，因此每次重新生成的拆分可能不同；如需严格复现，请固定随机种子并保存划分清单。

对于需要类别语义监督的变体（如 `cycle_gan_semantic`），除了 CycleGAN 图像数据，还要提供 `--original_data_dir` 指向按类别名组织的原始源域图像目录，并正确设置 `--out_feature_num`。该流程需要先训练源域分类器，再将分类器权重按代码要求放入 CycleGAN 实验 checkpoint 目录并命名为 `latest_net_CLS.pth`。

### 输出空间适配：分类数据

ST/CBST/CRST 和 Adapter 使用 PyTorch `ImageFolder` 风格的分类目录：

```text
<domain>/
├── class_1/
│   ├── image_001.jpg
│   └── ...
├── class_2/
└── ...
```

源域 `--src_path` 和目标域 `--tgt_path` 应有相同类别集合，且类别目录名称/索引映射必须一致。目标域训练阶段使用图像而不使用目标标签；若保留带标签测试集，只能用于最终评估，不能参与伪标签训练。Office-31、Office-Home、PACS 等数据集的下载来源记录在 `UDA_outputspace/original_datasets/datasource.txt`。

## 输入空间适配流程

### CycleGAN 图像翻译

从 `UDA_inputspace/` 目录运行训练入口，命令参数由 `options/base_options.py` 和 `options/train_options.py` 定义。基本示例：

```bash
cd UDA_inputspace
python train_cyclegan.py \
  --dataroot ./datasets/photo2sketch \
  --name photo2sketch_cyclegan \
  --model cycle_gan \
  --dataset_mode unaligned \
  --load_size 150 \
  --crop_size 128 \
  --batch_size 16 \
  --n_epochs 200 \
  --n_epochs_decay 0
```

仓库还定义了 `fg_cycle_gan` 频域相关模型和 `cycle_gan_semantic` 语义分类引导模型；具体可用损失参数可查看 `models/fg_cycle_gan_model.py`、`models/cycle_gan_semantic_model.py`。使用语义模型时还须核对 `dataset_mode`、分类器权重、源域类别数及数据格式。checkpoint 和网页预览保存在 `--checkpoints_dir`、`--name` 对应的实验目录。

测试示例（需先完成训练，或准备兼容的预训练 checkpoint）：

```bash
python test_cyclegan.py \
  --dataroot ./datasets/photo2sketch \
  --name photo2sketch_cyclegan \
  --model cycle_gan \
  --phase test \
  --no_dropout
```

结果默认写入 `./results/`，可用 `--results_dir` 指定目录。只转换单个域的输入时，可使用 `--model test` 并将 `--dataroot` 指向单个图像目录，具体参数见 `options/test_options.py`。

### CyCADA/分类与 ADDA 阶段

项目中面向语义适配的一般步骤为：

1. 使用 `train_resnet_18.py` 在有标签源域训练分类网络，并确认分类类别数与数据集一致。
2. 将该分类器权重放到相应的 CycleGAN 实验目录，按模型代码约定保存为 `latest_net_CLS.pth`。
3. 使用 `train_cyclegan.py --model cycle_gan_semantic` 训练图像翻译/语义损失模型。数据集类型、`--original_data_dir` 和 `--out_feature_num` 应与数据集匹配。
4. 使用 `cycle_gan_generate.py` 生成转换后的图像。脚本末尾的源目录、生成器 checkpoint 和目标目录目前是写死的，使用前需编辑。
5. 运行 `train_adda_net.py` 或相应分类训练流程进行特征域适配，并用 `UDA_test.py` 检查目标域分类结果。训练数据路径和 checkpoint 路径需在脚本配置区按实际实验修改。

`tc2run_cada.sh`、`tc2run_sec.sh` 是集群作业脚本范例。它们包含特定集群的模块/Conda 环境设置，不适用于普通本地终端；提交前请更新环境、数据路径、GPU 资源和日志设置。

## 输出空间适配流程

### ST、CBST 和 CRST 自训练

从 `UDA_outputspace/` 目录启动。程序首先在源域训练或加载分类模型，再按置信度从无标签目标域逐轮生成伪标签并训练。`--method` 可选 `ST`、`CBST` 或 `CRST`：

- `ST` 使用全局置信度阈值，逐轮扩大目标域伪标签采样比例。
- `CBST` 按类别选择伪标签，缓解类别不平衡。
- `CRST` 在 CBST 基础上提供标签/模型正则化参数 `alpha`、`beta`、`gamma`、`delta`。

Office-31 示例：

```bash
cd UDA_outputspace
python ST.py \
  --arch resnet50 \
  --method ST \
  --src_path ./original_datasets/office_31/amazon \
  --tgt_path ./original_datasets/office_31/webcam \
  --num_classes 31 \
  --apply_aug \
  --num_rounds 20 \
  --epochs_per_round 3 \
  --init_portion 0.2 \
  --portion_step 0.05 \
  --max_portion 0.8 \
  --lr 2e-4 \
  --save_dir ./checkpoints/amazon_to_webcam_ST
```

将 `--method ST` 改成 `CBST` 可运行类别平衡伪标签版本。运行 CRST 时可按消融实验设置正则项，例如：

```bash
python ST.py \
  --arch resnet50 \
  --method CRST \
  --src_path ./original_datasets/office_31/amazon \
  --tgt_path ./original_datasets/office_31/webcam \
  --num_classes 31 \
  --apply_aug \
  --alpha 0.05 --beta 0.001 --gamma 0 --delta 0 \
  --save_dir ./checkpoints/amazon_to_webcam_CRST
```

主要参数见 `ST.py` 末尾的 ArgumentParser。`--kc_value` 支持 `conf` 和 `prob`；当前项目说明指出 `prob`（软标签）模式尚未充分测试，应将其结果视为实验性。`ST_batch_run.py` 中也写有多组 Office-Home/PACS 实验配置，可按需编辑后批量执行。

### CLIP Adapter / Tip-Adapter

`UDA_outputspace/Adapter.py` 使用 CLIP 特征和缓存分类器，包含零样本、目标域 few-shot 及 Tip-Adapter-F 训练/评估流程。PACS 示例：

```bash
cd UDA_outputspace
python Adapter.py \
  --source_path ./original_datasets/PACS/photo \
  --target_path ./original_datasets/PACS/sketch \
  --dataset PACS \
  --backbone RN50 \
  --shot 3 \
  --batch_size 32 \
  --save_dir ./checkpoints/photo_to_sketch_Adapter
```

Adapter backbone、few-shot 数量、学习率、训练轮数、早停和输出路径均可通过脚本参数调整。代码中对 Adapter 最后线性层的激活做过本地改动，与参考实现存在差异；进行论文复现或横向比较时应记录这一实现差别。

## 主要文件职责

### `UDA_inputspace/`

| 文件/目录 | 用途 |
|---|---|
| `train_cyclegan.py`、`test_cyclegan.py` | CycleGAN 训练和图像翻译评估入口。 |
| `models/cycle_gan_model.py` | 基础 CycleGAN 模型。 |
| `models/fg_cycle_gan_model.py` | 频域相关 CycleGAN 改造。 |
| `models/cycle_gan_semantic_model.py` | 结合分类网络语义损失的 CycleGAN 变体。 |
| `train_resnet_18.py` | 训练下游 ResNet-18 分类器。 |
| `train_adda_net.py`、`models/adda_net.py`、`models/resnet18.py` | ADDA 特征域适配及网络。 |
| `cycle_gan_generate.py` | 批量运行生成器并写出转换后图像；路径在脚本内配置。 |
| `UDA_test.py`、`UDA_test_batchjob.py` | 分类 checkpoint 评估及批量实验。 |
| `options/` | CycleGAN 数据、模型、训练和测试参数。 |
| `datasets/`、`cycada/`、`util/` | 图像数据读取、数据转换、图像缓存和日志/可视化工具。 |
| `docs/` | CycleGAN 结构、数据准备、FAQ、技巧和 Docker 的参考说明。 |

### `UDA_outputspace/`

| 文件/目录 | 用途 |
|---|---|
| `ST.py` | ST/CBST/CRST 自训练主逻辑，包括伪标签选择、源域预热和目标域迭代。 |
| `Adapter.py` | CLIP zero-shot、few-shot 和 Tip-Adapter-F 实验。 |
| `ST_batch_run.py` | 多组方法和数据集配置的批量执行脚本。 |
| `original_datasets/datasource.txt` | 数据集下载链接/来源记录。 |
| `offical codes/` | 用作参考的原始/官方代码片段，不一定属于当前入口依赖。 |
| `README.md` | 该目录原有的方法说明、运行参数和实验局限。 |


## 已知限制

- `UDA_outputspace/README.md` 明确提醒：ST/CBST/CRST 部分逻辑是从分割任务参考实现迁移到分类任务的实验性实现，不等同于官方分类代码；软标签模式尚未充分测试。用于研究结论前需检查伪标签生成、阈值和损失实现，并做消融验证。
- `UDA_inputspace/datasets/generate_UDA_dataset.py` 的当前随机划分没有固定种子，重复运行可能产生不同 train/test 文件。

## 参考工作

- CycleGAN：Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017。
- CyCADA：Hoffman et al., *Cycada: Cycle-consistent adversarial domain adaptation*, ICML 2018。
- CRST：Zou et al., *Confidence Regularized Self-Training*, ICCV 2019。
