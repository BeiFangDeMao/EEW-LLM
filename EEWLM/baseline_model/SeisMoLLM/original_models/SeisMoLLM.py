from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo.config
from einops import rearrange
import transformers.models.gpt2 as GPT2
from peft import get_peft_model, LoraConfig
from functools import partial
from ._factory import register_model

torch._dynamo.config.cache_size_limit = 1024


def lora_setting(target_modules, r=16, lora_alpha=16, lora_dropout=0.1, bias="lora_only"):
    return LoraConfig(target_modules=target_modules, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=bias)

GPT2_lora = lora_setting("all-linear")
GPT_file_path = '/ailab/user/wangxinghao/project/EQLLM/LLM/'  # the path of pre-trained model files

#->返回类型注解，下函数为 一维卷积层自动填充工具。 该操作可以自动计算数据的填充量，避免因卷积核过大而导致输出维度远小于输入维度。
def _auto_pad_1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    dim: int = -1,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Auto pad for conv layer.
    The output of conv-layer has the shape as `ceil(x.size(dim)/stride)`.
    Use this function to replace `padding='same'` which `torch.jit` and `torch.onnx` do not support.
    #Args，全程Arguments 集中说明函数中的输入参数信息，
    Args:
        x (torch.Tensor): N-dimensional tensor.
        input (Tensor): N-dimensional tensor
        kernel_size (int): Conv kernel size.
        stride (int): Conv stride.
        dim (int): Dimension to pad.
        padding_value (float): fill value.

    Raises: AssertionError: `kernel_size` is less than `stride`.

    Returns: torch.Tensor : padded tensor.
    """
    #assert 断言检查，可以进行一个判断，这里的判断是“卷积核大小 >= 步长”，这个检查的目的是为了填充数据。
    assert (
        kernel_size >= stride
    ), f"`kernel_size` must be greater than or equal to `stride`, got {kernel_size}, {stride}"
    pos_dim = dim if dim >= 0 else x.dim() + dim
    pds = (stride - (x.size(dim) % stride)) % stride + kernel_size - stride
    padding = (0, 0) * (x.dim() - pos_dim - 1) + (pds // 2, pds - pds // 2)
    padded_x = F.pad(x, padding, "constant", padding_value)
    return padded_x

#类：init 方法是初始化方法，表明该类的具体对象有哪些标签（比如人类 有名字 身高 体重等 标签）。def是这个类的行为，即这个函数的功能（比如人类的学习 吃饭 等行为）。
#定义了一个pytorch神经网络模块，该类是继承了nn.module类。其功能是对激活函数的输出进行缩放。
class ScaledActivation(nn.Module):
    def __init__(self, act_layer: nn.Module, scale_factor: float):
        super().__init__() #继承了父类nn.Module 初始化模块
        self.scale_factor = scale_factor  #缩放因子
        self.act = act_layer() #初始化激活函数实例

    def forward(self, x):
        return self.act(x) * self.scale_factor #定义一个前向传播的方法，先将输入的X传到Self.act()函数中，输出的结果会乘一个缩放因子。
#定义了一个一维卷积模块，整合了输入投影、自动填充、主卷积、归一化和激活函数等操作形成一个完整的特征提取单元，通常用于处理一维序列数据（如音频、时序信号等）。
#in_dim：输入特征通道数，out是输出。Kernel_size：主卷积层的卷积核大小。stride：主卷积层的步长。。act_layer：激活函数类。norm_layer：归一化层类。
#定义了四个子模块 self.in_proj（输出投影层）、self.conv（主卷积层）、self.norm（归一化层）和self.act（激活函数）。
#定义了一个前向传播的方法。先投影，然后用_auto_pad_1d函数进行填充，接着主卷积，归一化，然后激活函数，
#nn.Conv1d 是 PyTorch 中 torch.nn 模块提供的一维卷积层（1D Convolutional Layer），专门用于处理一维序列数据（如音频信号、时序数据、文本的词嵌入序列等），核心功能是提取序列中的局部特征。
# nn.Conv1d(
#     in_channels,    # 输入的通道数（如单声道音频为1，多特征时序数据可能为n）
#     out_channels,   # 输出的通道数（即卷积核的数量，每个核提取一种特征）
#     kernel_size,    # 卷积核的长度（如3表示每次处理序列中连续的3个元素）
#     stride=1,       # 卷积核的滑动步长（控制输出序列的长度，步长越大输出越短）
#     padding=0,      # 输入序列两端的填充长度（用于保持输出长度或控制边界特征）
#     dilation=1,     # 卷积核元素之间的间隔（用于扩大感受野，如 dilation=2 会跳过1个元素）
#     groups=1,       # 分组卷积的组数（用于减少参数，如 groups=in_channels 即为深度可分离卷积）
#     bias=True       # 是否添加偏置项（与输出特征相加的常数）
# )

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, act_layer, norm_layer):
        super().__init__()

        self.in_proj = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False
        ) #输入维度、输出维度、卷积块尺寸、bias=False(无偏置，对输入特征进行线性投影，在不改变通道数的前提下调整特征分布)


        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, 
                              stride=stride, bias=False) #输入维度、输出维度、卷积块尺寸、卷积步长、无偏
        self.norm = norm_layer(out_dim) #由传入的 norm_layer 初始化。输入参数为 out_dim（与主卷积的输出通道数一致）。通过归一化特征的均值和方差，稳定训练过程，加速收敛。。
        self.act = act_layer() #由传入的 act_layer 初始化。对归一化后的特征引入非线性变换，让模型能够拟合更复杂的函数关系。

    def forward(self, x):
        x = self.in_proj(x)
        x = _auto_pad_1d(x, self.conv.kernel_size[0], self.conv.stride[0])
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

#该处定义了一个多尺度的卷积模块。初始化多了多尺度分支数量，尺度步长（用于控制不同分支卷积核大小的差异，值越大，不同卷积核尺寸相差越大）。
# self.convs多尺度卷积分支。self.out_proj多尺度卷积特征融合层。self.norm最终归一化层
#并行多尺度特征提取 + 特征融合。设计多个不同尺寸的卷积块，并行计算特征，最后将每个卷积快得到的特征进行融合（torch.cat方法）然后输出一个提取特征后的向量。

class Multi_Scale_Conv_Block(nn.Module):
    def __init__(
        self, scale_num, scale_stride, in_dim, out_dim, kernel_size, stride, act_layer, norm_layer
    ):
        super().__init__() #继承了父类

        self.convs = nn.ModuleList(
            [
                ConvBlock(
                    in_dim,
                    out_dim,
                    kernel_size + int(scale_stride * scale),
                    stride,
                    act_layer,
                    norm_layer,
                )
                for scale in range(scale_num) #使用nn.ModuleList创建一个可包含多个模块的列表。通过循环range(scale_num)，为每个scale值创建一个ConvBlock模块。
            ]
        )

        self.out_proj = nn.Conv1d(
            in_channels=scale_num * out_dim, out_channels=out_dim, kernel_size=1, bias=False
        )
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        outs = list() # 每个分支独立处理输入x，得到不同尺度的特征
        for conv in self.convs:
            xi = conv(x) # xi 是单个分支的输出，形状为 (batch_size, out_dim, length)
            outs.append(xi)
        x = torch.cat(outs, dim=1)  # 在通道维度拼接所有分支的输出：(batch_size, scale_num*out_dim, length)。torch.cat 是 PyTorch 中用于拼接张量的函数，它能将多个形状相同的张量沿着指定维度（dim 参数）连接成一个新张量。
        x = self.out_proj(x) # 融合多尺度特征并调整通道数：(batch_size, out_dim, length)
        x = self.norm(x)  # 归一化
        return x


#定义了一个名为 LLM_Block 的自定义 PyTorch 模块，其核心功能是将LLM，这里具体使用 GPT-2）适配为序列特征处理模块，通过分块（patch）操作将输入序列转换为 LLM 可处理的格式，利用 LLM 的特征提取能力后再将结果转换回原始序列形状。
#支持预训练模型加载、参数冻结和 LoRA（Low-Rank Adaptation）参数高效微调，适用于将语言模型的能力迁移到其他序列数据（如音频、时序信号等）的任务中
#start_layer, end_layer。截取 GPT-2 模型中间层的起始和结束索引（只使用 GPT-2 从 start_layer 到 end_layer 的部分层，而非完整模型）。
#patch_size：输入序列的分块大小（类似图像处理中的 “patch”，将长序列分割为固定长度的子序列）。
#lora_config：LoRA 微调的配置参数（用于参数高效微调，减少需要训练的参数数量）。
#pretrain：布尔值，是否使用预训练的 GPT-2 模型（True 则加载预训练权重，False 则初始化随机权重模型）。
#freeze：布尔值，是否冻结 GPT-2 大部分参数（仅训练部分关键参数或 LoRA 参数）。

class LLM_Block(nn.Module):
    def __init__(self, start_layer, end_layer, patch_size, lora_config, pretrain=True, freeze=True):
        super(LLM_Block, self).__init__()
        
        self.pretrain = pretrain
        self.freeze = freeze
        self.lora_config = lora_config
        self.patch_size = patch_size

        # 加载或初始化 GPT-2 模型
        if pretrain:
            self.llm = GPT2.GPT2Model.from_pretrained(
                GPT_file_path+'GPT2', output_hidden_states=True, 
                vocab_size=0, ignore_mismatched_sizes=True
            )  # loads a pretrained GPT-2 small base model（加载预训练 GPT-2 模型（small 版本）。
            # 这里通过 vocab_size=0 和 ignore_mismatched_sizes=True 适配非文本输入（不需要词嵌入层，直接接收预计算的嵌入向量）。
        else:
            print("------------------no pretrain------------------")
            self.llm = GPT2.GPT2Model(GPT2.configuration_gpt2.GPT2Config(vocab_size=0))
        self.llm.h = self.llm.h[start_layer : end_layer] #h 属性是其所有 Transformer 层的列表，这里只保留了从 start_layer 到 end_layer 的层，减少模型复杂度或适配特定特征提取需求。
        
        # print("LLM blocks = {}".format(self.llm))
        # print(f"using LLM layers: {end_layer - start_layer}")
        # 若使用预训练模型且需要冻结参数：应用 LoRA 并仅解冻部分参数
        # get_peft_model 来自 PEFT（Parameter-Efficient Fine-Tuning）库，将 GPT-2 转换为支持 LoRA 的模型。通过在原始权重旁添加低秩矩阵，仅训练低秩参数，大幅减少训练成本，同时保持模型性能。
        # 冻结模式下：仅训练层归一化（稳定训练）、位置编码（适配新序列的位置信息）和 LoRA 新增参数，冻结 GPT-2 原始权重，实现高效微调。
        # 非冻结模式下：仅冻结词嵌入层（因输入不是文本 token，词嵌入层无用），其他参数均可训练。

        if self.freeze and self.pretrain:
            self.llm = get_peft_model(self.llm, self.lora_config)  # apply LoRA to finetune the base model（将模型转换为支持 LoRA 的版本）
            for name, param in self.llm.named_parameters():
                if "ln" in name or "wpe" in name or "lora" in name:
                    param.requires_grad = True # 解冻层归一化（ln）、位置编码（wpe）和 LoRA 参数
                else:
                    param.requires_grad = False # 冻结其他参数（如原始 Transformer 权重）
        else:
            # 不冻结或不使用预训练时：仅冻结词嵌入层（wte，因输入非文本，无需更新）
            # wte is "Word Token Embeddings", won't participate in producing loss
            # wte = Word Token Embeddings（词嵌入层）
            for name, param in self.llm.named_parameters():
                if "wte" in name:  
                    param.requires_grad = False

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size) # 步骤1：将输入序列分块（patch）
        x = rearrange(x, 'b c n p -> b n (c p)') # 步骤2：重排维度以适配 GPT-2 输入格式
        x = self.llm(inputs_embeds = x).last_hidden_state # 步骤3：通过 GPT-2 处理特征
        x = rearrange(x, 'b n (c p) -> b c (n p)', p=self.patch_size) # 步骤4：重排维度恢复原始序列形状
        return x


# 定义了一个名为 HeadDetectionPicking 的自定义 PyTorch 模块，中文可理解为 “检测与相位拾取头”，主要用于将神经网络的中间特征映射转换为最终的预测结果.
# 该模块的核心是通过渐进式上采样（逐步放大特征的序列长度）和卷积处理，将低分辨率的特征图恢复到与输入原始信号相同的长度，最终输出用于检测或相位拾取的预测结果。


class HeadDetectionPicking(nn.Module):
    """Head of detection and phase-picking."""

    def __init__(
        self,
        feature_channels, # 输入特征的通道数 承接网络上游的特征输出
        layer_channels,  # dp_head_channels: [128, 160, 192, 224] # 各中间层的通道数列表（如 [128, 160, 192, 224]）
        layer_kernel_sizes, # 各中间层的卷积核大小列表 layer_channels和layer_kernel_sizes定义了中间处理层的通道数和卷积核大小，两者长度需一致（每个层对应一组参数）
        act_layer, # 中间层激活函数类（如 nn.ReLU）
        norm_layer, # 中间层归一化层类（如 nn.BatchNorm1d）。act_layer和norm_layer用于特征转换。
        out_act_layer=nn.Identity, # 输出层激活函数（默认恒等映射，可改为 sigmoid 等）。最终输出的激活函数（如检测任务可能用 nn.Sigmoid 输出概率，默认不激活
        out_channels=1, # 输出通道数（如 1 表示单通道预测，如概率）。最终输出的激活函数（如检测任务可能用 nn.Sigmoid 输出概率，默认不激活
        **kwargs,
    ):
        super().__init__() # 调用父类的初始化方法（__init__ 方法）。

        assert len(layer_channels) == len(layer_kernel_sizes) # 确保层参数一一对应，assert用于断言判断，若满足后续的条件，则继续运行，若不满足，程序停止运行并发出报错。

        self.depth = len(layer_channels) # 中间层的数量

        self.up_layers = nn.ModuleList() # 存储上采样相关的层序列

        # 循环构建中间层（每个层包含：卷积 + 归一化 + 激活）
        # inc，输入通道数。outc，输出通道数。kers，卷积核大小。三个参数通过zip函数一一对应，循环构建出包含卷积归一化和激活函数的完整层结构。
        for inc, outc, kers in zip(
                [feature_channels] + layer_channels[:-1], # 输入通道：初始为feature_channels，后续为前一层输出
                layer_channels[:-1] + [out_channels * 2], # 输出通道：中间为layer_channels，最后一层为out_channels*2
                layer_kernel_sizes, # 卷积核大小
        ):
            conv = nn.Conv1d(in_channels=inc, out_channels=outc, kernel_size=kers) # 1D卷积
            norm = norm_layer(outc) # 归一化（与输出通道匹配）
            act = act_layer() # 激活函数

            # 用Sequential组合为层序列（保留名称便于访问），它是 PyTorch 中用于按顺序组合神经网络层的容器，会按添加顺序依次执行各层操作

            self.up_layers.append(
                nn.Sequential(
                    OrderedDict([("conv", conv), ("norm", norm), ("act", act)])
                )
            )

        # 最终输出卷积层（将中间结果转换为目标通道数）
        # 每个层包含卷积、归一化、激活，作用是在逐步上采样的同时处理特征，通道数从 feature_channels 逐步过渡到 out_channels*2（多通道便于保留更多特征信息）
        # out_conv 的作用：将 out_channels*2 通道的特征压缩为 out_channels 通道，卷积核 7+padding3 确保序列长度不变，最终通过 out_act 输出预测结果。
        self.out_conv = nn.Conv1d(
            in_channels=out_channels * 2, # 输入为上一层的out_channels*2
            out_channels=out_channels,  # 输出为目标通道数
            kernel_size=7,
            padding=3, # 7-1=6，6/2=3，保证卷积后长度不变（same padding）
        )
        self.out_act = out_act_layer() # 输出激活（如概率预测用sigmoid）

    # 计算每个中间层的上采样目标尺寸，实现渐进式上采样（从输入特征的长度 in_size 逐步放大到 out_size，即与 x0 匹配的长度）
    def _upsampling_sizes(self, in_size: int, out_size: int):
        sizes = [out_size] * self.depth  # 初始化所有层的目标尺寸为最终输出尺寸
        factor = (out_size / in_size) ** (1 / self.depth)  # 计算每层的缩放因子（等比缩放）
        # 从后往前调整尺寸，确保逐步从in_size放大到out_size
        for i in range(self.depth - 2, -1, -1):
            sizes[i] = int(sizes[i + 1] / factor)
        return sizes

    def forward(self, x, x0):
        N, C, L = x.size() # x：输入特征，形状为 (batch, channels, length)
        # 计算各层的上采样目标尺寸（最终与x0的长度一致，x0通常是早期高分辨率特征）
        up_sizes = self._upsampling_sizes(in_size=L, out_size=x0.size(-1))

        # 逐层上采样并处理特征
        for i, layer in enumerate(self.up_layers):
            upsize = up_sizes[i] # 当前层的目标尺寸
            x = F.interpolate(x, size=upsize, mode="linear")  # 上采样：用线性插值将x放大到upsize（适合1D序列）
            x = _auto_pad_1d(x, layer.conv.kernel_size[0], layer.conv.stride[0]) # 自动填充：匹配当前层卷积的核大小和步长（确保卷积后尺寸正确）
            x = layer(x) # 通过当前层（卷积+归一化+激活）

        # 最终卷积和激活，输出预测结果
        x = self.out_conv(x)
        x = self.out_act(x)
        return x

# HeadClassification中文可理解为 “分类头”，其核心功能是将输入的序列特征转换为最终的分类结果，通常作为神经网络的输出层，用于解决分类任务
# feature_channels(特征通道数), num_classes（几种类别的分类）, out_act_layer（输出层激活函数）, **kwargs（可选参数预留备用）


class HeadClassification(nn.Module):
    """Head of classification."""

    def __init__(self, feature_channels, num_classes, out_act_layer, **kwargs):
        super().__init__()

        # 两个1D卷积层：用于特征降维（序列长度）和进一步提取关键特征。通过大步长（stride=4）实现下采样（序列长度变为原来的 1/4），减少后续计算量。
        self.convs = nn.ModuleList([nn.Conv1d(feature_channels, feature_channels, 16, 4) for _ in range(2)])
        # 自适应平均池化：将序列长度压缩为1（保留通道维度）会将输入序列的最后一个维度（长度维度）压缩为 1，即对每个通道的整个序列取平均值。
        self.pool = nn.AdaptiveAvgPool1d(1)
        # 展平层：将多维特征转换为一维向量，适配后续全连接层的输入格式。
        self.flatten = nn.Flatten(1, -1)
        # 全连接层：将特征映射到类别数量维度，实现从特征到类别分数（logits）的线性映射。
        self.lin = nn.Linear(feature_channels , num_classes)
        # 输出激活函数：将全连接层输出转换为最终分类结果（如概率）（输出激活函数是将特征转换成分类结果的函数，比如二分类，特征通过激活函数变成了1和0，分别代表不同的值）
        self.out_act = out_act_layer()

    #前向传播方法 forward，for循环通过两个1D卷积层实现下采样，强化特征。特征然后进入了池化层，展平层，全连接层生成了分类的线性映射。最后通过激活层，将类别分数转化成分类结果。

    def forward(self, x, _: torch.Tensor = None):
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lin(x)
        x = self.out_act(x)
        return x

# HeadRegression中文可理解为 “回归头”，其核心功能是将输入的序列特征转换为最终的回归预测结果（即一个连续的数值）
# 输入特征通道数，输出层激活函数，其他的关键参数
# 与分类头（HeadClassification）的对比
# 相同点：两者均通过 “卷积下采样→池化→展平→全连接” 的流程处理特征，结构高度相似。
# 核心区别：
# 输出维度：回归头的全连接层输出为 1（单个连续值），分类头输出为 num_classes（类别数）。
# 任务目标：回归头用于预测连续变量（如 “预测温度值”），分类头用于预测离散类别（如 “判断图像类别”）。

class HeadRegression(nn.Module):
    """Head of regression."""

    def __init__(self, feature_channels, out_act_layer, **kwargs):
        super().__init__()

        #两个1D卷积层 池化层 展平层 全连接层 激活函数
        self.convs = nn.ModuleList([nn.Conv1d(feature_channels, feature_channels, 16, 4) for _ in range(2)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        # 将特征映射到单个输出值（回归任务的连续值）
        self.lin = nn.Linear(feature_channels , 1)
        # 输出激活函数：对回归结果进行后处理（如限制范围）
        self.out_act = out_act_layer()

    def forward(self, x, _: torch.Tensor = None):
        # x : [b c (n p)]
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lin(x)
        x = self.out_act(x)
        return x





class HeadBAZ(nn.Module):
    """Head of Back-Azimuth Estimation."""

    def __init__(self, feature_channels, out_act_layer, **kwargs):
        super().__init__()

        self.convs = nn.ModuleList([nn.Conv1d(feature_channels, feature_channels, 16, 4) for _ in range(2)])
        # 全局聚合：池化和展平操作将序列特征转换为全局向量，综合整个序列的信息。
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        # 全连接层：将特征映射到2个输出分量（用于表示角度的参数化形式）
        # 输出维度为2，这是该模块的核心设计 —— 背方位角是周期性变量（如0° 与360° 等价），直接预测角度值（如0 - 360）容易因边界问题
        # （如359° 与1° 实际接近但数值差异大）导致预测困难。因此，通常用角度的正弦（sin）和余弦（cos）值来参数化角度（两者取值范围均为[-1, 1]，
        # 且能唯一确定角度：azimuth = arctan2(sin, cos)）。这里的2个输出分量正对应这两个值。
        self.lin = nn.Linear(feature_channels , 2)
        # 输出激活函数：将输出分量限制在特定范围（如[-1, 1]，适配角度函数的取值）
        self.out_act = out_act_layer()

    def forward(self, x, _: torch.Tensor = None):
        # x : [b c (n p)]
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lin(x)
        x = self.out_act(x)
        return x[:, :1], x[:, 1:]

# 一个专为地震数据处理设计的混合模型，结合了多尺度卷积特征提取与大语言模型（LLM）的序列建模能力，最终通过不同的输出头（Head）完成特定任务（如地震事件检测、相位拾取、回归预测等）。
# SeisMoLLM 是一个面向地震数据的端到端处理模型，核心优势是融合了多尺度卷积的局部特征提取能力和 LLM 的长序列建模能力，适用于多种地震学任务：
# 地震事件检测与相位拾取（用 HeadDetectionPicking）；
# 震级、深度等参数回归（用 HeadRegression）；
# 震源背方位角估计（用 HeadBAZ）；
# 地震信号分类（如天然地震 vs 人工爆破，用 HeadClassification）。

class SeisMoLLM(nn.Module):
    def __init__(
        self,
        in_channels=3, # 输入数据的通道数（如地震数据的三分量：南北、东西、垂直）
        conv_scale_num=4, # 每个多尺度卷积块的分支数量
        conv_scale_strides=[8, 6, 4, 2], # 每个多尺度卷积块的尺度步长（控制分支卷积核差异）
        conv_channels=[16, 48, 96], # 卷积层通道数列表（逐步提升特征维度） 第一层输出16 第二层48 第三层96 逐步提升。
        conv_kernel_sizes=[16, 8, 6, 1], # 每个卷积块的基础卷积核大小
        conv_strides=[2, 2, 2, 1], # 每个卷积块的步长（控制下采样）
        llm_layers = 3, # 使用的 LLM 层数（从 GPT-2 中截取）
        d_model=768, # LLM 的隐藏层维度（GPT-2 小模型通常为 768）
        patch_size=8, # LLM 输入的分块大小（将特征序列分块后输入 LLM）
        dp_head_channels = [128, 160, 192, 224], # 检测/拾取头的通道数
        path_drop_rate=0.2, # 路径丢弃率（正则化）例如在神经网络训练时，以一定路径丢弃率随机断开神经元间连接，使模型不会过度依赖某些特定路径，从而避免过拟合。
        mlp_drop_rate=0.2, # MLP 层丢弃率（正则化）在多层感知机（MLP）神经网络结构中，应用丢弃
        mlp_ratio=4, # MLP 扩展比例若原隐藏层有 100 个神经元，按 1.5 的扩展比例增加，新的神经元数量就是 150 个 。它用于精确控制 MLP 模型在规模、复杂度等方面的扩展程度，以适应不同的数据特征和任务需求 。
        mlp_bias=True, # MLP 是否使用偏置
        act_layer=nn.GELU, # 激活函数（默认 GELU，适合 Transformer 类模型）
        norm_layer=nn.BatchNorm1d, # 归一化层（1D 批归一化，适合序列数据）
        use_checkpoint=False, # 是否使用检查点（节省内存）
        output_head=HeadRegression,  # 输出头类型（决定任务：回归、检测等）
        **kwargs
    ):
        super().__init__()

        # 断言：确保卷积通道数、卷积核大小、步长的数量匹配（卷积块数量 = len(conv_channels)）
        assert len(conv_channels) + 1 == len(conv_kernel_sizes) == len(conv_strides)
        # 调整最后一个卷积通道数，使其与 LLM 输入维度适配（d_model // patch_size）
        conv_channels.append(d_model // patch_size)

        self.use_checkpoint = use_checkpoint
        self.patch_size = patch_size
        self.feature_channels = conv_channels[-1]

        # 多尺度卷积嵌入器 self.convs
        # 作为模型的 “前端”，将原始地震数据（in_channels = 3，可能对应三分量地震信号）通过多个多尺度卷积块逐步提取特征，
        # 通道数从初始值提升到 d_model // patch_size，同时通过步长实现下采样，压缩序列长度。
        # 多尺度卷积块能捕捉地震信号中不同频率（如高频噪声、低频有效信号）的特征，适配地震数据的多尺度特性。

        # Multi-Scale Convolutional Embedder
        self.convs = nn.Sequential(
            *[
                Multi_Scale_Conv_Block( # 每个块都是多尺度卷积块（之前解析过）
                    scale_num=conv_scale_num, # 每个块的分支数量
                    scale_stride=ss,  # 尺度步长（控制分支卷积核差异）
                    in_dim=inc, # 输入通道（初始为in_channels，后续为前一层输出）
                    out_dim=outc, # 输出通道（conv_channels中的值）
                    kernel_size=kers, # 基础卷积核大小
                    stride=strd, # 步长（控制下采样）
                    act_layer=act_layer, # 激活函数
                    norm_layer=norm_layer, # 归一化层
                )
                for ss, inc, outc, kers, strd in zip(
                    conv_scale_strides,
                    [in_channels] + conv_channels[:-1], # 输入通道序列
                    conv_channels, # 输出通道序列
                    conv_kernel_sizes, # 卷积核大小序列
                    conv_strides, # 步长序列
                )
            ]
        )


        # LLM 序列建模块 self.llm_blocks
        # 作为模型的 “中端”，接收卷积提取的特征，通过LLM（GPT - 2）处理。具体来说，卷积输出的特征会被分块（patch_size），转换为LLM可处理的序列格式，利用
        # GPT - 2的Transformer层捕捉特征间的长距离依赖（如地震波的传播时序关系）。

        # Pre-trained LLM Blocks
        self.llm_blocks = LLM_Block(
            start_layer=0, # 截取 GPT-2 的起始层
            end_layer=llm_layers, # 截取 GPT-2 的结束层（共使用 llm_layers 层)
            patch_size=patch_size, # 分块大小（与之前通道调整匹配）
            lora_config=GPT2_lora  # GPT-2 的 LoRA 配置（参数高效微调）
        )
        # self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=12,
        #                                        dropout=0.1, batch_first=True)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=12, dim_feedforward=d_model * 4, batch_first=True)
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 输出头 self.out_head（任务适配）
        # 功能：作为模型的 “后端”，将LLM处理后的特征转换为具体任务的输出。
        # 例如：
        # 若output_head = HeadDetectionPicking，则输出与原始输入等长的序列，标记地震事件的位置；
        # 若output_head = HeadRegression，则输出一个连续值（如震级）；
        # 若output_head = HeadBAZ，则输出背方位角的参数化分量。
        # 设计细节：检测 / 拾取头需要上采样到原始数据长度，因此反转通道和卷积核顺序，与前端下采样过程对称，确保尺度匹配。


        # 若输出头是检测/拾取头（HeadDetectionPicking）
        if (output_head in [HeadDetectionPicking]) or (
            isinstance(output_head, partial)
            and (output_head.func in [HeadDetectionPicking])
        ):
            # 调整输出头的通道和卷积核（反转顺序，适配上采样）
            out_layer_channels = []
            out_layer_kernel_sizes = []
            for channel, kernel in zip(dp_head_channels, conv_kernel_sizes):
                out_layer_channels.insert(0, channel) # 反转通道顺序
                out_layer_kernel_sizes.insert(0, kernel) # 反转卷积核顺序

            # 初始化检测/拾取头
            self.out_head = output_head(
                in_channels=in_channels,
                feature_channels=self.feature_channels,
                layer_channels=out_layer_channels,
                layer_kernel_sizes=out_layer_kernel_sizes,
                act_layer=act_layer,
                norm_layer=norm_layer,
                path_drop_rate=path_drop_rate,
                mlp_drop_rate=mlp_drop_rate,
                mlp_ratio=mlp_ratio,
                mlp_bias=mlp_bias
            )

        # 其他输出头（如回归、分类、方位角估计）
        else:
            self.out_head = output_head(
                feature_channels=self.feature_channels,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )

    def forward(self, x):
        x_input = x
        # Multi Scale Conv Embedder
        x = self.convs(x)
        # LLM Blocks
        x = self.llm_blocks(x)
        
        '''
        for ablations of changing LLM to attention layer or Transformer layer

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        x = rearrange(x, 'b c n p -> b n (c p)')

        x, _ = self.attention(x, x, x)
        x = self.transformer(x)

        x = rearrange(x, 'b n (c p) -> b c (n p)', p=self.patch_size)
        '''

        # Output head
        x = self.out_head(x, x_input)
        return x


@register_model
def SeisMoLLM_dpk(**kwargs):
    """Detection and Phase-Picking."""
    # Only for picking in this work, you can add detection by modifying config.py
    model = SeisMoLLM(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(
            HeadDetectionPicking, out_act_layer=nn.Sigmoid, out_channels=3
        ),  # actually in use channel is 2
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_pmp(**kwargs):
    """P-motion-polarity classification."""
    model = SeisMoLLM(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(
            HeadClassification, out_act_layer=partial(nn.Softmax, dim=-1), num_classes=2
        ),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_emg(**kwargs):
    """Magnitude estimation."""
    model = SeisMoLLM(
        output_head=partial(
            HeadRegression,
            out_act_layer=partial(
                ScaledActivation, act_layer=nn.Sigmoid, scale_factor=8
            ),
        ),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_baz(**kwargs):
    """Azimuth estimation."""
    model = SeisMoLLM(
        output_head=partial(
            HeadBAZ,
            out_act_layer=partial(
                nn.Tanh
            ),
        ),
        **kwargs,
    )
    return model

@register_model
def SeisMoLLM_dis(**kwargs):
    """Epicentral distance estimation."""
    model = SeisMoLLM(
        output_head=partial(
            HeadRegression,
            out_act_layer=partial(
                ScaledActivation, act_layer=nn.Sigmoid, scale_factor=500
            ),
        ),
        **kwargs,
    )
    return model
