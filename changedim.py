import torch

# 加载checkpoint
checkpoint_path = r"D:\nnUNetWeb\STUNet\Pre-trained Models\small_ep4k_changedkey.pth"
checkpoint = torch.load(checkpoint_path)

# 获取模型的state_dict
state_dict = checkpoint['network_weights']

# 定义目标尺寸
target_conv1_size = torch.Size([32, 4, 3, 3, 3])
target_conv3_size = torch.Size([32, 4, 1, 1, 1])
target_seg_output_sizes = [
    torch.Size([4, 512, 1, 1, 1]),
    torch.Size([4, 256, 1, 1, 1]),
    torch.Size([4, 128, 1, 1, 1]),
    torch.Size([4, 64, 1, 1, 1]),
    torch.Size([4, 32, 1, 1, 1])
]
target_bias_size = torch.Size([4])

# 修改conv1的输入通道数
if 'conv_blocks_context.0.0.conv1.weight' in state_dict:
    param = state_dict['conv_blocks_context.0.0.conv1.weight']
    if param.size(1) != target_conv1_size[1]:
        # 重复通道数以匹配目标通道数
        new_param = param.repeat(1, target_conv1_size[1] // param.size(1), 1, 1, 1) / (target_conv1_size[1] // param.size(1))
        state_dict['conv_blocks_context.0.0.conv1.weight'] = new_param

# 修改conv3的输入通道数
if 'conv_blocks_context.0.0.conv3.weight' in state_dict:
    param = state_dict['conv_blocks_context.0.0.conv3.weight']
    if param.size(1) != target_conv3_size[1]:
        # 重复通道数以匹配目标通道数
        new_param = param.repeat(1, target_conv3_size[1] // param.size(1), 1, 1, 1) / (target_conv3_size[1] // param.size(1))
        state_dict['conv_blocks_context.0.0.conv3.weight'] = new_param

# 修改seg_output的输出类别数
for i, target_size in enumerate(target_seg_output_sizes):
    if f'seg_outputs.{i}.weight' in state_dict:
        param = state_dict[f'seg_outputs.{i}.weight']
        if param.size() != target_size:
            # 创建一个全零张量，大小与目标大小匹配
            new_param = torch.zeros(target_size)
            # 计算原始张量和目标张量在每个维度上的最小尺寸
            min_size = torch.Size([min(param.size(0), target_size[0]), min(param.size(1), target_size[1]), min(param.size(2), target_size[2]), min(param.size(3), target_size[3]), min(param.size(4), target_size[4])])
            # 将原始张量的值复制到新的全零张量中
            new_param[:min_size[0], :min_size[1], :min_size[2], :min_size[3], :min_size[4]] = param[:min_size[0], :min_size[1], :min_size[2], :min_size[3], :min_size[4]]
            state_dict[f'seg_outputs.{i}.weight'] = new_param
    if f'seg_outputs.{i}.bias' in state_dict:
        param = state_dict[f'seg_outputs.{i}.bias']
        if param.size(0) != target_bias_size[0]:
            # 创建一个全零张量，大小与目标大小匹配
            new_param = torch.zeros(target_bias_size)
            # 计算原始张量和目标张量在每个维度上的最小尺寸
            min_size = min(param.size(0), target_bias_size[0])
            # 将原始张量的值复制到新的全零张量中
            new_param[:min_size] = param[:min_size]
            state_dict[f'seg_outputs.{i}.bias'] = new_param

# 保存修改后的checkpoint
checkpoint['network_weights'] = state_dict
torch.save(checkpoint, r"D:\nnUNetWeb\STUNet\Pre-trained Models\small_ep4k_v2.pth")