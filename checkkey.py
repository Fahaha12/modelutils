# import torch

# # 加载 PyTorch 保存的数据集文件
# dataset = torch.load(r"D:\nnUNetWeb\STUNet\nnUNet_results\Dataset043_BraTS2019\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_1\checkpoint_best.pth")

# # 查看数据集的类型
# print("数据集的类型:", type(dataset))

# # 查看数据集的长度
# print("数据集的长度:", len(dataset))

# # 查看数据集的第一个元素
# print("数据集的第一个元素:")
# print(dataset[0])

# # 如果数据集是一个字典，可以查看字典的键
# if isinstance(dataset, dict):
#     print("数据集包含的键:")
#     for key in dataset.keys():
#         print(key)

# # 如果数据集是一个列表或元组，可以查看前几个元素
# if isinstance(dataset, (list, tuple)):
#     print("数据集的前五个元素:")
#     for i in range(min(5, len(dataset))):
#         print(dataset[i])

# # 查看数据集元素的类型
# print("数据集元素的类型:", type(dataset[0]))

# # 如果数据集元素是字典，可以查看字典的键
# if isinstance(dataset[0], dict):
#     print("数据集元素包含的键:")
#     for key in dataset[0].keys():
#         print(key)

# # 如果数据集元素是张量，可以查看张量的形状
# if isinstance(dataset[0], torch.Tensor):
#     print("数据集元素的形状:", dataset[0].shape)

import torch

# 加载 PyTorch 保存的数据集文件
dataset = torch.load(r"D:\nnUNetWeb\STUNet\Pre-trained Models\small_ep4k_v2.pth")

# 创建一个txt文件用于保存输出信息
with open("small_ep4k_v2.model.txt", "w") as file:
    # 遍历数据集字典的每个键值对
    for key, value in dataset.items():
        file.write("键: {}\n".format(key))
        file.write("值的类型: {}\n".format(type(value)))
        
        # 如果值是字典,递归显示其结构
        if isinstance(value, dict):
            file.write("值的结构:\n")
            for sub_key, sub_value in value.items():
                file.write("  键: {}\n".format(sub_key))
                file.write("  值的类型: {}\n".format(type(sub_value)))
                if isinstance(sub_value, torch.Tensor):
                    file.write("  值的形状: {}\n".format(sub_value.shape))
        

        # 如果值是张量,显示其形状
        elif isinstance(value, torch.Tensor):
            file.write("值的形状: {}\n".format(value.shape))
        
        # 如果值是其他类型,直接显示其值
        else:
            file.write("值: {}\n".format(value))
        
        file.write("\n")