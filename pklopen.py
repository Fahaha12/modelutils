import pickle

# 指定要打开的 pkl 文件路径
pkl_file_path = r"D:\nnUNetWeb\STUNet\Pre-trained Models\base_ep4k.model.pkl"

# 打开 pkl 文件
with open(pkl_file_path, "rb") as file:
    # 使用 pickle.load() 函数加载 pkl 文件内容
    data = pickle.load(file)

# 打印加载的数据
print("加载的数据:")
print(data)

# 查看数据的类型
print("数据的类型:", type(data))

# 如果数据是字典类型,可以访问其键值对
if isinstance(data, dict):
    print("数据包含的键:")
    for key in data.keys():
        print(key)
    
    print("访问字典中的元素:")
    for key, value in data.items():
        print(f"{key}: {value}")

# 如果数据是列表类型,可以访问其元素
elif isinstance(data, list):
    print("数据包含的元素:")
    for item in data:
        print(item)

# 如果数据是其他类型,可以根据需要进行处理
else:
    print("数据是其他类型,请根据需要进行处理")