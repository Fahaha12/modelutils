import torch
import numpy as np

# 加载 PyTorch 保存的模型文件
model_path = r"D:\nnUNetWeb\STUNet\Pre-trained Models\small_ep4k.model"
model_data = torch.load(model_path)

# 修改键名
if "epoch" in model_data:
    model_data["current_epoch"] = model_data.pop("epoch")
    print("已将键名 'epoch' 修改为 'current_epoch'")
else:
    print("模型文件中不存在键名 'epoch'")

if "optimizer_state_dict" in model_data:
    model_data["optimizer_state"] = model_data.pop("optimizer_state_dict")
    print("已将键名 'optimizer_state_dict' 修改为 'optimizer_state'")
else:
    print("模型文件中不存在键名 'optimizer_state_dict'")

if "state_dict" in model_data:
    model_data["network_weights"] = model_data.pop("state_dict")
    print("已将键名 'state_dict' 修改为 'network_weights'")
else:
    print("模型文件中不存在键名 'state_dict'")

if "amp_grad_scaler" in model_data:
    model_data["grad_scaler_state"] = model_data.pop("amp_grad_scaler")
    print("已将键名 'amp_grad_scaler' 修改为 'grad_scaler_state'")
else:
    print("模型文件中不存在键名 'state_dict'")

# 删除键
if "lr_scheduler_state_dict" in model_data:
    del model_data["lr_scheduler_state_dict"]
    print("已删除键 'lr_scheduler_state_dict'")
else:
    print("模型文件中不存在键 'lr_scheduler_state_dict'")

if "plot_stuff" in model_data:
    del model_data["plot_stuff"]
    print("已删除键 'plot_stuff'")
else:
    print("模型文件中不存在键 'plot_stuff'")

if "best_stuff" in model_data:
    del model_data["best_stuff"]
    print("已删除键 'best_stuff'")
else:
    print("模型文件中不存在键 'best_stuff'")

# 新建空白键
model_data["logging"] = {
    "mean_fg_dice": [],
    "ema_fg_dice": [],
    "dice_per_class_or_region": [],
    "train_losses": [],
    "val_losses": [],
    "lrs": [],
    "epoch_start_timestamps": [],
    "epoch_end_timestamps": []
}
print("已新建空白键 'logging'")

model_data["_best_ema"] = np.float64(0.7113046541216426)
print("已新建键 '_best_ema'")

model_data["init_args"] = {
    "plans": {},
    "configuration": "",
    "fold": 0,
    "dataset_json": {},
    "unpack_dataset": False,
    "device": torch.device("cpu")
}
print("已新建空白键 'init_args'")

model_data["trainer_name"] = "STUNetTrainer_base"
print("已新建键 'trainer_name'")

model_data["inference_allowed_mirroring_axes"] = (0, 1, 2)
print("已新建键 'inference_allowed_mirroring_axes'")

# 保存修改后的模型文件
new_model_path = r"D:\nnUNetWeb\STUNet\Pre-trained Models\small_ep4k_changedkey.pth"
torch.save(model_data, new_model_path)
print("已保存修改后的模型文件:", new_model_path)