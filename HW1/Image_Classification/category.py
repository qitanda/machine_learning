import os
import shutil
import random

# 定义路径
_dataset_dir = os.path.join(os.getcwd(), "food11")
train_dir = os.path.join(_dataset_dir,"training")
val_dir = os.path.join(_dataset_dir,"validation")
combined_dir = os.path.join(_dataset_dir,"combined")
new_train_dir = os.path.join(_dataset_dir,"new_train")
new_val_dir = os.path.join(_dataset_dir,"new_validation")

# 合并train和validation中的图片，重命名为类别_编号.jpg
def merge_and_rename_images(source_dir, combined_dir, start_idx=0):
    current_idx = start_idx
    for filename in os.listdir(source_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            class_label = filename.split('_')[0]  # 从文件名提取类别（例如0_0.jpg中的0）
            new_filename = f"{class_label}_{current_idx}.jpg"
            src = os.path.join(source_dir, filename)
            dst = os.path.join(combined_dir, new_filename)
            shutil.copy(src, dst)
            current_idx += 1
    return current_idx

# 按比例划分数据集
def split_dataset(combined_dir, new_train_dir, new_val_dir, split_ratio=0.9):
    images = os.listdir(combined_dir)
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)

    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for img in train_images:
        shutil.copy(os.path.join(combined_dir, img), os.path.join(new_train_dir, img))

    for img in val_images:
        shutil.copy(os.path.join(combined_dir, img), os.path.join(new_val_dir, img))

    print("数据集合并和重新划分完成。")

if __name__ == "__main__":
    # 先复制train中的图片并重命名
    current_idx = merge_and_rename_images(train_dir, combined_dir)

    # 再复制validation中的图片并重命名
    merge_and_rename_images(val_dir, combined_dir, start_idx=current_idx)
    
    # 按比例重新划分数据集 (例如 80% 作为训练，20% 作为验证)
    split_dataset(combined_dir, new_train_dir, new_val_dir, split_ratio=0.9)
    

