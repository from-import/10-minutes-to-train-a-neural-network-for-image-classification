import os
import random
import shutil

# 源文件夹和目标文件夹路径
train_dir = 'data/train'
val_dir = 'data/val'

# 获取train目录下的子文件夹列表
categories = os.listdir(train_dir)

# 遍历每个子文件夹
for category in categories:
    category_train_dir = os.path.join(train_dir, category)
    category_val_dir = os.path.join(val_dir, category)

    # 创建相应的val子文件夹
    os.makedirs(category_val_dir, exist_ok=True)

    # 获取train子文件夹中的图片列表
    images = os.listdir(category_train_dir)

    # 计算移动的图片数量
    num_images_to_move = int(len(images) * 0.1)

    # 随机选择要移动的图片
    images_to_move = random.sample(images, num_images_to_move)

    # 移动图片到val子文件夹中
    for image in images_to_move:
        image_path = os.path.join(category_train_dir, image)
        new_image_path = os.path.join(category_val_dir, image)
        shutil.move(image_path, new_image_path)

print("移动完成！")
