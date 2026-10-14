import os
import random
import math
import numpy as np
from PIL import Image
from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.utils import split_labeled_unlabeled_data
from semilearn.datasets.augmentation import RandAugment

# 使用 ImageNet 的均值和方差作为默认值，适用于大多数自然图像
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def load_custom_data_from_folder(root_dir, img_size):
    """
    遍历文件夹读取数据 (增强版：自动过滤隐藏文件和垃圾文件夹)
    Structure: root_dir/class_name/image_file
    """
    data = []
    targets = []
    
    # === 修改点开始：增加过滤逻辑 ===
    # 1. 读取所有文件夹
    all_dirs = os.listdir(root_dir)
    # 2. 过滤掉：
    #    - 非文件夹
    #    - 以 '.' 开头的隐藏文件夹 (如 .ipynb_checkpoints)
    #    - 名字包含 'labeled' 或 'unlabeled' 的索引文件夹
    valid_classes = []
    for d in all_dirs:
        full_path = os.path.join(root_dir, d)
        if os.path.isdir(full_path):
            if d.startswith('.'):
                continue
            if 'labeled' in d or 'unlabeled' in d: # 过滤 labeled_idx 等
                continue
            valid_classes.append(d)
    
    desired_class_aliases = [
        ("正常", ["正常"]),
        ("柱塞磨损", ["柱塞磨损", "柱塞端面磨损"]),
        ("配油盘磨损", ["配油盘磨损", "配流盘磨损"]),
        ("斜盘磨损", ["斜盘磨损", "斜盘低压侧磨损"]),
        ("滑靴塌边", ["滑靴塌边"]),
        ("缸体轴外圈疲劳", ["缸体轴外圈疲劳", "外圈故障"]),
        ("缸体轴内圈疲劳", ["缸体轴内圈疲劳", "内圈故障"]),
        ("主轴疲劳", ["主轴疲劳"]),
        ("弹簧失效", ["弹簧失效"]),
    ]

    classes = []
    missing_classes = []
    for display_name, aliases in desired_class_aliases:
        matched_class = next((name for name in aliases if name in valid_classes), None)
        if matched_class is None:
            missing_classes.append(f"{display_name} ({'/'.join(aliases)})")
        else:
            classes.append(matched_class)

    extra_classes = sorted(set(valid_classes) - set(classes))
    if missing_classes:
        raise ValueError(f"Missing pic class folders: {missing_classes}")
    if extra_classes:
        raise ValueError(f"Unexpected pic class folders: {extra_classes}")
    # === 修改点结束 ===

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print("-" * 30)
    print(f"【DEBUG】实际检测到的有效类别 ({len(classes)}个):")
    for k, v in class_to_idx.items():
        print(f"  ID {v}: {k}")
    print("-" * 30)
    
    # 再次检查：如果你配置的 num_classes 不等于检测到的数量，抛出异常提醒你
    # 注意：你需要把 args 传进来，或者在这里手动硬编码检查
    # if len(classes) != 9: 
    #    print(f"警告：你配置了 num_classes=9，但实际找到了 {len(classes)} 个文件夹！")

    for cls_name in classes:
        cls_dir = os.path.join(root_dir, cls_name)
        img_files = os.listdir(cls_dir)
        
        # 记录当前类读了多少张，防止空文件夹导致 NaN
        count = 0 
        for img_name in img_files:
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
                
            img_path = os.path.join(cls_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((img_size, img_size)) 
                img_np = np.asarray(img)
                
                data.append(img_np)
                targets.append(class_to_idx[cls_name])
                count += 1
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
        
        if count == 0:
            print(f"【严重警告】类别 '{cls_name}' 是空的！这将导致除以零错误(NaN)！请删除该文件夹！")

    return np.array(data, dtype=np.uint8), np.array(targets)

def get_pic(args, name, data_dir='./data', include_lb_to_ulb=False, return_test=False):
    """
    适配自定义数据的加载函数
    """
    # 数据放在 data_dir/pic 下，例如 ./data/pic/正常/xxx.jpg
    root_path = os.path.join(data_dir, 'pic') 
    
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Data directory not found: {root_path}")

    # 1. 加载所有数据到内存
    all_data, all_targets = load_custom_data_from_folder(root_path, args.img_size)
    num_total = len(all_targets)
    
    if num_total == 0:
        raise ValueError("No images found in the directory.")

    # 2. 手动按类别分层划分 Train/Eval/Test。
    # Train 用于 labeled/unlabeled 采样，Eval 只用于训练中的定期评估，
    # Test 只用于最终模型测试。
    train_idx = []
    eval_idx = []
    test_idx = []
    for c in range(args.num_classes):
        cls_idx = np.where(all_targets == c)[0]
        np.random.shuffle(cls_idx)

        train_end = int(len(cls_idx) * 0.7)
        eval_end = train_end + int(len(cls_idx) * 0.1)

        train_idx.extend(cls_idx[:train_end])
        eval_idx.extend(cls_idx[train_end:eval_end])
        test_idx.extend(cls_idx[eval_end:])

    train_idx = np.asarray(train_idx)
    eval_idx = np.asarray(eval_idx)
    test_idx = np.asarray(test_idx)
    np.random.shuffle(train_idx)
    np.random.shuffle(eval_idx)
    np.random.shuffle(test_idx)

    train_data = all_data[train_idx]
    train_targets = all_targets[train_idx]
    eval_data = all_data[eval_idx]
    eval_targets = all_targets[eval_idx]
    test_data = all_data[test_idx]
    test_targets = all_targets[test_idx]

    # 3. 定义数据增强 (Transforms) - 保持原 CIFAR 代码逻辑
    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_medium = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 4. 使用 Semilearn 的工具划分有标签和无标签数据
    lb_idx, ulb_idx, lb_clean_idx, lb_noise_idx = split_labeled_unlabeled_data(args, train_data, train_targets,
                                                                               num_classes=args.num_classes,
                                                                               lb_num_labels=args.num_labels,
                                                                               ulb_num_labels=args.ulb_num_labels,
                                                                               lb_imbalance_ratio=args.lb_imb_ratio,
                                                                               ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                               noise_ratio=args.noise_ratio,
                                                                               noise_per_class=args.noise_per_class,
                                                                               lb_imb_type=args.lb_imb_type,
                                                                               ulb_imb_type=args.ulb_imb_type,
                                                                               num_steps=args.num_steps,
                                                                               include_lb_to_ulb=include_lb_to_ulb)

    # 5. 处理标签噪音逻辑 (与原代码保持一致)
    data, targets = train_data, train_targets
    noised_targets = targets.copy() # 初始化
    
    lb_count = [0 for _ in range(args.num_classes)]
    lb_clean_count = [0 for _ in range(args.num_classes)]
    lb_noise_count = [0 for _ in range(args.num_classes)]
    new_lb_noise_count = [0 for _ in range(args.num_classes)]
    ulb_count = [0 for _ in range(args.num_classes)]

    for c in targets[lb_idx]:
        lb_count[c] += 1
    for c in targets[ulb_idx]:
        ulb_count[c] += 1
    for c in targets[lb_clean_idx]:
        lb_clean_count[c] += 1
    if len(lb_noise_idx) > 0:
        for c in targets[lb_noise_idx]:
            lb_noise_count[c] += 1

    # 计算噪音概率矩阵 (仅当 noise_ratio > 0 时生效)
    p_noise = np.zeros((args.num_classes, args.num_classes))
    for i in range(args.num_classes):
        for j in range(args.num_classes):
            if i != j:
                # 防止除以0
                denominator = (sum(lb_count) - lb_count[i])
                if denominator > 0:
                    p_noise[i][j] = lb_count[j] / denominator
    
    row_sums = p_noise.sum(axis=1, keepdims=True)
    # 处理全0行防止 NaN
    row_sums[row_sums == 0] = 1 
    p_noise = p_noise / row_sums

    for i in lb_noise_idx:
        if args.noise_type == 'sym':
            noised_targets[i] = (random.randint(1, args.num_classes - 1) + targets[i]) % args.num_classes
        elif args.noise_type == 'asym':
            noised_targets[i] = np.random.choice(args.num_classes, p=p_noise[targets[i]])
        elif args.noise_type == 'circle':
            noised_targets[i] = (targets[i] + 1) % args.num_classes
    if len(lb_noise_idx) > 0:
        for c in noised_targets[lb_noise_idx]:
            new_lb_noise_count[c] += 1

    split_count = {
        "train": [int(np.sum(train_targets == c)) for c in range(args.num_classes)],
        "eval": [int(np.sum(eval_targets == c)) for c in range(args.num_classes)],
        "test": [int(np.sum(test_targets == c)) for c in range(args.num_classes)],
    }
    print(f"Data Loaded: Total {num_total} images.")
    print("split count: train {}, eval {}, test {}".format(
        split_count["train"], split_count["eval"], split_count["test"]
    ))
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))

    # 6. 构建 Dataset 对象
    lb_dset = BasicDataset(data[lb_idx], targets[lb_idx], noised_targets[lb_idx], args.num_classes, False,
                           weak_transform=transform_weak, strong_transform=transform_strong, onehot=False)

    ulb_dset = BasicDataset(data[ulb_idx], targets[ulb_idx], None, args.num_classes, True,
                            weak_transform=transform_weak, strong_transform=transform_strong, onehot=False)

    eval_dset = BasicDataset(eval_data, eval_targets, None, args.num_classes, False, weak_transform=transform_val,
                             strong_transform=None, onehot=False)
    test_dset = BasicDataset(test_data, test_targets, None, args.num_classes, False, weak_transform=transform_val,
                             strong_transform=None, onehot=False)

    lb_count_message = {'lb_count': lb_count, 'ulb_count': ulb_count, 'lb_clean_count': lb_clean_count,
                        'lb_noise_count': lb_noise_count, 'new_lb_noise_count': new_lb_noise_count}

    if return_test:
        return lb_dset, ulb_dset, eval_dset, test_dset, lb_count_message
    return lb_dset, ulb_dset, eval_dset, lb_count_message
