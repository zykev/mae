import os
import glob
import random
from collections import Counter
from collections.abc import Mapping

import PIL.Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from util import dental_records

DEFAULT_CATEGORIES = "full,tooth,sextant,group"


def parse_category_sampling(spec):
    """Parse "--category_sampling full:1.0,tooth:0.5,..." into {crop_type: fraction}."""
    if spec is None:
        return None
    if isinstance(spec, Mapping):
        return {str(k): float(v) for k, v in spec.items()}
    result = {}
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            crop_type, fraction = token.split(":", 1)
            result[crop_type.strip()] = float(fraction)
        else:
            result[token] = 1.0
    return result


# 自定义 Dataset 映射，用于读取图片列表
class DentalDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = PIL.Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # MAE 训练通常忽略 label，这里返回 0 作为占位符
        return img, 0


def _crop_from_full(full_img, box_original):
    """Cut a crop out of the full-resolution image. Boxes are floats; floor/ceil
    the bounds so every valid box covers at least one source pixel."""
    x1, y1, x2, y2 = box_original
    return full_img.crop((int(x1), int(y1), int(x2) + 1, int(y2) + 1))


class DentalRecordDataset(Dataset):
    """SSL dataset over case-level split records (see util/dental_records.py).

    Each record is one crop: full / tooth / sextant / group. Crops other than
    "full" are re-cut from the full-resolution source image (record
    ``source_image_path`` + ``box_original``) so pixel quality matches what
    downstream feature evaluation (compare/) will feed the model.

    The dataset exports ``<output_dir>/<partition>_manifest.jsonl`` and tracks
    per-sample exposure counts for coverage reporting (§8 coverage.json).
    """

    def __init__(self, records, transform=None, is_train=True, output_dir=None,
                 category_sampling=None, seed=0):
        self.records = list(records)
        self.transform = transform
        self.is_train = is_train
        self.output_dir = output_dir
        self.category_sampling = parse_category_sampling(category_sampling)
        self.seed = seed
        self.exposures = Counter()  # sample_key -> number of times fetched
        self.image_paths = [r["source_image_path"] for r in self.records]  # compat shim
        if self.category_sampling and self.is_train:
            self._apply_category_sampling()
        self.manifest_path = None

    def _apply_category_sampling(self):
        """Deterministically downsample per-category pools (subset per epoch).

        Keeps the dataset length fixed: for each crop type, keep
        ceil(fraction * n_type) records selected with a fixed seed, then the
        epoch sampler shuffles the union. A fraction of 1.0 keeps every record.
        """
        by_crop = {}
        for record in self.records:
            by_crop.setdefault(record["crop_type"], []).append(record)
        kept = []
        for crop_type, records in by_crop.items():
            fraction = self.category_sampling.get(crop_type, 1.0)
            if fraction >= 1.0:
                kept.extend(records)
                continue
            rng = random.Random(self.seed)
            pool = list(records)
            rng.shuffle(pool)
            n_keep = max(1, int(len(pool) * fraction))
            kept.extend(pool[:n_keep])
        self.records = kept
        self.image_paths = [r["source_image_path"] for r in self.records]

    def set_epoch(self, epoch):
        """Called by the epoch sampler to reshuffle a deterministic per-epoch order."""
        rng = random.Random(self.seed + epoch)
        indices = list(range(len(self.records)))
        rng.shuffle(indices)
        self.records = [self.records[i] for i in indices]
        self.image_paths = [r["source_image_path"] for r in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if self.is_train:
            self.exposures[record["sample_key"]] += 1
        full_img = PIL.Image.open(record["source_image_path"]).convert("RGB")
        if record["crop_type"] == "full":
            img = full_img
        else:
            img = _crop_from_full(full_img, record["box_original"])
        if self.transform is not None:
            img = self.transform(img)
        return img, 0

    def write_manifest(self, name="train_manifest"):
        if not self.output_dir:
            return None
        from util.dental_records import write_manifest
        partition = "train" if self.is_train else "val"
        self.manifest_path = write_manifest(
            self.records, os.path.join(self.output_dir, f"{name}.jsonl"))
        return self.manifest_path

    def audit(self, partition=None):
        from util.dental_records import audit_records
        return audit_records(self.records, partition or ("train" if self.is_train else "val"))

def get_intraoral_images(data_path):
    """
    根据特定结构爬取所有图片路径
    结构: Collector / *_process / {process, sextant, single_tooth} / sample_id / ...
    """
    all_images = []
    
    # 1. 遍历收集者文件夹 (e.g., peiliu)
    for collector in os.listdir(data_path):
        collector_path = os.path.join(data_path, collector)
        if not os.path.isdir(collector_path):
            continue
            
        # 2. 遍历以 _process 结尾的日期文件夹
        for date_folder in os.listdir(collector_path):
            if not date_folder.endswith('_process'):
                continue
            
            base_folder = os.path.join(collector_path, date_folder)
            
            # --- 分支 A: process 文件夹 ---
            # 路径: .../process/sample_id/{D,F,L,R,U}.png
            process_root = os.path.join(base_folder, 'process')
            if os.path.exists(process_root):
                # 匹配所有 sample_id 下的 png
                all_images.extend(glob.glob(os.path.join(process_root, "*", "*.png")))

            # --- 分支 B: sextant 文件夹 ---
            # 路径: .../sextant/sample_id/{F,L,R}/*.png
            sextant_root = os.path.join(base_folder, 'sextant')
            if os.path.exists(sextant_root):
                # 匹配 sample_id 下 F, L, R 文件夹内的所有 png
                all_images.extend(glob.glob(os.path.join(sextant_root, "*", "[FLR]", "*.png")))

            # --- 分支 C: single_tooth 文件夹 ---
            # 路径: .../single_tooth/sample_id/{D,F,L,R,U}/*.png
            tooth_root = os.path.join(base_folder, 'single_tooth')
            if os.path.exists(tooth_root):
                # 匹配 sample_id 下 D, F, L, R, U 文件夹内的所有 png
                all_images.extend(glob.glob(os.path.join(tooth_root, "*", "[DFLRU]", "*.png")))

    return sorted(all_images)

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    # 优先使用病例级 split：--data_path + --split_json + --split
    # 按 case 过滤 records，支持 --categories full,tooth,sextant,group
    split_json = getattr(args, "split_json", None)
    if split_json:
        split = getattr(args, "split", "train")
        categories = getattr(args, "categories", None) or DEFAULT_CATEGORIES
        if isinstance(categories, str):
            categories = [c.strip() for c in categories.split(",") if c.strip()]
        print(f"正在从 {split_json} 读取 split[{split}]，categories={categories}")
        records = dental_records.build_records(
            args.data_path, split_json, split, categories)
        dataset = DentalRecordDataset(
            records, transform=transform, is_train=is_train,
            output_dir=args.output_dir,
            category_sampling=getattr(args, "category_sampling", None),
            seed=getattr(args, "seed", 0),
        )
        print(f"{'训练' if is_train else '验证'}集构建完成，包含记录数量: {len(dataset)}")
        return dataset

    # 旧逻辑兜底：从磁盘结构爬取所有图片（无 case 级 split 时使用）
    # 优先 .datasets/intraoral，挂载点缺失/符号链接失效时回退 .datasets/intraoral1。
    data_path = str(dental_records.resolve_compatible_path(args.data_path))
    print(f"正在从 {data_path} 检索口腔影像数据...")
    all_imgs = get_intraoral_images(data_path)
    dataset = DentalDataset(all_imgs, transform=transform)

    print(f"{'训练' if is_train else '验证'}集构建完成，包含图片数量: {len(dataset)}")
    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        # transform = create_transform(
        #     input_size=args.input_size,
        #     is_training=True,
        #     color_jitter=args.color_jitter,
        #     auto_augment=args.aa,
        #     interpolation='bicubic',
        #     re_prob=args.reprob,
        #     re_mode=args.remode,
        #     re_count=args.recount,
        #     mean=mean,
        #     std=std,
        # )
        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean, std=std)])
        

        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                args.input_size,
                scale=(0.5, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.15,
                    hue=0.03,
                )
            ], p=0.5),
            transforms.RandomRotation(
                degrees=10,
                interpolation=transforms.InterpolationMode.BICUBIC,
                fill=0,
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
