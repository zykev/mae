import os
import glob
import random
random.seed(42) # 保证每次可视化的图片是一样的，方便对比不同权重
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import models_mae

# --- 配置区 ---
DATA_PATH = ".datasets/intraoral/intraoral"
# 填入你最新的 checkpoint 路径
CHKPT_DIR = 'exp/pretrain_v2/checkpoint-399.pth' 
SAVE_DIR = "exp/pretrain_v2/visualizations_4col"
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# --- 工具函数 ---

def get_categorized_paths(data_path):
    categories = {'process': [], 'sextant': [], 'single_tooth': []}
    if not os.path.exists(data_path):
        print(f"错误: 数据集路径 {data_path} 不存在")
        return categories
        
    for collector in os.listdir(data_path):
        c_path = os.path.join(data_path, collector)
        if not os.path.isdir(c_path): continue
        for date_folder in os.listdir(c_path):
            if not date_folder.endswith('_process'): continue
            base = os.path.join(c_path, date_folder)
            
            p_path = os.path.join(base, 'process')
            if os.path.exists(p_path):
                categories['process'].extend(glob.glob(os.path.join(p_path, "*", "*.png")))
            s_path = os.path.join(base, 'sextant')
            if os.path.exists(s_path):
                categories['sextant'].extend(glob.glob(os.path.join(s_path, "*", "[FLR]", "*.png")))
            t_path = os.path.join(base, 'single_tooth')
            if os.path.exists(t_path):
                categories['single_tooth'].extend(glob.glob(os.path.join(t_path, "*", "[DFLRU]", "*.png")))
    return categories

def prepare_model(chkpt_dir, arch='mae_vit_base_patch16'):
    model = getattr(models_mae, arch)()
    # 兼容性加载，处理 weights_only 问题
    try:
        checkpoint = torch.load(chkpt_dir, map_location='cpu', weights_only=False)
    except TypeError:
        # 旧版本 PyTorch 不支持 weights_only
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"模型加载状态: {msg}")
    model.eval()
    return model

def process_img(img_path):
    img = Image.open(img_path).convert("RGB") # 解决 RGBA 问题
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img

def denormalize(img_tensor):
    img = img_tensor.detach().cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

@torch.no_grad()
def get_4col_inference(model, img_path, mask_ratio=0.75):
    """
    运行单张图推理
    返回反归一化后的四张图：原图, 掩码图, 纯预测图, 拼接图
    """
    img_np = process_img(img_path)
    x = torch.tensor(img_np).float().permute(2, 0, 1).unsqueeze(0)
    
    # 运行 MAE
    loss, y, mask = model(x, mask_ratio=mask_ratio)
    y = model.unpatchify(y) # 模型预测的全图
    
    # 处理 Mask (用于制作第2列和第4列)
    mask = mask.detach().unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
    mask = model.unpatchify(mask)
    
    # 1. 掩码图 (原始图片 * 可见部分) - 展示给模型看的部分
    im_masked = x * (1 - mask)
    
    # 2. 拼接图 (可见部分像素 + 预测部分像素)
    im_paste = x * (1 - mask) + y * mask
    
    # 反归一化所有 Tensor
    orig = denormalize(x.squeeze(0).permute(1, 2, 0))
    masked = denormalize(im_masked.squeeze(0).permute(1, 2, 0))
    recon_pure = denormalize(y.squeeze(0).permute(1, 2, 0)) # 纯预测
    recon_paste = denormalize(im_paste.squeeze(0).permute(1, 2, 0)) # 拼接
    
    return orig, masked, recon_pure, recon_paste

def save_4col_grid(category_name, paths, model, n=20, mask_ratio=0.75):
    """选择n张图并保存 20x4 的对比大图"""
    if len(paths) == 0: return
    selected = random.sample(paths, min(len(paths), n))
    
    # 创建画布：n行，4列
    fig, axes = plt.subplots(len(selected), 4, figsize=(20, 5 * len(selected)))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    print(f"正在处理类别: {category_name} (共采样 {len(selected)} 张图)...")
    for i, path in enumerate(selected):
        # 即使模型预测了全图，我们也给它 75% 的 Mask 难度来观察它的推理能力
        orig, masked, recon_pure, recon_paste = get_4col_inference(model, path, mask_ratio=mask_ratio)
        
        # 第一列：原图
        axes[i, 0].imshow(orig)
        axes[i, 0].axis('off')
        if i == 0: axes[i, 0].set_title("Original (GT)", fontsize=20)
        
        # 第二列：掩码图
        axes[i, 1].imshow(masked)
        axes[i, 1].axis('off')
        if i == 0: axes[i, 1].set_title("Masked Input", fontsize=20)
        
        # 第三列：纯模型预测
        axes[i, 2].imshow(recon_pure)
        axes[i, 2].axis('off')
        if i == 0: axes[i, 2].set_title("Pure Prediction", fontsize=20)
        
        # 第四列：拼接图
        axes[i, 3].imshow(recon_paste)
        axes[i, 3].axis('off')
        if i == 0: axes[i, 3].set_title("Paste (Visible+Pred)", fontsize=20)
        
    out_path = os.path.join(SAVE_DIR, f"4col_comparison_{category_name}.jpg")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"四列对比图已保存至: {out_path}")

# --- 执行主程序 ---
if __name__ == "__main__":
    # 1. 加载模型
    if not os.path.exists(CHKPT_DIR):
        print(f"错误: 权重文件 {CHKPT_DIR} 未找到，请检查路径。")
    else:
        model_mae = prepare_model(CHKPT_DIR)
        
        # 2. 获取路径
        all_categories = get_categorized_paths(DATA_PATH)
        
        # 3. 批量生成
        for cat_name, paths in all_categories.items():
            save_4col_grid(cat_name, paths, model_mae, n=20, mask_ratio=0.1)
        
        print(f"\n可视化完成！请查看目录: {SAVE_DIR}")