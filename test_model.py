#!/usr/bin/env python3
import torch
import numpy as np
from transformers import ViTConfig
from model import VitDet3D

def test_model_loading():
    """测试模型加载和基本功能"""
    print("=== 肺结节检测模型测试 ===")
    
    # 加载配置
    try:
        config = ViTConfig.from_pretrained("model_config.json")
        print(f"✓ 配置加载成功")
        print(f"  - 图像尺寸: {config.image_size}")
        print(f"  - 补丁尺寸: {config.patch_size}")
        print(f"  - 隐藏层大小: {config.hidden_size}")
        print(f"  - 注意力头数: {config.num_attention_heads}")
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False
    
    # 创建模型
    try:
        model = VitDet3D(config)
        print(f"✓ 模型创建成功")
        
        # 统计参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - 总参数数: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False
    
    # 尝试加载预训练权重
    try:
        checkpoint_path = "./pretrained_model/pytorch_model.bin"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        print(f"✓ 预训练权重加载成功")
    except Exception as e:
        print(f"⚠ 预训练权重加载失败: {e}")
        print("  继续使用随机初始化权重...")
    
    # 测试前向传播
    try:
        model.eval()
        batch_size = 2
        # 创建虚拟输入数据 [batch, channels, depth, height, width]
        dummy_input = torch.randn(batch_size, config.num_channels, *config.image_size)
        dummy_labels = torch.zeros(batch_size)
        dummy_bbox = torch.randn(batch_size, 6)  # [x1, y1, z1, x2, y2, z2]
        
        print(f"  - 输入形状: {dummy_input.shape}")
        
        with torch.no_grad():
            outputs = model(
                pixel_values=dummy_input,
                labels=dummy_labels,
                bbox=dummy_bbox
            )
        
        print(f"✓ 前向传播成功")
        print(f"  - 分类输出形状: {outputs['logits'].shape}")
        print(f"  - 边界框输出形状: {outputs['bbox'].shape}")
        print(f"  - 损失值: {outputs['loss'].item():.4f}")
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False
    
    print("\n=== 测试完成 ===")
    return True

def main():
    success = test_model_loading()
    if success:
        print("✓ 模型部署验证成功！")
        print("\n使用说明:")
        print("1. 训练模型: python train.py")
        print("2. 评估模型: python eval.py") 
        print("3. 查看详细评估: jupyter notebook eval.ipynb")
    else:
        print("✗ 模型部署验证失败")
    
    return success

if __name__ == "__main__":
    main()