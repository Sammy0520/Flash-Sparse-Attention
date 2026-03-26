import torch

from impl.linear_compress_decode import _linear_compress_decode
from impl.linear_compress_decode_fp8 import _linear_compress_decode_fp8

def test_fp8_linear_compress():
    # 1. 参数设置
    device = "cuda"
    dtype = torch.bfloat16
    K = 1024
    num_heads = 8
    head_dim = 128
    kernel_size = 32
    kernel_stride = 16
    
    # 2. 生成随机测试数据
    torch.manual_seed(42)
    new_tokens = torch.randn(K, num_heads, head_dim, device=device, dtype=dtype)
    compress_weight = torch.randn(num_heads, head_dim * kernel_size, head_dim, device=device, dtype=dtype)
    intra_block_pe = torch.randn(num_heads, kernel_size, head_dim, device=device, dtype=dtype)
    
    print(f"Testing with K={K}, heads={num_heads}, dim={head_dim}")

    # 3. 运行 Baseline (BF16)
    # 假设 baseline 返回的是 BF16 tensor
    with torch.no_grad():
        y_baseline = _linear_compress_decode(
            new_tokens,
            compress_weight,
            kernel_size,
            kernel_stride,
            intra_block_pe=intra_block_pe,
            prev_total_len=0,
            token_buffer=None
        )

    # 4. 运行 FP8 算子
    # 返回 (y_fp8: [N, H, D], scale: [N, H])
    with torch.no_grad():
        y_fp8, scale = _linear_compress_decode_fp8(
            new_tokens,
            compress_weight,
            kernel_size,
            kernel_stride,
            intra_block_pe=intra_block_pe,
            prev_total_len=0,
            token_buffer=None
        )

    # 5. 反量化逻辑
    # y_dequant = y_fp8 * scale
    # 注意：scale 是 [N, H]，y_fp8 是 [N, H, D]，需要 unsqueeze 最后一维进行广播
    y_dequant = y_fp8.to(dtype) * scale.unsqueeze(-1)

    # 6. 精度对比
    # 计算平均绝对误差 (MAE) 和余弦相似度
    mae = torch.mean(torch.abs(y_baseline - y_dequant)).item()
    
    # 计算余弦相似度 (越高越好，1.0 为完美对齐)
    cos_sim = torch.nn.functional.cosine_similarity(
        y_baseline.flatten(), 
        y_dequant.flatten(), 
        dim=0
    ).item()

    # 计算信噪比 (SNR) 也是衡量量化损失的好指标
    snr = 20 * torch.log10(torch.norm(y_baseline) / torch.norm(y_baseline - y_dequant)).item()

    print("-" * 30)
    print(f"MAE: {mae:.6f}")
    print(f"Cosine Similarity: {cos_sim:.6f}")
    print(f"SNR: {snr:.2f} dB")
    
    # 7. 判定
    if cos_sim > 0.99:
        print("✅ Test Passed: FP8 results are highly aligned with BF16.")
    else:
        print("❌ Test Failed: Precision loss is too significant.")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_fp8_linear_compress()
    else:
        print("CUDA not available.")
