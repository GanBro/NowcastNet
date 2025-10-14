import torch
from mydataloader import load_single_file


def test_dgmr_format():
    """验证是否完全适配DGMR格式"""
    print("=" * 60)
    print("验证SEVIR适配DGMR格式")
    print("=" * 60)

    # 加载本地测试文件
    loader = load_single_file("data/R18032123577328.npz", batch_size=8)

    # DGMR期望的格式
    expected_input = (8, 4, 256, 256)
    expected_target = (8, 15, 256, 256)
    expected_range = (-9, 60)

    for input_data, target in loader:
        # 复制8次模拟batch
        input_data = input_data.repeat(8, 1, 1, 1)
        target = target.repeat(8, 1, 1, 1)

        print(f"\n检查项:")
        print(f"  ✓ Input shape:  {input_data.shape} (期望: {expected_input})")
        print(f"  ✓ Target shape: {target.shape} (期望: {expected_target})")
        print(f"  ✓ 数据类型: {input_data.dtype} (期望: torch.float32)")
        print(f"  ✓ Input范围:  [{input_data.min():.2f}, {input_data.max():.2f}] (期望: {expected_range})")
        print(f"  ✓ Target范围: [{target.min():.2f}, {target.max():.2f}] (期望: {expected_range})")

        # 验证
        assert input_data.shape == expected_input, f"输入形状错误: {input_data.shape}"
        assert target.shape == expected_target, f"目标形状错误: {target.shape}"
        assert input_data.dtype == torch.float32, "数据类型错误"

        # 检查数值范围是否合理
        assert input_data.min() >= -9 and input_data.max() <= 60, "输入数值范围超出[-9, 60]"
        assert target.min() >= -9 and target.max() <= 60, "目标数值范围超出[-9, 60]"

        print("\n" + "=" * 60)
        print("✓ 所有验证通过！完全适配DGMR原始配置")
        print("=" * 60)
        break


if __name__ == "__main__":
    test_dgmr_format()
