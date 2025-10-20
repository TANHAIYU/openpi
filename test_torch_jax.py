import sys

def check_pytorch():
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        
        # 验证张量创建和基本运算
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        z = x + y
        print("PyTorch 张量加法结果:", z)
        
        # 验证矩阵乘法
        a = torch.tensor([[1, 2], [3, 4]])
        b = torch.tensor([[5, 6], [7, 8]])
        c = torch.matmul(a, b)
        print("PyTorch 矩阵乘法结果:\n", c)
        
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            print(f"CUDA 可用，设备数: {torch.cuda.device_count()}")
            print(f"当前设备: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("CUDA 不可用，使用CPU计算")
            
        print("PyTorch 验证成功!\n")
        return True
        
    except ImportError:
        print("PyTorch 未安装或无法导入")
        return False
    except Exception as e:
        print(f"PyTorch 验证出错: {str(e)}")
        return False

def check_jax():
    try:
        import jax
        import jax.numpy as jnp
        print(f"JAX 版本: {jax.__version__}")
        
        # 验证张量创建和基本运算
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])
        z = x + y
        print("JAX 数组加法结果:", z)
        
        # 验证矩阵乘法
        a = jnp.array([[1, 2], [3, 4]])
        b = jnp.array([[5, 6], [7, 8]])
        c = jnp.matmul(a, b)
        print("JAX 矩阵乘法结果:\n", c)
        
        # 检查GPU是否可用
        if jax.default_backend() == "gpu":
            print(f"GPU 可用，设备数: {jax.device_count()}")
        else:
            print(f"当前后端: {jax.default_backend()}")
            
        print("JAX 验证成功!")
        return True
        
    except ImportError:
        print("JAX 未安装或无法导入")
        return False
    except Exception as e:
        print(f"JAX 验证出错: {str(e)}")
        return False

def main():
    print(f"Python 版本: {sys.version}")
    print("="*50)
    
    # 检查PyTorch
    torch_success = check_pytorch()
    
    # 检查JAX
    jax_success = check_jax()
    
    print("="*50)
    if torch_success and jax_success:
        print("所有库验证成功!")
    else:
        print("部分库验证失败，请检查安装情况")

if __name__ == "__main__":
    main()
