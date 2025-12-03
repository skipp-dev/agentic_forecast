import torch
import sys

def verify_cuda_operations():
    print("--- PyTorch/CUDA Verification ---")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        return False

    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    try:
        print("\nAttempting tensor allocation on GPU...")
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        print("Allocation successful.")
        
        print("Attempting matrix multiplication on GPU...")
        z = torch.matmul(x, y)
        print("Matrix multiplication successful.")
        
        print("Moving result back to CPU...")
        z_cpu = z.cpu()
        print("Move to CPU successful.")
        
        print("\nSUCCESS: PyTorch/CUDA environment is fully functional.")
        return True
        
    except Exception as e:
        print(f"\nERROR: CUDA operation failed: {e}")
        return False

if __name__ == "__main__":
    success = verify_cuda_operations()
    sys.exit(0 if success else 1)
