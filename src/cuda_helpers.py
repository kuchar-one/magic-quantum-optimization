import torch
import time
import gc
import os
from functools import wraps

class DeviceMonitor:
    """Utility class to monitor tensor device locations and operations"""
    
    @staticmethod
    def get_tensor_device(tensor):
        """
        Get the device of a tensor in a human-readable format
        
        Args:
            tensor (torch.Tensor): The tensor to check the device of.
        
        Returns:
            str: The device of the tensor or "not a tensor" if the input is not a tensor.
        """
        if not isinstance(tensor, torch.Tensor):
            return "not a tensor"
        return str(tensor.device)
    
    @staticmethod
    def get_current_device():
        """
        Get the current CUDA device if available
        
        Returns:
            str: The current CUDA device or "cpu" if CUDA is not available.
        """
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"
    
    @staticmethod
    def print_cuda_info():
        """
        Print detailed CUDA device information
        
        Prints:
            Detailed information about the current CUDA device if available, 
            otherwise indicates that no CUDA device is available.
        """
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            device_props = torch.cuda.get_device_properties(current_device)
            print(f"\nCUDA Device Information:")
            print(f"  Current device: {current_device}")
            print(f"  Device name: {device_props.name}")
            print(f"  Total memory: {device_props.total_memory / 1024**2:.2f} MB")
            print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"  Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        else:
            print("\nNo CUDA device available. Running on CPU.")
    
    @staticmethod
    def device_trace(func):
        """
        Decorator to trace tensor device locations in functions
        
        Args:
            func (callable): The function to be decorated.
        
        Returns:
            callable: The wrapped function with device tracing.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"\nTracing device locations in {func.__name__}:")
            print(f"  Current device: {DeviceMonitor.get_current_device()}")
            
            # Track tensor arguments
            for i, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    print(f"  Arg {i} device: {DeviceMonitor.get_tensor_device(arg)}")
            
            for name, arg in kwargs.items():
                if isinstance(arg, torch.Tensor):
                    print(f"  Kwarg {name} device: {DeviceMonitor.get_tensor_device(arg)}")
            
            # Time the function execution
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # Track result device
            if isinstance(result, torch.Tensor):
                print(f"  Result device: {DeviceMonitor.get_tensor_device(result)}")
            elif isinstance(result, (tuple, list)):
                for i, item in enumerate(result):
                    if isinstance(item, torch.Tensor):
                        print(f"  Result[{i}] device: {DeviceMonitor.get_tensor_device(item)}")
            
            print(f"  Execution time: {end_time - start_time:.4f} seconds")
            return result
        return wrapper


def set_gpu_device(device_id):
    """
    Set the GPU device for computation

    Args:
        device_id (int): Integer ID of the GPU device to use.
    
    Raises:
        ValueError: If the specified device_id is not available.
        RuntimeError: If no CUDA-capable GPU devices are found.
    """
    if torch.cuda.is_available():
        if device_id >= torch.cuda.device_count():
            raise ValueError(f"GPU device {device_id} not found. Available devices: 0-{torch.cuda.device_count()-1}")
        torch.cuda.set_device(device_id)
        print(f"Using GPU device {device_id}: {torch.cuda.get_device_name(device_id)}")
    else:
        raise RuntimeError("No CUDA-capable GPU devices found")

class GPUDeviceWrapper:
    """
    Wrapper class that redirects tensor operations to a specific device
    
    Args:
        wrapped_instance (object): The instance to wrap.
        device_id (int): The ID of the GPU device to use.
    """
    def __init__(self, wrapped_instance, device_id):
        self._wrapped = wrapped_instance
        self.device = f'cuda:{device_id}'
        
        # Force the base class's tensors to the correct device
        self._wrapped.device = self.device
        self._wrapped.d = self._wrapped.d.to(self.device)
        self._wrapped.identity = self._wrapped.identity.to(self.device)
        
    def __getattr__(self, name):
        """
        Redirect attribute access to the wrapped instance, ensuring tensor operations are on the correct device.
        
        Args:
            name (str): The name of the attribute to access.
        
        Returns:
            The attribute from the wrapped instance, with tensor operations redirected to the correct device.
        """
        attr = getattr(self._wrapped, name)
        if callable(attr):
            def wrapped_method(*args, **kwargs):
                new_args = [arg.to(self.device) if torch.is_tensor(arg) else arg for arg in args]
                new_kwargs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in kwargs.items()}
                result = attr(*new_args, **new_kwargs)
                if torch.is_tensor(result):
                    return result.to(self.device)
                elif isinstance(result, (list, tuple)):
                    return type(result)(x.to(self.device) if torch.is_tensor(x) else x for x in result)
                return result
            return wrapped_method
        elif torch.is_tensor(attr):
            return attr.to(self.device)
        return attr


def aggressive_memory_cleanup():
    """
    Perform aggressive GPU memory cleanup
    
    This function:
    1. Runs Python garbage collection
    2. Clears PyTorch CUDA cache
    3. Synchronizes CUDA operations
    4. Forces memory deallocation
    5. Sets memory allocation configuration for better fragmentation handling
    6. Forces tensor deletion and memory reclamation
    """
    if torch.cuda.is_available():
        # Set memory allocation configuration to reduce fragmentation
        try:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        except:
            pass
        
        # Force garbage collection (reduced rounds for better performance)
        for _ in range(2):  # Reduced from 5 to 2
            gc.collect()
        
        # Clear PyTorch CUDA cache (reduced rounds for better performance)
        for _ in range(2):  # Reduced from 5 to 2
            torch.cuda.empty_cache()
        
        # Synchronize all CUDA operations to ensure cleanup is complete
        torch.cuda.synchronize()
        
        # Try to reclaim fragmented memory
        try:
            # Force memory compaction if available
            torch.cuda.ipc_collect()
        except:
            # Some CUDA versions don't support ipc_collect, ignore
            pass
        
        # Additional cleanup: reset peak memory stats
        try:
            torch.cuda.reset_peak_memory_stats()
        except:
            pass
        
        # Note: Removed memory reclamation trick to improve performance


def get_memory_usage():
    """
    Get current GPU memory usage information
    
    Returns:
        dict: Dictionary with memory usage statistics
    """
    if not torch.cuda.is_available():
        return {"allocated": 0, "cached": 0, "total": 0, "free": 0}
    
    allocated = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    
    return {
        "allocated": allocated / 1024**2,  # MB
        "cached": cached / 1024**2,        # MB
        "total": total / 1024**2,          # MB
        "free": (total - allocated) / 1024**2  # MB
    }


def print_memory_status(label=""):
    """
    Print current memory status
    
    Args:
        label (str): Optional label for the memory status
    """
    if torch.cuda.is_available():
        mem_info = get_memory_usage()
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}GPU Memory: {mem_info['allocated']:.1f}MB allocated, "
              f"{mem_info['cached']:.1f}MB cached, {mem_info['free']:.1f}MB free")


def check_memory_and_cleanup(memory_threshold_mb=1000, label=""):
    """
    Check GPU memory usage and perform cleanup if needed
    
    Args:
        memory_threshold_mb (float): Memory threshold in MB to trigger cleanup
        label (str): Optional label for logging
    
    Returns:
        bool: True if cleanup was performed, False otherwise
    """
    if not torch.cuda.is_available():
        return False
    
    mem_info = get_memory_usage()
    allocated_mb = mem_info['allocated']
    
    if allocated_mb > memory_threshold_mb:
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}High memory usage detected ({allocated_mb:.1f}MB > {memory_threshold_mb}MB), performing cleanup...")
        aggressive_memory_cleanup()
        
        # Check memory after cleanup
        mem_info_after = get_memory_usage()
        freed_mb = allocated_mb - mem_info_after['allocated']
        print(f"{prefix}Cleanup completed. Freed {freed_mb:.1f}MB. Current usage: {mem_info_after['allocated']:.1f}MB")
        return True
    
    return False


def memory_cleanup_context():
    """
    Context manager for automatic memory cleanup
    """
    class MemoryCleanupContext:
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            aggressive_memory_cleanup()
    
    return MemoryCleanupContext()