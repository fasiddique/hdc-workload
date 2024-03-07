import sys
import time
import torch


def trace_calls(frame, event, arg):
    if event == 'call':
        # Capture the current time to measure execution duration
        frame.f_locals['start_time'] = time.time()
        
        # Extracting function name, file name, and line number
        code = frame.f_code
        func_name = code.co_name
        file_name = code.co_filename
        line_no = frame.f_lineno
        
        # Logging function call details
        print(f"Call to function: {func_name}() in {file_name}:{line_no}")
        
        # Optionally, you can inspect frame.f_locals to log function arguments
        
    elif event == 'return':
        # Compute execution duration
        duration = time.time() - frame.f_locals['start_time']
        
        # Extracting function name
        code = frame.f_code
        func_name = code.co_name
        
        # Logging return value and execution time
        print(f"Function {func_name}() returned {arg})")
        
    return trace_calls

def monitor_function_calls(func):
    """A decorator to apply tracing to a function."""
    def wrapper(*args, **kwargs):
        sys.settrace(trace_calls)  # Start tracing
        result = func(*args, **kwargs)
        sys.settrace(None)  # Stop tracing
        return result
    return wrapper

# Example usage
@monitor_function_calls
def test_function(x, y):
    return x + y




def aggregate_tensor_mod(tensor):
    return torch.sum(tensor) % 10000