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




def write_tensor_to_file(tensor, file_path):
    """
    Writes a PyTorch tensor to a file in a format that matches the expected input
    of the C++ function. The tensor is written as a space-separated matrix with each
    row on a new line.

    Parameters:
    - tensor: The PyTorch tensor to be written to the file.
    - file_path: The path to the file where the tensor will be written.
    """
    # Ensure the tensor is on CPU and convert it to a numpy array
    tensor_np = tensor.cpu().numpy()

    # Open the file for writing
    with open(file_path, 'w') as f:
        # Iterate over the rows of the tensor
        for row in tensor_np:
            # Convert each row to a string, with values separated by spaces
            row_str = ' '.join(map(str, row))
            # Write the row to the file, followed by a newline character
            f.write(f"{row_str}\n")


def read_tensor_from_file(file_path):
    """
    Reads a tensor from a file, assuming each line in the file represents a row of the tensor,
    with values being space-separated.

    Parameters:
    - file_path: The path to the file containing the tensor data.

    Returns:
    - A PyTorch tensor constructed from the file contents.
    """
    tensor_list = []
    with open(file_path, 'r') as file:
        for line in file:
            # Convert each line to a list of floats
            row = list(map(float, line.strip().split()))
            tensor_list.append(row)

    # Convert the list of lists to a PyTorch tensor
    tensor = torch.tensor(tensor_list)
    return tensor

# # Example usage
# if __name__ == "__main__":
#      # Create a sample tensor
#     sample_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)

#     # Specify the file path
#     file_path = 'tensor_output.txt'

#     # Write the tensor to the file
#     write_tensor_to_file(sample_tensor, file_path)
    
#     tensor = read_tensor_from_file(file_path)
#     print(tensor)


