def get_shape(tensor):
	"""Helper function to get the shape of a tensor, without having to worry about its structure.

	Args:
			tensor (Torch tensor): Tensor

	Returns:
			string: (x,y,z,etc.)
	"""
	shapes = []
	sub_tensor = tensor
	while True:
		try:
			shapes.append(str(len(sub_tensor)))
			sub_tensor = sub_tensor[0]
		except Exception as ex:
			break
	return f"({', '.join(shapes)})"


import time
import logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)

@contextmanager
def log_time(task_name="Task"):
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"{task_name} took {elapsed_time:.4f} seconds to execute.")