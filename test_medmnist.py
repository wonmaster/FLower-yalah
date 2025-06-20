from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from flower_proj_tensor.task import load_data, load_model

num_partitions = Context.node_config["num-partitions"]

print(f"Number of partitions: {num_partitions}")