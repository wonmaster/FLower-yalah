"""flower-proj-tensor: A Flower / TensorFlow app."""
"""tensorflow-example: A Flower / TensorFlow app."""

from flower_proj_tensor.strategy import CustomFedAvg
from flower_proj_tensor.task import load_model
from medmnist import PathMNIST
import numpy as np
from datasets import load_dataset
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig


def gen_evaluate_fn(
    x_test,
    y_test,
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        model = load_model()
        model.set_weights(parameters_ndarrays)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return loss, {"centralized_accuracy": accuracy}

    return evaluate

def process_fit_metrics(metrics:list[tuple[int,Metrics]]) -> Metrics:
    for _,m in metrics:
        print(m)
    return {}

def on_fit_config(server_round: int)-> Metrics:
    """Construct `config` that clients receive when running `fit()`"""
    lr = 0.002
    # Enable a simple form of learning rate decay
    if server_round > 10:
        lr /= 2
    return {"lr": lr}


# Define metric aggregation function
def weighted_average(metrics:list[tuple[int,Metrics]]) -> Metrics:
    """_summary_
    Aggregate metrics from clients using a weighted average.

    Args:
        metrics : list of tuples
        Each tuple contains the number of examples used by the client and a dictionary
        with the metrics computed by the client, e.g. (num_examples, {"accuracy": accuracy}).

    Returns:
        _type_: 
        A dictionary with the aggregated metric.
        The metric is a weighted average of the accuracies of each client,
        weighted by the number of examples used by each client.
        The metric is named "federated_evaluate_accuracy".
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    """_summary_
    Server function that initializes the server components for the Flower app.

    Args:
        context (Context): Context object containing run configuration and other parameters.

    Returns:
        _type_: ServerAppComponents
        The components of the server app, including the strategy and configuration.
    """
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]

    # Initialize model parameters
    ndarrays = load_model().get_weights()
    parameters = ndarrays_to_parameters(ndarrays)

    # Prepare dataset for central evaluation

    # This is the exact same dataset as the one downloaded by the clients via
    # FlowerDatasets. However, we don't use FlowerDatasets for the server since
    # partitioning is not needed.
    # We make use of the "test" split only
    test_dataset = PathMNIST(split="test", download=True)

    x_test = test_dataset.imgs.astype(np.float32) / 255.0
    y_test = test_dataset.labels.squeeze().astype(np.int64)
    


    # Define strategy
    strategy = CustomFedAvg(
        run_config=context.run_config,
        use_wandb=context.run_config["use-wandb"],
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_eval,
        initial_parameters=parameters,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=gen_evaluate_fn(x_test, y_test),
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=process_fit_metrics,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)