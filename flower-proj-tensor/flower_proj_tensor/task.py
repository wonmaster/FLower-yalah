"""flower-proj-tensor: A Flower / TensorFlow app."""

import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import keras
from keras import layers

from medmnist import PathMNIST
from flwr.common.typing import UserConfig

# Configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
IMG_SHAPE = (28, 28, 3)
N_CLASSES = 9
DATA_CACHE = None  # cache global


def load_model(learning_rate: float = 0.001):
    """Create and compile CNN model for PathMNIST."""
    model = keras.Sequential([
        keras.Input(shape=IMG_SHAPE),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(N_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_data(partition_id: int, num_partitions: int, seed: int = 42, alpha: float = 0.5):
    """Load and partition PathMNIST with Dirichlet distribution (optimized)."""
    import gc  # pour forcer le garbage collection si besoin

    global DATA_CACHE
    if DATA_CACHE is None:
        # Utiliser dtype plus léger (float16) au lieu de float32
        train_dataset = PathMNIST(split="train", download=True)
        test_dataset = PathMNIST(split="test", download=True)

        # Cast vers float16 pour diviser la mémoire utilisée par deux (28x28x3 float16 ~ 4.4KB/img)
        x_full = train_dataset.imgs.astype(np.float32) / 255.0
        y_full = train_dataset.labels.squeeze().astype(np.uint8)

        x_test = test_dataset.imgs.astype(np.float32) / 255.0
        y_test = test_dataset.labels.squeeze().astype(np.uint8)

        DATA_CACHE = (x_full, y_full, x_test, y_test)
        del train_dataset, test_dataset
        gc.collect()  # nettoyage mémoire

    x_full, y_full, x_test, y_test = DATA_CACHE

    # Dirichlet partitioning
    np.random.seed(seed)
    n_classes = N_CLASSES

    # Utilisation plus compacte de l’indexation par classe
    idx_by_class = [np.flatnonzero(y_full == c) for c in range(n_classes)]

    client_indices = [[] for _ in range(num_partitions)]
    for c in range(n_classes):
        idx_c = idx_by_class[c]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet([alpha] * num_partitions)
        proportions = np.cumsum(proportions[:-1]) * len(idx_c)
        splits = np.split(idx_c, proportions.astype(int))
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    part_idx = client_indices[partition_id]
    x_train, y_train = x_full[part_idx], y_full[part_idx]

    return x_train, y_train, x_test, y_test



def create_run_dir(config: UserConfig) -> tuple[Path, str]:
    """Create directory to save experiment results."""
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    with open(save_path / "run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir
