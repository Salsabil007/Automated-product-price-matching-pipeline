import json
import os
from argparse import ArgumentParser
from collections import defaultdict, deque
from functools import reduce
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from torch import nn, relu
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import TransformerConv
from torch_geometric.transforms import ToUndirected
from tqdm import tqdm


def reverse_dict(label_set):
    reversed_dict = {}
    for key, values in label_set.items():
        for value in values:
            if value not in reversed_dict:
                reversed_dict[value] = []
            reversed_dict[value].append(key)
    return reversed_dict


def find_connected_components(label_set):
    graph = defaultdict(set)
    for key, values in label_set.items():
        for value in values:
            graph[value].add(key)
    visited = set()
    groups = []

    def bfs(start):
        queue = deque([start])
        group = set()
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                group.add(node)
                # Add all neighbors
                for value in label_set[node]:
                    queue.extend(graph[value])
        return group

    for key in label_set:
        if key not in visited:
            groups.append(bfs(key))

    return [set(map(int, s)) for s in groups]


def get_connected_groups(label_set):
    product_groups = find_connected_components(label_set)
    tag_groups = find_connected_components(reverse_dict(label_set))
    return product_groups, tag_groups


def create_adj_matrices_from_label_json(label_json_filepath: str):
    """
    {
        "len": 11,
        "associations": {
            0: [7, 9],
            1: [7, 9],
            2: [7, 9],
            3: [8, 10],
            4: [8, 10],
            5: [8, 10],
            6: [8, 10],
        }
    }
        :param label_json_filepath:
        :return:
    """
    label: dict[
        Literal["len", "associations", "products", "tags"], int | dict[int, list[int]]
    ] = json.load(open(label_json_filepath))
    associations = label["associations"]
    del label["associations"]
    label["associations"] = {
        int(k): list(map(lambda _v: int(_v), vs)) for k, vs in associations.items()
    }
    gt_assoc = np.zeros(shape=(label["len"], label["len"]))
    prod_to_prod = np.zeros(shape=(label["len"], label["len"]))
    price_to_price = np.zeros(shape=(label["len"], label["len"]))
    # generate ground truth set
    for product_idx, price_tag_idxs in label["associations"].items():
        gt_assoc[product_idx, price_tag_idxs] = 1

    # generate ground prod_to_prod connections
    if "products" not in label:
        product_groups, tag_groups = get_connected_groups(label["associations"])
        for product_set in product_groups:
            for i in product_set:
                for j in product_set:
                    prod_to_prod[i, j] = 1

        for tag_set in tag_groups:
            for i in tag_set:
                for j in tag_set:
                    price_to_price[i, j] = 1
    else:
        for product_idx in label["products"]:
            others = set(label["products"]) - {product_idx}
            prod_to_prod[product_idx, list(others)] = 1
        for tag_idx in label["tags"]:
            others = set(label["tags"]) - {tag_idx}
            price_to_price[tag_idx, list(others)] = 1

    return {
        "ground_truth": torch.Tensor(gt_assoc),
        "prod_to_prod": torch.Tensor(prod_to_prod),
        "price_to_price": torch.Tensor(price_to_price),
    }


def get_positive_negative_edges(
    batch: Batch, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Given a batch of graphs (represented as a mega-graph with a block diagonal adjacency matrix), sample graph-wise positive/negative edges
    such that we specify examples of actual and fake product-price and price-price connections.

    Args:1
        batch (Batch): A batch of data from a HomogeneousPriceAttributionScenes dataset.
        batch_size (int): The number of graphs in the batch.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tensors specifying positive/negative edges in the mega-graph.
    """
    positive_edges = []
    negative_edges = []
    for batch_idx in range(batch_size):
        batch_indices = torch.where(batch.batch == batch_idx)[0]
        if batch_indices.numel() == 0:
            continue
        indices_of_products_in_batch = torch.where(batch.x[batch_indices, -1] == 1)[0]
        indices_of_prices_in_batch = torch.where(batch.x[batch_indices, -1] == 0)[0]
        positive_edges_in_batch: torch.Tensor = batch.gt_edge_index[
            :, batch.gt_edge_index[0] <= batch_indices.max()
        ]
        all_possible_prod_price = torch.cartesian_prod(
            indices_of_products_in_batch, indices_of_prices_in_batch
        ).T
        all_possible_price_price = torch.cartesian_prod(
            indices_of_prices_in_batch, indices_of_prices_in_batch
        ).T

        all_possible_edges = torch.cat(
            [all_possible_prod_price, all_possible_price_price], dim=-1
        )
        negative_edges_in_batch = index_pair_set_diff(
            all_possible_edges, positive_edges_in_batch
        )

        k = min(positive_edges_in_batch.size(1), negative_edges_in_batch.size(1))
        if k == 0:
            continue
        positive_edges_in_batch = positive_edges_in_batch[
            :,
            torch.multinomial(
                torch.ones(positive_edges_in_batch.size(1)), num_samples=k
            ),
        ]
        negative_edges_in_batch = negative_edges_in_batch[
            :,
            torch.multinomial(
                torch.ones(negative_edges_in_batch.size(1)), num_samples=k
            ),
        ]

        positive_edges.append(positive_edges_in_batch)
        negative_edges.append(negative_edges_in_batch)

    positive_edges = torch.cat(positive_edges, dim=-1)
    negative_edges = torch.cat(negative_edges, dim=-1)

    return positive_edges, negative_edges


def index_pair_set_diff(
    edge_index_a: torch.Tensor, edge_index_b: torch.Tensor
) -> torch.Tensor:
    set_a = set(map(tuple, edge_index_a.T.tolist()))
    set_b = set(map(tuple, edge_index_b.T.tolist()))
    unique_pairs = set_a - set_b

    if unique_pairs:
        unique_tensor = torch.tensor(list(unique_pairs)).T
    else:
        unique_tensor = torch.empty((2, 0), dtype=edge_index_a.dtype)

    return unique_tensor


class HomogeneousGraph(InMemoryDataset):
    def __init__(
        self,
        root: str,
        run: Literal["train", "val"] = "train",
        transform: Callable | None = None,
        **kwargs,
    ) -> None:
        self.run = run
        super().__init__(root, transform=transform, **kwargs)
        self.load(self.processed_paths[0])

    @property
    def dataset_root(self):
        return "./gold/association_labels"

    @property
    def raw_file_names(self):
        return os.listdir(os.path.join(self.dataset_root, "json", self.run))

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @staticmethod
    def extract_embeddings(label_file: str) -> torch.Tensor:
        lines = list(map(str.strip, open(label_file).readlines()))
        tensor = []
        for line in lines:
            cls, ctr_x, ctr_y, width, height = line.split()
            is_product = int(cls) > 0
            tensor.append(
                [float(ctr_x), float(ctr_y), float(width), float(height), is_product]
            )
        return torch.Tensor(tensor)

    def process(self) -> None:
        json_files = list(
            filter(
                lambda f: f.endswith(".json"),
                os.listdir(os.path.join(self.dataset_root, "json", self.run)),
            )
        )
        dataset: list[Data] = []
        for json_file in json_files:
            data = Data()
            img_id = json_file.replace(".json", "")
            matrices = create_adj_matrices_from_label_json(
                os.path.join(self.dataset_root, "json", self.run, json_file),
            )
            node_embeddings = self.extract_embeddings(
                os.path.join(self.dataset_root, "labels", f"{img_id}.txt"),
            )
            gt_prod_prod = matrices["prod_to_prod"]
            gt_price_price = matrices["price_to_price"]
            gt_adj_matrix = matrices["ground_truth"]

            product_indices = torch.where(node_embeddings[:, -1] == 1)[0]
            price_indices = torch.where(node_embeddings[:, -1] == 0)[0]
            prod_prod__coords = torch.nonzero(gt_prod_prod).T
            price_price__coords = torch.cartesian_prod(price_indices, price_indices).T
            prod_price__coords = torch.cartesian_prod(product_indices, price_indices).T

            data.x = node_embeddings
            data.edge_index = torch.cat(
                [price_price__coords, prod_prod__coords, prod_price__coords], dim=1
            )

            actual_price_price_coords = torch.nonzero(gt_price_price).T
            actual_prod_price_coords = torch.nonzero(gt_adj_matrix).T
            data.gt_edge_index = torch.cat(
                [actual_price_price_coords, actual_prod_price_coords], dim=1
            )

            data.product_indices = product_indices
            data.price_indices = price_indices
            dataset.append(data)

        self.save(dataset, self.processed_paths[0])


class HomogeneousGNN(nn.Module):
    def __init__(self, hidden_channels: int, num_layers: int):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = TransformerConv((-1, -1), hidden_channels)
            self.convs.append(conv)

        self.link_predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = relu(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)
        return x

    def decode(
        self, z: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
    ) -> torch.Tensor:
        z_src = z[src]
        z_dst = z[dst]

        return self.link_predictor(torch.cat([z_src, z_dst], dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        return_encoding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict the link probability for each pair indexed by src/dst.

        Args:
            x (torch.Tensor): Node embeddings, with shape (n, d).
            edge_index (torch.Tensor): Edge indices specifying the adjacency matrix for the graph being predicted on, with shape (2, num_edges).
            src (torch.Tensor): Indices of source nodes in the pairs we are predicting links between, with shape (num_links_to_predict,).
            dst (torch.Tensor): Indices of destination nodes in the pairs we are predicting links between, with shape (num_links_to_predict,).
            return_encoding (bool, optional): Whether/not to return the (n, self.hidden_channels) encoding of the graph along with link probs. Defaults to `False`.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: The link probabilities, with shape (num_links_to_predict,)
             for the specified node pairs.
             If `return_encoding` is `True`, then we also return the (n, self.hidden_channels) encoding of the graph.
        """
        z = self.encode(x, edge_index)
        output = self.decode(z, src, dst)
        if return_encoding:
            return output, z
        else:
            return output


def train_epoch(
    model: HomogeneousGNN, loader: DataLoader, optimizer: torch.optim.Optimizer
) -> tuple[float, float, float, float]:
    """Train a link predictor for one epoch.

    Args:
        model (HomogeneousGNN): The model to train.
        loader (DataLoader): DataLoader with training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.

    Returns:
        tuple[float, float, float, float]: Avg. loss / batch over the epoch, along with precision, recall, and accuracy.
    """
    model.train()
    total_loss = 0
    tp, fp, tn, fn = 0, 0, 0, 0

    for batch in loader:
        optimizer.zero_grad()
        positive_edges, negative_edges = get_positive_negative_edges(
            batch, loader.batch_size
        )

        pos_pred: torch.Tensor = model(
            x=batch.x,
            edge_index=batch.edge_index,
            src=positive_edges[0],
            dst=positive_edges[1],
        )
        neg_pred: torch.Tensor = model(
            x=batch.x,
            edge_index=batch.edge_index,
            src=negative_edges[0],
            dst=negative_edges[1],
        )

        pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
        loss = pos_loss + neg_loss

        tp += (pos_pred.round() == 1).sum().item()
        fp += (neg_pred.round() == 1).sum().item()
        tn += (neg_pred.round() == 0).sum().item()
        fn += (pos_pred.round() == 0).sum().item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn) if tp + fp + fn + tn != 0 else 0

    return avg_loss, precision, recall, acc


@torch.inference_mode()
def eval_model(
    model: HomogeneousGNN, loader: DataLoader
) -> tuple[float, float, float, float]:
    """Evaluate a link predictor.

    Args:
        model (HomogeneousGNN): The model to train.
        loader (DataLoader): DataLoader with evaluation data.

    Returns:
        tuple[float, float, float, float]: Avg. loss / batch over the eval data, along with precision, recall, and accuracy.
    """
    model.eval()

    total_loss = 0
    tp, fp, tn, fn = 0, 0, 0, 0

    for batch in loader:
        positive_edges, negative_edges = get_positive_negative_edges(
            batch, loader.batch_size
        )

        pos_pred: torch.Tensor = model(
            x=batch.x,
            edge_index=batch.edge_index,
            src=positive_edges[0],
            dst=positive_edges[1],
        )
        neg_pred: torch.Tensor = model(
            x=batch.x,
            edge_index=batch.edge_index,
            src=negative_edges[0],
            dst=negative_edges[1],
        )

        pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
        loss = pos_loss + neg_loss

        tp += (pos_pred.round() == 1).sum().item()
        fp += (neg_pred.round() == 1).sum().item()
        tn += (neg_pred.round() == 0).sum().item()
        fn += (pos_pred.round() == 0).sum().item()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn) if tp + fp + fn + tn != 0 else 0

    return avg_loss, precision, recall, acc


def train(num_epochs: int, chkp_dir: Path):
    if not chkp_dir.exists():
        os.makedirs(chkp_dir)

    train_dataset = HomogeneousGraph(
        root="./",
        transform=ToUndirected(),
        force_reload=True,
    )
    eval_dataset = HomogeneousGraph(
        root="./",
        run="val",
        transform=ToUndirected(),
        force_reload=True,
    )
    model = HomogeneousGNN(hidden_channels=64, num_layers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

    best_f1 = 0
    for _ in tqdm(range(num_epochs)):
        train_loss, train_precision, train_recall, train_acc = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
        )
        eval_loss, eval_precision, eval_recall, eval_acc = eval_model(
            model, eval_loader
        )
        f1 = 2 * (eval_precision * eval_recall) / (eval_precision + eval_recall) if eval_precision + eval_recall != 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                obj={
                    "state_dict": model.state_dict(),
                    "f1": best_f1,
                    "precision": train_precision,
                    "recall": train_recall,
                    "val_loss": eval_loss,
                },
                f=chkp_dir / "best.pt",
            )
        print(
            f"Train | Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, Acc: {train_acc:.4f}"
        )
        print(
            f"Eval | Loss: {eval_loss:.4f}, Precision: {eval_precision:.4f}, Recall: {eval_recall:.4f}, Acc: {eval_acc:.4f}"
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num-epochs", default=100)
    parser.add_argument("--chkp-dir", default=Path("chkp"))
    args = parser.parse_args()
    train(num_epochs=args.num_epochs, chkp_dir=args.chkp_dir)
