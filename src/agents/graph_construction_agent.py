import torch
from typing import List, Dict, Tuple

class GraphConstructionAgent:
    """
    An agent responsible for constructing a graph of stock relationships.
    """

    def __init__(self, stock_metadata: Dict[str, Dict]):
        """
        Initializes the GraphConstructionAgent.

        Args:
            stock_metadata (Dict[str, Dict]): A dictionary containing metadata for each stock,
                                              such as its industry sector.
        """
        self.stock_metadata = stock_metadata
        self.symbol_to_idx = {symbol: i for i, symbol in enumerate(stock_metadata.keys())}

    def create_graph(self, symbols: List[str]) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Creates a graph where stocks in the same sector are connected.

        Args:
            symbols (List[str]): A list of stock symbols to include in the graph.

        Returns:
            Tuple[torch.Tensor, Dict[str, int]]: A tuple containing:
                - edge_index (torch.Tensor): A tensor representing the graph's edge index.
                - symbol_to_idx (Dict[str, int]): A mapping from stock symbols to their graph indices.
        """
        edges = []
        num_nodes = len(symbols)
        
        # Update symbol_to_idx to include all symbols passed
        symbol_to_idx = {symbol: i for i, symbol in enumerate(symbols)}

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                symbol_i = symbols[i]
                symbol_j = symbols[j]

                # Connect stocks if they are in the same sector
                if self.stock_metadata[symbol_i]['sector'] == self.stock_metadata[symbol_j]['sector']:
                    edges.append([symbol_to_idx[symbol_i], symbol_to_idx[symbol_j]])
                    edges.append([symbol_to_idx[symbol_j], symbol_to_idx[symbol_i]]) # Add reverse edge for undirected graph

        if not edges:
            # If no edges are created (e.g., all stocks in different sectors),
            # return an empty edge index.
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return edge_index, symbol_to_idx
