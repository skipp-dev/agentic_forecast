"""
Graph Construction Agent

Creates relationship graphs between financial symbols for GNN-based forecasting.
Builds correlation-based graphs and manages symbol relationships.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import torch
import logging

logger = logging.getLogger(__name__)

class GraphConstructionAgent:
    """
    Agent responsible for constructing relationship graphs between financial symbols.

    This agent analyzes correlations between symbols and creates graph structures
    suitable for Graph Neural Network (GNN) models.
    """

    def __init__(self, stock_metadata: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize the graph construction agent.

        Args:
            stock_metadata: Dictionary mapping symbols to metadata (sector, etc.)
        """
        self.stock_metadata = stock_metadata or {}
        logger.info("Graph Construction Agent initialized")

    def create_graph(self, symbols: List[str], raw_data: Dict[str, Any],
                    corr_threshold: float = 0.5, top_k: int = 5) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Create a relationship graph from symbol data.

        Args:
            symbols: List of stock symbols
            raw_data: Raw data dictionary with symbol data
            corr_threshold: Minimum correlation threshold for edges
            top_k: Maximum number of edges per node

        Returns:
            Tuple of (edge_index, symbol_to_idx)
        """
        logger.info(f"Creating graph for {len(symbols)} symbols")

        # Create symbol to index mapping
        symbol_to_idx = {symbol: i for i, symbol in enumerate(symbols)}

        # Calculate correlations between symbols
        correlations = self._calculate_symbol_correlations(symbols, raw_data)

        # Create edges based on correlations
        edges = self._create_edges_from_correlations(
            correlations, symbol_to_idx, corr_threshold, top_k
        )

        # Convert to PyTorch tensor
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            # Create fully connected graph if no correlations meet threshold
            logger.warning("No edges meet correlation threshold, creating fully connected graph")
            edge_index = self._create_fully_connected_edges(symbol_to_idx)

        logger.info(f"Created graph with {edge_index.shape[1]} edges for {len(symbols)} nodes")
        return edge_index, symbol_to_idx

    def _calculate_symbol_correlations(self, symbols: List[str],
                                     raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate correlation matrix between symbols.

        Args:
            symbols: List of symbols
            raw_data: Raw data dictionary

        Returns:
            Correlation matrix as DataFrame
        """
        # Extract closing prices for each symbol
        price_data = {}

        for symbol in symbols:
            if symbol in raw_data:
                data = raw_data[symbol]
                if hasattr(data, 'close') and data.close is not None:
                    price_data[symbol] = data.close
                elif isinstance(data, dict) and 'close' in data:
                    price_data[symbol] = data['close']
                elif hasattr(data, 'data') and hasattr(data.data, 'close'):
                    price_data[symbol] = data.data.close

        if not price_data:
            logger.warning("No price data found for correlation calculation")
            # Return identity matrix
            return pd.DataFrame(np.eye(len(symbols)), index=symbols, columns=symbols)

        # Create DataFrame from price data
        price_df = pd.DataFrame(price_data)

        # Calculate returns (percentage change)
        returns_df = price_df.pct_change().dropna()

        # Calculate correlation matrix
        corr_matrix = returns_df.corr()

        # Fill NaN values with 0
        corr_matrix = corr_matrix.fillna(0)

        return corr_matrix

    def _create_edges_from_correlations(self, correlations: pd.DataFrame,
                                      symbol_to_idx: Dict[str, int],
                                      corr_threshold: float,
                                      top_k: int) -> List[List[int]]:
        """
        Create graph edges based on correlation matrix.

        Args:
            correlations: Correlation matrix
            symbol_to_idx: Symbol to index mapping
            corr_threshold: Minimum correlation for edge creation
            top_k: Maximum edges per node

        Returns:
            List of [source, target] edge pairs
        """
        edges = []

        for symbol1 in correlations.index:
            if symbol1 not in symbol_to_idx:
                continue

            # Get correlations for this symbol
            symbol_corrs = correlations.loc[symbol1].abs()  # Use absolute correlation

            # Filter by threshold and exclude self-correlation
            valid_corrs = symbol_corrs[
                (symbol_corrs >= corr_threshold) &
                (symbol_corrs.index != symbol1)
            ]

            # Sort by correlation strength and take top_k
            top_correlations = valid_corrs.nlargest(top_k)

            # Create edges
            for symbol2, corr_value in top_correlations.items():
                if symbol2 in symbol_to_idx:
                    edges.append([symbol_to_idx[symbol1], symbol_to_idx[symbol2]])

        return edges

    def _create_fully_connected_edges(self, symbol_to_idx: Dict[str, int]) -> torch.Tensor:
        """
        Create a fully connected graph when no correlations meet threshold.

        Args:
            symbol_to_idx: Symbol to index mapping

        Returns:
            Edge index tensor for fully connected graph
        """
        edges = []
        symbols = list(symbol_to_idx.keys())

        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i != j:  # No self-loops
                    edges.append([symbol_to_idx[symbol1], symbol_to_idx[symbol2]])

        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def get_sector_relationships(self, symbols: List[str]) -> Dict[str, List[str]]:
        """
        Get sector-based relationships between symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dictionary mapping symbols to related symbols in same sector
        """
        sector_groups = {}

        # Group symbols by sector
        for symbol in symbols:
            metadata = self.stock_metadata.get(symbol, {})
            sector = metadata.get('sector', 'Unknown')
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(symbol)

        # Create relationships within sectors
        relationships = {}
        for symbol in symbols:
            metadata = self.stock_metadata.get(symbol, {})
            sector = metadata.get('sector', 'Unknown')
            sector_symbols = sector_groups.get(sector, [])
            # Exclude self
            related_symbols = [s for s in sector_symbols if s != symbol]
            relationships[symbol] = related_symbols

        return relationships

    def enhance_graph_with_sector_info(self, edge_index: torch.Tensor,
                                     symbol_to_idx: Dict[str, int],
                                     symbols: List[str]) -> torch.Tensor:
        """
        Enhance the correlation-based graph with sector relationships.

        Args:
            edge_index: Current edge index
            symbol_to_idx: Symbol to index mapping
            symbols: List of symbols

        Returns:
            Enhanced edge index
        """
        sector_relationships = self.get_sector_relationships(symbols)

        # Convert existing edges to set for deduplication
        existing_edges = set()
        if edge_index.numel() > 0:
            for i in range(edge_index.shape[1]):
                edge = (edge_index[0, i].item(), edge_index[1, i].item())
                existing_edges.add(edge)

        # Add sector-based edges
        new_edges = []
        for symbol1, related_symbols in sector_relationships.items():
            if symbol1 not in symbol_to_idx:
                continue

            idx1 = symbol_to_idx[symbol1]
            for symbol2 in related_symbols:
                if symbol2 not in symbol_to_idx:
                    continue

                idx2 = symbol_to_idx[symbol2]
                edge = (idx1, idx2)

                # Only add if not already present
                if edge not in existing_edges:
                    new_edges.append([idx1, idx2])
                    existing_edges.add(edge)

        # Combine edges
        if new_edges:
            sector_edges = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
            combined_edges = torch.cat([edge_index, sector_edges], dim=1)
            logger.info(f"Enhanced graph with {len(new_edges)} sector-based edges")
            return combined_edges

        return edge_index