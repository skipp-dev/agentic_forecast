# Phase 4: Execution System Implementation Plan

## Objective
Implement the Execution System to take generated orders from the Portfolio Construction phase and "execute" them. This phase will focus on a **Paper Trading** simulation first, with hooks for future live execution.

## Components

### 1. Execution Node (`src/nodes/trade_execution_nodes.py`)
*   **Responsibility**: Receive `orders` from the state.
*   **Action**: Pass orders to the `ExecutionAgent`.
*   **Output**: Update state with `execution_results` (fills, rejections).
*   **Note**: The existing `src/nodes/execution_nodes.py` is used for data/forecasting. We will create a NEW file `src/nodes/trade_execution_nodes.py` to avoid conflicts.

### 2. Execution Agent (`src/agents/execution_agent.py`)
*   **Responsibility**: Manage the interaction with the broker interface.
*   **Logic**:
    *   Receive list of target orders.
    *   Perform pre-trade risk checks (e.g., "Are we buying too much of one asset?").
    *   Route orders to the configured broker.
    *   Record transactions.

### 3. Broker Interface (`src/interfaces/broker_interface.py`)
*   **Abstract Base Class**: Defines methods like `get_positions()`, `get_cash()`, `place_order()`.

### 4. Paper Broker (`src/brokers/paper_broker.py`)
*   **Implementation**: In-memory or file-based tracking of cash and positions.
*   **Features**:
    *   Simulate fills (assume market price).
    *   Track commission (optional).
    *   Persist state to `data/paper_portfolio.json`.

## Workflow
1.  **Portfolio Node** outputs `orders` (e.g., `[{'symbol': 'AAPL', 'action': 'BUY', 'quantity': 10}]`).
2.  **Execution Node** calls `ExecutionAgent.execute_orders(orders)`.
3.  **Execution Agent** checks risks.
4.  **Execution Agent** calls `PaperBroker.place_order()`.
5.  **Paper Broker** updates local state and returns `Fill` object.
6.  **Execution Node** logs results and updates pipeline state.

## Constraints
*   **Safety**: No live trading by default. "Paper" mode is the default.
*   **Policy**: Adhere to `LLM_POLICY.md` (no direct unauthorized trading).

## Next Steps
1.  Refactor/Clean `src/nodes/execution_nodes.py`.
2.  Create `src/interfaces/broker_interface.py`.
3.  Create `src/brokers/paper_broker.py`.
4.  Create `src/agents/execution_agent.py`.
5.  Update `main.py` to include the Execution Node.
