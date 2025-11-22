# Agentic AI Stock Forecasting System - User Guide

## 1. System Overview
This is an autonomous, graph-based AI system designed to manage and forecast a portfolio of stocks (defined in `watchlist_ibkr.csv`, typically ~500+ symbols like the S&P 500).

Unlike simple forecasting scripts, this system acts as an **Autonomous Hedge Fund Analyst Team**. It doesn't just predict numbers; it monitors data quality, detects when the market "drifts" (changes behavior), reads the news, and decides *when* to retrain its own models without human intervention.

It is optimized to run on a single consumer GPU (e.g., RTX 5070 12GB) by intelligently swapping specialized AI models in and out of memory.

## 2. How to Operate

### **Starting the System**
1.  Open your terminal (PowerShell).
2.  Navigate to the project folder.
3.  Run the main command:
    ```powershell
    python main.py
    ```

### **Interactive Mode (Chat & Simulate)**
To ask questions about your portfolio or simulate market events:
```powershell
python interactive.py
```
*   **Ask:** "How is AAPL performing?"
*   **Simulate:** "What happens if the Fed raises rates by 0.5%?"

### **Stopping the System**
*   Press `Ctrl + C` in the terminal.
*   **Note:** The system includes a safety mechanism to automatically unload AI models and free up your GPU memory (VRAM) when it shuts down.

## 3. The AI Team (LLM Mapping)

The system uses different "brains" (LLMs) for different tasks to maximize efficiency and intelligence:

| Role | Agent Name | Model Used | Why? |
| :--- | :--- | :--- | :--- |
| **The Boss** | **Orchestrator** | **Gemma 3 12B** | High reasoning capability. Decides *what* to do (Retrain? Optimize? Sleep?) based on complex metrics. |
| **The Analyst** | **Writer** | **Llama 3.1 8B** | Excellent context window and writing skills. Synthesizes numbers, news, and drift data into human-readable reports. |
| **The Assistant** | **Tool User** | **Phi-3 Mini** | Fast and lightweight. Used for simple tasks like checking severity levels or formatting news to save time/VRAM. |

| **The Researcher**| **Research Agent**| **Llama 3.1 8B** | (Same as Writer) Performs deep dives into market trends if specific anomalies are detected. |

## 4. Workflow & Triggers

The system follows a logical "Graph" workflow. Here is the sequence of events:

1.  **Start:** You run `python main.py`.
2.  **Data Ingestion:**
    *   *Action:* Fetches latest data for ALL symbols in `watchlist_ibkr.csv` (e.g., 500+ symbols).
3.  **Drift Check:**
    *   *Trigger:* Completion of data fetching.
    *   *Action:* Statistically analyzes incoming data. Has the market volatility changed? Are correlations broken?
4.  **Orchestrator Review (The Brain):**
    *   *Trigger:* Drift check complete.
    *   *Action:* **Gemma 3** looks at the drift flag and current error rates (MAPE).
    *   *Decision:*
        *   **"Market Changed!"** -> Triggers **Retraining** of LSTM models.
        *   **"Models are bad!"** -> Triggers **HPO (Hyperparameter Optimization)**.
        *   **"All good."** -> Proceeds to forecasting.
5.  **News Enrichment:**
    *   *Trigger:* Before final reporting.
    *   *Action:* **Phi-3** fetches and summarizes relevant news for the top movers.
6.  **Analytics & Reporting (The Output):**

    *   **Location:** `docs/forecasts/YYYY-MM-DD_forecast_report.md`

    *   **Content:** Real-time status of which agent is working (`--- Node: Orchestrator Agent ---`).
| :--- | :--- | :--- |
| **Portfolio Management** | ✅ **Automated** | Handles 500+ symbols defined in your CSV. |
| **Model Maintenance** | ✅ **Automated** | Self-healing. Retrains models automatically when accuracy drops. |
| **Resource Management** | ✅ **Automated** | Swaps 3 different LLMs in/out of 12GB VRAM automatically. |
| **Reporting** | ✅ **Automated** | Writes Markdown reports to disk. |
| **Configuration** | ✋ **Manual** | You set API keys and the symbol list (`watchlist_ibkr.csv`) once. |

## 7. Troubleshooting
*   **"Orchestrator failed after max retries":** The local model might be struggling with a complex prompt. The system will retry 3 times automatically.
*   **"CUDA Out of Memory":** Should not happen. If it does, ensure no other heavy GPU apps are running. The system is tuned for 12GB VRAM.
