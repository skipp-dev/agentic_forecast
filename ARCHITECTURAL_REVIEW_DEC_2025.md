# Architectural Review - December 2025

## State of the System
**Summary:** "Solid Components, Fragile Glue."

The system has matured significantly with the introduction of **SQLAlchemy** (Persistence), **Pandera** (Validation), and **MLflow** (Registry). The individual components are now closer to production-grade. However, the **integration layer** remains brittle. The "glue" code—specifically the `OrchestratorAgent` and the dependency management between services—shows signs of rapid iteration without architectural enforcement. The recent `DataSpec` incident (where a core data class was missing and had to be monkey-patched) is a critical indicator of structural instability.

*   **Solid:** Database abstraction (SQLAlchemy), Risk Controls (Circuit Breakers), Docker setup (CUDA optimized).
*   **Fragile:** Dependency management (imports are messy), Orchestration logic (monolithic `if/else` blocks), Error recovery (synchronous retries).
*   **Missing:** Asynchronous execution model, comprehensive integration tests (too much mocking), unified configuration management.

## Top 10 Concerns & Risks

1.  **Synchronous Blocking in Execution Path (Severity: Critical)**
    *   **Symptom:** The `ExecutionGateway` uses `time.sleep(self.retry_delay)` inside the retry loop.
    *   **Risk:** In a trading system, blocking the main thread for 1 second is unacceptable. It halts market data ingestion, heartbeat signals, and other agent activities. This must be asynchronous (`await asyncio.sleep`).

2.  **"Ghost" Dependencies & Monkey-Patching (Severity: High)**
    *   **Symptom:** We had to inject a dummy `DataSpec` class into three different files to get tests to pass.
    *   **Risk:** The system is operating on "fake" data contracts. If `DataSpec` was intended to define tensor shapes or feature columns, the current code is flying blind. This technical debt needs immediate repayment.

3.  **Dual-Write Inconsistency (Severity: High)**
    *   **Symptom:** `PaperBroker` writes state to both SQLite (`db_service`) and a JSON file (legacy).
    *   **Risk:** Dual writes eventually diverge. If the process crashes between the DB commit and the JSON write, the system restarts in an ambiguous state. The JSON fallback should be killed immediately.

4.  **Monolithic Orchestrator Logic (Severity: Medium)**
    *   **Symptom:** `OrchestratorAgent.coordinate_workflow` is a growing procedural block of `if/elif` statements handling GPU checks, drift, and regime changes.
    *   **Risk:** This is a "God Object" anti-pattern. It makes testing specific workflow paths (like "Regime Change -> HPO") difficult without mocking the entire world. It should be a proper State Machine (e.g., using LangGraph nodes).

5.  **MLflow Integration Depth (Severity: Medium)**
    *   **Symptom:** We replaced the registry backend, but `TrainingService` and `InferenceService` likely still expect the old return types or file paths.
    *   **Risk:** Saving a model might work, but *loading* it for inference will fail if the consumer expects a local path and gets an MLflow Run ID. The integration is likely superficial.

6.  **Mock-Heavy Unit Tests (Severity: Medium)**
    *   **Symptom:** `tests/test_metrics.py` mocks almost every internal method of the agents.
    *   **Risk:** We are testing the mocks, not the code. We have zero assurance that the `Orchestrator` actually talks to the `Broker` correctly in a real run.

7.  **Hardcoded Configuration (Severity: Low)**
    *   **Symptom:** `ExecutionGateway` defaults to `max_retries=3` in code; `metrics.py` defines buckets in code.
    *   **Risk:** Tuning system behavior requires code changes and redeployment. These should be in `config.yaml` or environment variables.

8.  **Lack of Structured Logging (Severity: Low)**
    *   **Symptom:** Logs are standard Python `logging.info`.
    *   **Risk:** In a distributed/containerized setup, we need structured logs (JSON) with `trace_id` to correlate an Orchestrator decision with a Broker action and a Database write.

9.  **GPU Resource Contention (Severity: Low)**
    *   **Symptom:** `_check_gpu_status` exists, but no reservation system.
    *   **Risk:** If `TrainingService` and `InferenceService` run concurrently (e.g., retraining while trading), they will OOM (Out of Memory) the GPU.

10. **Secret Management (Severity: Low)**
    *   **Symptom:** API keys are likely loaded from `.env`, but are they passed securely to the Docker container?
    *   **Risk:** Leaking keys in logs or Docker history.

## Prioritized ToDo List

| Priority | Task | Why it Matters | Expected Impact |
| :--- | :--- | :--- | :--- |
| **1** | **Restore `DataSpec` Source of Truth** | The system is currently patched with dummy classes. | Prevents runtime `AttributeError` when code tries to access data properties. |
| **2** | **Asyncio Migration for Gateway** | Blocking calls (`time.sleep`) freeze the entire agent loop. | **Non-blocking execution**; system remains responsive to market data during retries. |
| **3** | **Deprecate JSON State in Broker** | Dual writes cause state drift. | **Data Integrity**; single source of truth (Postgres/SQLite) for portfolio state. |
| **4** | **Deepen MLflow Integration** | Consumers need to know how to load models from MLflow URIs. | **Working Inference**; ensures models trained can actually be served. |
| **5** | **Refactor Orchestrator to State Graph** | The `if/else` logic is unmaintainable. | **Testability**; isolate logic for "Drift" vs "Trading" into separate, testable nodes. |
| **6** | **Implement Structured Logging** | Text logs are hard to query. | **Observability**; ability to trace a specific Order ID across all services. |
| **7** | **Create Integration Test Suite** | Mocks are hiding bugs. | **Confidence**; a real test running Orchestrator -> Broker -> DB -> Registry. |

## Architectural Review Q&A

### 1. “Where is the Event Loop?”

> *Is this system running inside a `while True` loop in `main.py`? A Cron job? A FastAPI background task? The execution context dictates how we handle concurrency.*

**Answer:**
`agentic_forecast` is a **batch system**. There is no infinite `while True` loop; each run is a one-shot LangGraph execution triggered by an external scheduler (cron / orchestration tool). That keeps concurrency and failure handling at the workflow/scheduler level, not buried in `main.py`.

*   **Today:** Single-run entrypoint triggered by CLI/Docker.
*   **Target:** Orchestrated by Airflow/Prefect/Cron for daily runs and weekend HPO.

### 2. “How do we handle Market Data gaps?”

> *If AlphaVantage returns NaN or times out, does the DataPipeline crash, or do we have a forward-fill/fallback strategy defined in Pandera?*

**Answer:**
Right now we don’t have full Data Contracts, so market data gaps can propagate too far. The plan is to validate all incoming data with Pandera, define explicit gap-handling rules (short gaps OK, unexpected gaps fail-fast), and treat timeouts as ‘no trade’ rather than limping along with stale or NaN-filled data.

*   **Today:** "Accepted on faith". Gaps can crash the pipeline or yield garbage.
*   **Target:** Strict Pandera schemas per layer. Explicit gap policies (drop symbol vs forward-fill).

### 3. “What is the deployment topology?”

> *Is the Database in the same container? (Docker Compose suggests yes). For production, is there a plan to move Postgres to a managed service (RDS/Cloud SQL)?*

**Answer:**
Today, DB and app sit together in Docker Compose for development. For production we plan to migrate to Postgres/TimescaleDB as a managed service (RDS/Cloud SQL style), with the app running in its own containers pointed at that shared DB. That’s necessary for concurrency, durability, and operational sanity.

*   **Today:** Single-stack Docker Compose (App + DB).
*   **Target:** App in K8s/Container, DB in Managed Service (RDS/Timescale Cloud).

### 4. “Who cleans up the MLflow artifacts?”

> *We are saving every model. Do we have a retention policy? The disk will fill up quickly with high-frequency retraining.*

**Answer:**
Right now we’re in ‘save everything’ mode with no MLflow retention policy, which is not sustainable. The plan is to enforce a retention policy (e.g. last N champions and last 90 days of experiments) and run a scheduled GC job to delete old artifacts from the store while keeping enough history for audit and comparability.

*   **Today:** Infinite retention.
*   **Target:** Retention policy (N champions, X days history) + GC job.
