# Metrics Health Dashboard
_for Grafana / frontend_

## 1. Goal

The **Metrics Health** panel answers one simple question:

> "Can I trust today's evaluation metrics enough to make decisions?"

It does **not** show raw error charts – those live elsewhere.
Instead, it gives an at-a-glance traffic-light view of:

- Overall metrics health (green / yellow / red)
- Which metrics are reliable vs unreliable (MAE, RMSE, MAPE, MASE, DA, etc.)
- Which issues QualityAgent detected (e.g. identical metrics, high MAPE)
- What GuardrailAgent decided (e.g. "no auto model rotation allowed today")

This lets a human user (or LLM agent) quickly decide:
- "Are we good to act?" or
- "Do we need to rerun / fix something first?"

---

## 2. Data Inputs

The dashboard is driven by **two logical sources**:

1. **QualityAgent output** (metrics quality & issues)
2. **GuardrailAgent decision** (what's allowed / blocked)

### 2.1 QualityAgent output

We assume QualityAgent writes something like:

`data/metrics/quality_report_latest.json`:

```json
{
  "status": "failed",
  "severity": "high",
  "issues": [
    {"type": "mape_high", "count": 37},
    {"type": "identical_metrics_per_horizon", "horizon": 1, "metric": "mae"}
  ],
  "metrics_quality": {
    "mae": "ok",
    "rmse": "ok",
    "mape": "unreliable",
    "smape": "suspect",
    "mase": "ok",
    "directional_accuracy": "ok"
  }
}
```

### 2.2 GuardrailAgent decision

We assume GuardrailAgent writes:

`data/metrics/guardrail_decision_latest.json`:

```json
{
  "run_id": "2024-12-01T10:30:00Z",
  "can_rotate_models": false,
  "can_update_champions": false,
  "reason": "identical MAE across symbols on 1-day horizon",
  "warnings": [
    "MAPE unreliable for 37 symbols; excluded from score",
    "Horizon 1 metrics deemed invalid for this run"
  ]
}
```

---

## 3. Exposing this to Grafana

You have three easy options:

### Option A – Export as Prometheus metrics (recommended)

Create a tiny exporter (FastAPI / Flask / simple script) that:

1. Reads the JSON files.
2. Exposes Prometheus-style metrics on `/metrics`, e.g.:

```text
metrics_health_status{run_id="2024-12-01T10:30:00Z"} 0
# 0 = failed, 1 = passed

metrics_health_severity_level{level="low"} 0
metrics_health_severity_level{level="medium"} 0
metrics_health_severity_level{level="high"} 1

metric_quality_status{metric="mae",status="ok"} 1
metric_quality_status{metric="mae",status="suspect"} 0
metric_quality_status{metric="mae",status="unreliable"} 0

metric_quality_status{metric="mape",status="ok"} 0
metric_quality_status{metric="mape",status="suspect"} 0
metric_quality_status{metric="mape",status="unreliable"} 1

guardrail_can_rotate_models 0
guardrail_can_update_champions 0

metrics_issue_count{type="mape_high"} 37
metrics_issue_count{type="identical_metrics_per_horizon"} 1
```

Then Grafana uses **Prometheus as a data source** and can render rich panels.

### Option B – Use a JSON / HTTP API panel

Expose the JSON via a small HTTP endpoint and use:

* Grafana's JSON API plugin → to parse and visualize fields.

This is fine if you don't want Prometheus, but Prometheus makes aggregations & history easier.

### Option C – Log-based (Loki)

Push the JSON as logs to Loki; parse with labels.
Nice if you already use Loki, but more work than Option A for structured metrics.

**For a clean v1: Option A (Prometheus exporter) is the simplest mental model.**

---

## 4. Dashboard Layout

### 4.1 Top-Level Badge – "Metrics Health Today"

**Panel type**: Stat or Single Value
**Title**: `Metrics Health – Latest Run`

**Logic** (Prometheus):

* Query: `metrics_health_status` (0 or 1)

Map:

* 1 → Green ("OK – metrics trusted")
* 0 with severity = `medium` → Yellow ("Issues – check warnings")
* 0 with severity = `high` → Red ("Critical – do not trust metrics")

Display text example:

* Green: `Healthy ✅`
* Yellow: `Degraded ⚠️`
* Red: `Failed ❌`

Subtitle:

* Show timestamp from `run_id` or a label:

  * "Last Evaluation: 2024-12-01 10:30 UTC"

---

### 4.2 Guardrail Decision Panel

**Panel type**: Stat or Table

**Title**: `Guardrails – Actions Allowed`

Fields:

* `can_rotate_models` (0/1)
* `can_update_champions` (0/1)

Example mapping:

* If both 1:

  * "Auto rotations allowed ✅"
* If `can_rotate_models = 0` but metrics OK:

  * "Rotations blocked (policy) ⚠️"
* If metrics health failed (severity high):

  * "Rotations blocked (metrics failure) ❌"

You can add a **Text panel** directly underneath that shows the `reason` string from the JSON, for example:

> **Guardrail reason:**
> `identical MAE across symbols on 1-day horizon`

---

### 4.3 Metric Quality Matrix

**Panel type**: Table or Heatmap-like table

**Title**: `Metric Reliability by Type`

Goal: One row per metric, one column per quality status.

Example table:

| Metric               | Status       | Comment                |
| -------------------- | ------------ | ---------------------- |
| MAE                  | ✅ ok         | Used in score          |
| RMSE                 | ✅ ok         | Used in score          |
| MAPE                 | ❌ unreliable | Ignored this run       |
| SMAPE                | ⚠️ suspect   | Interpret with caution |
| MASE                 | ✅ ok         | Used in guardrails     |
| Directional Accuracy | ✅ ok         | Used for decisions     |

How to drive this from Prometheus:

* Query: `metric_quality_status{metric=~".+"}`
  (You'll have one row per (metric, status) with value 0/1)
* Use value==1 to select the **active status** per metric.
* You can use Grafana's value mappings to show:

  * ✅ ok
  * ⚠️ suspect
  * ❌ unreliable

---

### 4.4 Issues List Panel

**Panel type**: Table or Logs-style

**Title**: `Detected Issues (QualityAgent)`

From Prometheus metrics like:

* `metrics_issue_count{type="mape_high"} = 37`
* `metrics_issue_count{type="identical_metrics_per_horizon"} = 1`

Render as a table:

| Issue Type                    | Count | Severity |
| ----------------------------- | ----- | -------- |
| mape_high                     | 37    | medium   |
| identical_metrics_per_horizon | 1     | high     |

Optionally:

* Add a Text panel that explains typical meaning per issue type (static help text).

---

### 4.5 Trend Panel – "Health over Time"

**Panel type**: Time series or Bar chart

**Title**: `Metrics Health – History`

Use metrics such as:

* `metrics_health_status` over time
* `guardrail_can_rotate_models` over time

This gives you a sense of:

* How often metrics are failing,
* Periods where guardrails were blocking automatic changes,
* Trends in quality (are things improving as you fix pipelines?).

---

## 5. Interaction with Agents

### 5.1 How agents should see this

* **AnalyticsAgent / PerformanceReportingAgent**
  Doesn't need Grafana – it reads the same underlying JSON/metrics and can:

  * annotate reports,
  * decide which metrics to highlight.

* **LLM Reporting Agent (ReportingLLM)**
  Can consume:

  * the QualityAgent JSON,
  * Guardrail decisions,
  * and optionally a summarised "metrics health" object (e.g., from the exporter).

LLM-friendly object to feed into prompts:

```json
{
  "metrics_health": {
    "status": "failed",
    "severity": "high",
    "metrics_quality": {
      "mae": "ok",
      "rmse": "ok",
      "mape": "unreliable",
      "mase": "ok",
      "directional_accuracy": "ok"
    },
    "guardrails": {
      "can_rotate_models": false,
      "can_update_champions": false,
      "reason": "identical MAE across symbols on 1-day horizon"
    },
    "top_issues": [
      "37 symbols with unstable MAPE",
      "1-day horizon MAE identical across symbols"
    ]
  }
}
```

The LLM can then say:

> "Metrics health is **red** for this run: MAPE is unreliable and 1-day MAE is structurally broken (identical across symbols). Guardrails have blocked automatic model rotations and champion updates. The analysis in this report focuses on MAE, RMSE, MASE and Directional Accuracy on multi-day horizons."

---

## 6. Minimal Implementation Checklist

To actually ship this:

1. **QualityAgent**

   * Already produces metrics quality and issues → write to JSON.

2. **GuardrailAgent**

   * Already produces decisions → write to JSON.

3. **Exporter (tiny service)**

   * Reads both JSONs periodically (or on request).
   * Renders all metrics text in Prometheus format.

4. **Grafana**

   * Add Prometheus data source.
   * Create dashboard `Metrics Health`.
   * Panels:

     * Stat: overall health
     * Stat: guardrail action allowed
     * Table: metric reliability matrix
     * Table: issues list
     * Time series: health/guardrail over time

5. **Docs**

   * Link `METRICS_POLICY.md` + `METRICS_QUALITY_PLAYBOOK.md` from the dashboard as "Help" / "Info" links.

---

## 7. Future Enhancements

Later, you can extend the "Metrics Health" dashboard to also show:

* **Per-bucket health** (e.g. `ai_basket` vs `energy` vs `crypto`).
* **Per-horizon health** (1d vs 5d vs 20d).
* **Regime-aware health** (values during peer_shock_flag=1 vs 0).

But a minimal v1 with:

* global health badge,
* guardrail status,
* metric reliability matrix,
* issues list,

will already give you a very clean, user-friendly entry point into "can I trust today's numbers?".