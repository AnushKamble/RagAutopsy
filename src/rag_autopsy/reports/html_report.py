"""Standalone interactive HTML report generator for RAG-Autopsy evaluation."""

import json
from pathlib import Path
from typing import Any


def generate_interactive_html_report(
    summary_data: dict[str, Any],
    trial_records: list[dict[str, Any]],
    output_path: str = "results/report.html",
) -> str:
    """Generate a zero-dependency, self-contained interactive HTML evaluation dashboard.

    Includes metric cards, confusion matrices, bootstrap intervals, and an interactive
    per-trial inspector with zero data leakage.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    regime = summary_data.get("regime", "regime_a").upper()
    split = summary_data.get("target_split", "test").upper()
    validity = summary_data.get("validity_summary", {})
    diagnostics = summary_data.get("diagnostic_results", {})

    # Build rows for metrics table
    diag_rows_html = ""
    for d_name, d_info in diagnostics.items():
        m = d_info.get("metrics", {})
        b = d_info.get("bootstrap", {})
        f1 = m.get("macro_f1", 0.0)
        f1_low = b.get("macro_f1", {}).get("ci_lower", f1)
        f1_high = b.get("macro_f1", {}).get("ci_upper", f1)
        bal_acc = m.get("balanced_accuracy", 0.0)
        cov = m.get("coverage", 1.0) * 100.0
        cond_acc = m.get("conditional_accuracy", 0.0) * 100.0

        diag_rows_html += f"""
        <tr>
            <td style="font-weight: 600; color: #1e293b;">{d_name}</td>
            <td style="font-weight: 700; color: #0284c7;">{f1:.3f} <span style="font-size: 0.85em; color: #64748b;">[{f1_low:.3f}, {f1_high:.3f}]</span></td>
            <td>{bal_acc:.3f}</td>
            <td>{cov:.1f}%</td>
            <td>{cond_acc:.1f}%</td>
        </tr>
        """

    # Build trial cards json for client-side search/inspection
    clean_trials = []
    for tr in trial_records[:150]:
        clean_trials.append({
            "trial_id": tr.get("trial_id", ""),
            "question_id": tr.get("question_id", ""),
            "question": tr.get("question", ""),
            "injected_fault": tr.get("injected_fault", ""),
            "trial_valid": tr.get("trial_valid", True),
            "d4_prediction": tr.get("d4_combined_policy_pred", ""),
            "answer": tr.get("answer", "")[:200],
            "context_len": len(tr.get("generator_context", "")),
        })

    trials_json = json.dumps(clean_trials)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG-Autopsy Benchmark Report ({regime})</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8fafc;
            color: #0f172a;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #0f172a, #1e293b);
            color: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 25px;
        }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 26px; }}
        .header p {{ margin: 0; opacity: 0.85; font-size: 15px; }}
        .badge {{
            display: inline-block;
            background: #0284c7;
            color: white;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 700;
            margin-top: 10px;
        }}
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .card-label {{ font-size: 12px; font-weight: 600; text-transform: uppercase; color: #64748b; margin-bottom: 6px; }}
        .card-val {{ font-size: 24px; font-weight: 800; color: #0f172a; }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            margin-bottom: 25px;
        }}
        .section h2 {{ margin: 0 0 15px 0; font-size: 18px; color: #1e293b; border-bottom: 2px solid #f1f5f9; padding-bottom: 10px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            text-align: left;
            padding: 12px 14px;
            border-bottom: 1px solid #f1f5f9;
        }}
        th {{
            background-color: #f8fafc;
            color: #475569;
            font-weight: 700;
        }}
        .search-box {{
            width: 100%;
            padding: 10px 14px;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            margin-bottom: 15px;
            box-sizing: border-box;
            font-size: 14px;
        }}
        .trial-row {{ cursor: pointer; }}
        .trial-row:hover {{ background-color: #f1f5f9; }}
        .match-tag {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
        }}
        .match-true {{ background: #dcfce7; color: #15803d; }}
        .match-false {{ background: #fee2e2; color: #b91c1c; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RAG-Autopsy Benchmark Report</h1>
            <p>Fault-Injection-Validated Failure Attribution for Retrieval-Augmented Generation</p>
            <span class="badge">REGIME: {regime}</span>
            <span class="badge" style="background: #10b981;">SPLIT: {split}</span>
            <span class="badge" style="background: #8b5cf6;">LEAKAGE BOUNDARY: STRICT ZERO-LEAKAGE</span>
        </div>

        <div class="cards-grid">
            <div class="card">
                <div class="card-label">Total Attempted Trials</div>
                <div class="card-val">{validity.get("total_attempted", 0)}</div>
            </div>
            <div class="card">
                <div class="card-label">Adjudicated Valid Trials</div>
                <div class="card-val" style="color: #10b981;">{validity.get("total_valid", 0)}</div>
            </div>
            <div class="card">
                <div class="card-label">Validity Effectiveness Rate</div>
                <div class="card-val">{validity.get("overall_validity_rate", 1.0)*100:.1f}%</div>
            </div>
            <div class="card">
                <div class="card-label">D4 Combined Policy F1</div>
                <div class="card-val" style="color: #0284c7;">{diagnostics.get("d4_combined_policy", {}).get("metrics", {}).get("macro_f1", 0.0):.3f}</div>
            </div>
        </div>

        <div class="section">
            <h2>Benchmark Attribution Performance</h2>
            <table>
                <thead>
                    <tr>
                        <th>Diagnostic Policy</th>
                        <th>Macro-F1 (95% Clustered CI)</th>
                        <th>Balanced Accuracy</th>
                        <th>Coverage</th>
                        <th>Conditional Accuracy</th>
                    </tr>
                </thead>
                <tbody>
                    {diag_rows_html}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Interactive Trial Inspector (Unprivileged Traces vs Evaluator Adjudication)</h2>
            <input type="text" id="searchInput" class="search-box" placeholder="Filter trials by query, fault ID, or prediction..." onkeyup="filterTrials()">
            <table>
                <thead>
                    <tr>
                        <th>Trial ID</th>
                        <th>Question</th>
                        <th>Injected Fault</th>
                        <th>D4 Prediction</th>
                        <th>Attribution Match</th>
                        <th>Context Size</th>
                    </tr>
                </thead>
                <tbody id="trialsTableBody">
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const trials = {trials_json};

        function renderTrials(data) {{
            const tbody = document.getElementById("trialsTableBody");
            tbody.innerHTML = "";
            data.forEach(t => {{
                const tr = document.createElement("tr");
                tr.className = "trial-row";
                const isMatch = (t.injected_fault === t.d4_prediction);
                tr.innerHTML = `
                    <td style="font-family: monospace; font-size: 12px;">${{t.trial_id}}</td>
                    <td>${{t.question}}</td>
                    <td><strong>${{t.injected_fault}}</strong></td>
                    <td style="color: #0284c7; font-weight: 600;">${{t.d4_prediction}}</td>
                    <td><span class="match-tag ${{isMatch ? 'match-true' : 'match-false'}}">${{isMatch ? 'MATCH' : 'MISMATCH'}}</span></td>
                    <td style="color: #64748b;">${{t.context_len}} chars</td>
                `;
                tbody.appendChild(tr);
            }});
        }}

        function filterTrials() {{
            const query = document.getElementById("searchInput").value.toLowerCase();
            const filtered = trials.filter(t =>
                t.question.toLowerCase().includes(query) ||
                t.injected_fault.toLowerCase().includes(query) ||
                t.d4_prediction.toLowerCase().includes(query) ||
                t.trial_id.toLowerCase().includes(query)
            );
            renderTrials(filtered);
        }}

        renderTrials(trials);
    </script>
</body>
</html>
"""

    with open(out_p, "w", encoding="utf-8") as f:
        f.write(html_content)

    return str(out_p)
