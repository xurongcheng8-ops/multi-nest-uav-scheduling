from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .data_generator import normalize_task_dataframe


ROUTE_COLORS = [
    "#32d7ff",
    "#23d18b",
    "#ffb44c",
    "#ff5b73",
    "#b58cff",
    "#66a3ff",
    "#ffd166",
    "#06d6a0",
    "#ef476f",
    "#118ab2",
]


def plot_schedule_map(
    nests: pd.DataFrame,
    tasks: pd.DataFrame,
    result: dict | None = None,
    zones: list[dict] | None = None,
    title: str = "UAV Scheduling Map",
) -> go.Figure:
    result = result or {}
    zones = zones or []
    served_ids = set(result.get("served_tasks", {}).keys())
    unserved_ids = set(result.get("unserved_tasks", []))
    tasks_df = normalize_task_dataframe(tasks)
    tasks_df["task_id"] = tasks_df["task_id"].astype(str)
    tasks_df["_status"] = tasks_df["task_id"].apply(
        lambda task_id: "unserved" if task_id in unserved_ids else ("served" if task_id in served_ids else "pending")
    )

    fig = go.Figure()
    for zone in zones:
        _add_zone_shape(fig, zone)

    fig.add_trace(
        go.Scatter(
            x=nests["x"],
            y=nests["y"],
            mode="markers+text",
            text=nests["nest_id"],
            textposition="top center",
            marker=dict(symbol="square", size=15, color="#f7fbff", line=dict(color="#1c7ed6", width=2)),
            name="机巢",
            hovertemplate="Nest %{text}<br>(%{x:.1f}, %{y:.1f})<extra></extra>",
        )
    )

    _add_task_trace(fig, tasks_df[(tasks_df["_status"] == "served") & (~tasks_df["is_emergency"])], "已服务任务", "#23d18b", "circle")
    _add_task_trace(fig, tasks_df[(tasks_df["_status"] == "pending") & (~tasks_df["is_emergency"])], "待调度任务", "#9fb9d2", "circle-open")
    _add_task_trace(fig, tasks_df[tasks_df["_status"] == "unserved"], "未服务任务", "#ff5b73", "x")
    _add_task_trace(fig, tasks_df[(tasks_df["priority"] >= 4) & (~tasks_df["is_emergency"])], "高优先级任务", "#ffb44c", "diamond")
    _add_task_trace(fig, tasks_df[tasks_df["is_emergency"]], "突发任务", "#ff2f5f", "star")

    for index, route in enumerate(result.get("routes", {}).values()):
        path = route.get("path", [])
        if len(path) < 2:
            continue
        xs = [point[0] for point in path]
        ys = [point[1] for point in path]
        color = ROUTE_COLORS[index % len(ROUTE_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=5, color=color),
                name=f"{route['uav_id']} 路径",
                hovertemplate=f"{route['uav_id']}<br>(%{{x:.1f}}, %{{y:.1f}})<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=560,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        xaxis=dict(range=[0, 100], title="x", gridcolor="rgba(255,255,255,0.09)"),
        yaxis=dict(range=[0, 100], title="y", gridcolor="rgba(255,255,255,0.09)", scaleanchor="x", scaleratio=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,17,31,0.55)",
    )
    return fig


def _add_zone_shape(fig: go.Figure, zone: dict) -> None:
    if zone.get("shape") == "rect":
        fig.add_shape(
            type="rect",
            x0=zone["x0"],
            y0=zone["y0"],
            x1=zone["x1"],
            y1=zone["y1"],
            fillcolor=zone.get("color", "rgba(255, 91, 115, 0.14)"),
            line=dict(color=zone.get("line", "#ff5b73"), width=2),
            layer="below",
        )
        fig.add_annotation(
            x=(float(zone["x0"]) + float(zone["x1"])) / 2,
            y=float(zone["y1"]) + 2,
            text=zone.get("name", zone.get("zone_id", "zone")),
            showarrow=False,
            font=dict(color=zone.get("line", "#ff5b73"), size=11),
        )
    elif zone.get("shape") == "circle":
        cx = float(zone["cx"])
        cy = float(zone["cy"])
        radius = float(zone["r"])
        fig.add_shape(
            type="circle",
            x0=cx - radius,
            y0=cy - radius,
            x1=cx + radius,
            y1=cy + radius,
            fillcolor=zone.get("color", "rgba(255, 180, 76, 0.14)"),
            line=dict(color=zone.get("line", "#ffb44c"), width=2),
            layer="below",
        )
        fig.add_annotation(
            x=cx,
            y=cy + radius + 2,
            text=zone.get("name", zone.get("zone_id", "zone")),
            showarrow=False,
            font=dict(color=zone.get("line", "#ffb44c"), size=11),
        )


def plot_metric_bars(metrics_df: pd.DataFrame, metric_columns: list[str]) -> go.Figure:
    fig = go.Figure()
    for column in metric_columns:
        if column not in metrics_df.columns:
            continue
        fig.add_trace(
            go.Bar(
                x=metrics_df["algorithm"],
                y=metrics_df[column],
                name=column,
                text=[_format_value(value) for value in metrics_df[column]],
                textposition="auto",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        height=420,
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,17,31,0.55)",
    )
    return fig


def plot_uav_task_counts(result: dict) -> go.Figure:
    rows = [
        {"uav_id": route["uav_id"], "task_count": len(route.get("task_sequence", []))}
        for route in result.get("routes", {}).values()
    ]
    df = pd.DataFrame(rows)
    fig = go.Figure()
    if not df.empty:
        fig.add_bar(x=df["uav_id"], y=df["task_count"], marker_color="#32d7ff", text=df["task_count"], textposition="auto")
    fig.update_layout(
        title="每架无人机服务任务数量",
        template="plotly_dark",
        height=320,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,17,31,0.55)",
    )
    return fig


def plot_dynamic_metric_compare(before: dict, after: dict) -> go.Figure:
    keys = ["task_completion_rate", "average_response_time", "total_flight_distance", "uav_utilization"]
    fig = go.Figure()
    fig.add_bar(x=keys, y=[before.get(key, 0) for key in keys], name="插入前", marker_color="#32d7ff")
    fig.add_bar(x=keys, y=[after.get(key, 0) for key in keys], name="插入后", marker_color="#ffb44c")
    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        height=360,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,17,31,0.55)",
    )
    return fig


def _add_task_trace(fig: go.Figure, df: pd.DataFrame, name: str, color: str, symbol: str) -> None:
    if df.empty:
        return
    hover = [
        f"{row.task_id}<br>type: {row.task_type}<br>priority: {row.priority}<br>window: {row.earliest_start:.1f}-{row.latest_finish:.1f}"
        for row in df.itertuples()
    ]
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=dict(size=11, color=color, symbol=symbol, line=dict(color="#ffffff", width=0.7)),
            name=name,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        )
    )


def _format_value(value) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)
