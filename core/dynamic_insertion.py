from __future__ import annotations

from copy import deepcopy

import pandas as pd

from .algorithms import evaluate_fixed_route_sequence
from .data_generator import normalize_nest_dataframe, normalize_task_dataframe, normalize_uav_dataframe
from .metrics import compare_plan_sequences


def insert_emergency_task(
    base_result: dict,
    tasks: pd.DataFrame,
    nests: pd.DataFrame,
    uavs: pd.DataFrame,
    emergency_task: pd.Series | dict,
    lambda_delay: float = 0.5,
    lambda_disruption: float = 3.0,
    lambda_priority: float = 2.0,
) -> dict:
    tasks_df = normalize_task_dataframe(tasks)
    emergency = _series_to_dict(emergency_task)
    combined_tasks = pd.concat([tasks_df, pd.DataFrame([emergency])], ignore_index=True)
    combined_tasks = normalize_task_dataframe(combined_tasks)
    nests_df = normalize_nest_dataframe(nests)
    uavs_df = normalize_uav_dataframe(uavs)

    tasks_by_id = {str(row["task_id"]): _series_to_dict(row) for _, row in combined_tasks.iterrows()}
    nests_by_id = {str(row["nest_id"]): _series_to_dict(row) for _, row in nests_df.iterrows()}
    uavs_by_id = {str(row["uav_id"]): _series_to_dict(row) for _, row in uavs_df.iterrows()}

    best = None
    emergency_id = str(emergency["task_id"])

    for uav_id, route in base_result.get("routes", {}).items():
        uav = uavs_by_id.get(str(uav_id))
        if uav is None:
            continue
        nest = nests_by_id.get(str(uav["nest_id"]))
        if nest is None:
            continue
        original_sequence = list(route.get("task_sequence", []))
        for position in range(len(original_sequence) + 1):
            new_sequence = original_sequence[:position] + [emergency_id] + original_sequence[position:]
            evaluation = evaluate_fixed_route_sequence(uav, nest, tasks_by_id, new_sequence)
            if not evaluation.get("feasible"):
                continue

            new_route = evaluation["route"]
            extra_distance = float(new_route["total_distance"]) - float(route.get("total_distance", 0.0))
            additional_delay = _additional_delay(route, new_route)
            disrupted_tasks = max(0, len(original_sequence) - position)
            cost = (
                extra_distance
                + lambda_delay * additional_delay
                + lambda_disruption * disrupted_tasks
                - lambda_priority * float(emergency.get("priority", 5))
            )
            candidate = {
                "uav_id": str(uav_id),
                "insert_position": position + 1,
                "new_route": new_route,
                "served_updates": evaluation["served_updates"],
                "extra_distance": extra_distance,
                "additional_delay": additional_delay,
                "affected_tasks": disrupted_tasks,
                "cost": cost,
            }
            if best is None or candidate["cost"] < best["cost"]:
                best = candidate

    if best is None:
        return {
            "feasible": False,
            "message": "当前计划下无法可行插入，建议触发局部重优化。",
            "emergency_task": emergency,
        }

    updated = deepcopy(base_result)
    updated["routes"][best["uav_id"]] = best["new_route"]
    updated["served_tasks"].update(best["served_updates"])
    updated["unserved_tasks"] = [
        task_id for task_id in updated.get("unserved_tasks", []) if str(task_id) != emergency_id
    ]
    updated["algorithm"] = f"{base_result.get('algorithm', 'Algorithm')} + Dynamic Insertion"
    disruption = compare_plan_sequences(base_result, updated)
    response_time = (
        float(best["served_updates"][emergency_id]["start_service_time"])
        - float(emergency.get("emergency_release_time", emergency.get("earliest_start", 0.0)))
    )

    return {
        "feasible": True,
        "message": "突发任务已插入现有路径。",
        "updated_result": updated,
        "emergency_task": emergency,
        "uav_id": best["uav_id"],
        "insert_position": best["insert_position"],
        "extra_distance": round(best["extra_distance"], 2),
        "additional_delay": round(best["additional_delay"], 2),
        "affected_tasks": int(best["affected_tasks"]),
        "response_time": round(max(0.0, response_time), 2),
        "plan_disruption_degree": round(disruption, 4),
        "cost": round(best["cost"], 2),
    }


def _additional_delay(old_route: dict, new_route: dict) -> float:
    old_starts = {stop["task_id"]: float(stop["start_service_time"]) for stop in old_route.get("stops", [])}
    delay = 0.0
    for stop in new_route.get("stops", []):
        task_id = stop["task_id"]
        if task_id in old_starts:
            delay += max(0.0, float(stop["start_service_time"]) - old_starts[task_id])
    return delay


def _series_to_dict(item) -> dict:
    if isinstance(item, dict):
        return dict(item)
    return {key: item[key] for key in item.index}

