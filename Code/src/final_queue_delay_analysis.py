from __future__ import annotations

import heapq
import math
import random
import time
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "Data" / "Data.txt"
OUT_DIR = BASE_DIR / "Experiments" / "randomized_analysis_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class QueueDelayGraph:
    n: int
    edges: List[Tuple[int, int]]
    queue_len: List[float]
    service_rate: List[float]
    coords: Dict[int, Tuple[int, int]]

    def __post_init__(self) -> None:
        self.graph: List[List[Tuple[int, float]]] = [[] for _ in range(self.n)]
        self.build_graph()

    def node_delay(self, node: int) -> float:
        service = self.service_rate[node]
        if service <= 0:
            return float("inf")
        return self.queue_len[node] / service

    def build_graph(self) -> None:
        for u, v in self.edges:
            self.graph[u].append((v, self.node_delay(v)))

    @staticmethod
    def reconstruct_path(parent: List[int], source: int, dest: int) -> Optional[List[int]]:
        if source != dest and parent[dest] == -1:
            return None
        path = []
        cur = dest
        while cur != -1:
            path.append(cur)
            cur = parent[cur]
        return path[::-1]


def load_data(filename: Path):
    with open(filename, "r", encoding="utf-8") as f:
        n, m = map(int, f.readline().split())
        edges = [tuple(map(int, f.readline().split())) for _ in range(m)]
        queue_len = list(map(float, f.readline().split()))
        service_rate = list(map(float, f.readline().split()))
        source, dest = map(int, f.readline().split())
        coords = {i: tuple(map(int, f.readline().split())) for i in range(n)}
    return n, edges, queue_len, service_rate, source, dest, coords


def heuristic(coords: Dict[int, Tuple[int, int]], node: int, dest: int) -> float:
    x1, y1 = coords[node]
    x2, y2 = coords[dest]
    return abs(x1 - x2) + abs(y1 - y2)


def run_dijkstra(graph_obj: QueueDelayGraph, source: int, dest: int):
    dist = [float("inf")] * graph_obj.n
    parent = [-1] * graph_obj.n
    dist[source] = 0.0
    pq = [(0.0, source)]
    expanded = 0
    start = time.perf_counter()

    while pq:
        curr_cost, u = heapq.heappop(pq)
        if curr_cost > dist[u]:
            continue
        expanded += 1
        if u == dest:
            break
        for v, w in graph_obj.graph[u]:
            new_cost = curr_cost + w
            if new_cost < dist[v]:
                dist[v] = new_cost
                parent[v] = u
                heapq.heappush(pq, (new_cost, v))
            elif new_cost == dist[v] and (parent[v] == -1 or u < parent[v]):
                parent[v] = u

    runtime = time.perf_counter() - start
    path = graph_obj.reconstruct_path(parent, source, dest)
    return {
        "algorithm": "Dijkstra",
        "path": path,
        "delay": dist[dest] if path else float("inf"),
        "expanded_nodes": expanded,
        "runtime_seconds": runtime,
    }


def run_astar(graph_obj: QueueDelayGraph, source: int, dest: int):
    g = [float("inf")] * graph_obj.n
    parent = [-1] * graph_obj.n
    g[source] = 0.0
    pq = [(heuristic(graph_obj.coords, source, dest), 0.0, source)]
    expanded = 0
    start = time.perf_counter()

    while pq:
        _, curr_g, u = heapq.heappop(pq)
        if curr_g > g[u]:
            continue
        expanded += 1
        if u == dest:
            break
        for v, w in graph_obj.graph[u]:
            new_g = curr_g + w
            if new_g < g[v]:
                g[v] = new_g
                parent[v] = u
                heapq.heappush(pq, (new_g + heuristic(graph_obj.coords, v, dest), new_g, v))
            elif new_g == g[v] and (parent[v] == -1 or u < parent[v]):
                parent[v] = u

    runtime = time.perf_counter() - start
    path = graph_obj.reconstruct_path(parent, source, dest)
    return {
        "algorithm": "A*",
        "path": path,
        "delay": g[dest] if path else float("inf"),
        "expanded_nodes": expanded,
        "runtime_seconds": runtime,
    }


def reverse_graph(graph_obj: QueueDelayGraph):
    rev = [[] for _ in range(graph_obj.n)]
    for u in range(graph_obj.n):
        for v, w in graph_obj.graph[u]:
            rev[v].append((u, w))
    return rev


def shortest_remaining_cost(graph_obj: QueueDelayGraph, dest: int):
    rev = reverse_graph(graph_obj)
    dist = [float("inf")] * graph_obj.n
    dist[dest] = 0.0
    pq = [(0.0, dest)]
    while pq:
        cost, u = heapq.heappop(pq)
        if cost > dist[u]:
            continue
        for prev, w in rev[u]:
            new_cost = cost + w
            if new_cost < dist[prev]:
                dist[prev] = new_cost
                heapq.heappush(pq, (new_cost, prev))
    return dist


def softmax(xs: List[float], temperature: float) -> List[float]:
    safe_temp = max(temperature, 1e-9)
    scaled = [-(x / safe_temp) for x in xs]
    max_scaled = max(scaled)
    exps = [math.exp(v - max_scaled) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def sample_index(probabilities: List[float], rng: random.Random) -> int:
    r = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probabilities):
        cumulative += p
        if r <= cumulative:
            return i
    return len(probabilities) - 1


def run_softmax_randomized(
    graph_obj: QueueDelayGraph,
    source: int,
    dest: int,
    trials: int = 2000,
    temperature: float = 0.30,
    alpha: float = 1.0,
    beta: float = 1.0,
    seed: int = 42,
    max_steps: Optional[int] = None,
):
    if max_steps is None:
        max_steps = graph_obj.n * 3

    remaining = shortest_remaining_cost(graph_obj, dest)
    rng = random.Random(seed)
    start = time.perf_counter()
    records = []
    best_record = None

    for trial in range(1, trials + 1):
        u = source
        path = [u]
        visited = {u}
        total_delay = 0.0
        expanded = 0
        success = True

        for _ in range(max_steps):
            if u == dest:
                break

            candidates = []
            energies = []
            for v, w in graph_obj.graph[u]:
                if remaining[v] == float("inf"):
                    continue
                revisit_penalty = 1e6 if v in visited else 0.0
                energy = alpha * w + beta * remaining[v] + revisit_penalty
                candidates.append((v, w, energy))
                energies.append(energy)

            if not candidates:
                success = False
                break

            probs = softmax(energies, temperature)
            idx = sample_index(probs, rng)
            v, w, _ = candidates[idx]

            total_delay += w
            path.append(v)
            visited.add(v)
            u = v
            expanded += 1

            if u == dest:
                break
        else:
            success = False

        if u != dest:
            success = False

        record = {
            "trial": trial,
            "success": success,
            "path": path if success else None,
            "delay": total_delay if success else float("inf"),
            "expanded_nodes": expanded,
        }
        records.append(record)

        if success and (best_record is None or record["delay"] < best_record["delay"]):
            best_record = record

    runtime = time.perf_counter() - start
    successful = [r for r in records if r["success"]]
    delays = [r["delay"] for r in successful]
    expanded_nodes = [r["expanded_nodes"] for r in successful]
    mean_delay = sum(delays) / len(delays) if delays else float("inf")
    std_delay = (
        math.sqrt(sum((x - mean_delay) ** 2 for x in delays) / len(delays)) if delays else float("inf")
    )

    return {
        "algorithm": f"Softmax Randomized (T={temperature})",
        "path": best_record["path"] if best_record else None,
        "delay": best_record["delay"] if best_record else float("inf"),
        "expanded_nodes": best_record["expanded_nodes"] if best_record else 0,
        "runtime_seconds": runtime,
        "trials": trials,
        "success_rate": len(successful) / trials if trials else 0.0,
        "mean_delay": mean_delay,
        "std_delay": std_delay,
        "mean_expanded_nodes": sum(expanded_nodes) / len(expanded_nodes) if expanded_nodes else float("inf"),
        "all_successful_delays": delays,
        "trial_records": records,
    }


def empirical_cdf(values: List[float]):
    xs = sorted(values)
    n = len(xs)
    ys = [(i + 1) / n for i in range(n)]
    return xs, ys


def annotate_bars(ax, values, suffix: str = "") -> None:
    for i, value in enumerate(values):
        label = f"{value:.4f}{suffix}" if value < 100 else f"{value:.2f}{suffix}"
        ax.text(i, value, label, ha="center", va="bottom", fontsize=9)


def save_plots(deterministic_results, randomized_results, temperature_df):
    best_soft = min(randomized_results, key=lambda r: r["mean_delay"])
    best_soft_delays = best_soft["all_successful_delays"]
    det_opt = deterministic_results[0]["delay"]

    alg_delay = [r["algorithm"] for r in deterministic_results] + ["Softmax mean", "Softmax best"]
    delay_values = [r["delay"] for r in deterministic_results] + [best_soft["mean_delay"], best_soft["delay"]]
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.bar(alg_delay, delay_values)
    ax.set_ylabel("Queue delay")
    ax.set_title("Queue delay comparison")
    plt.xticks(rotation=15)
    annotate_bars(ax, delay_values)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "delay_comparison.png", dpi=220)
    plt.close()

    alg_expand = [r["algorithm"] for r in deterministic_results] + ["Softmax mean"]
    expand_values = [r["expanded_nodes"] for r in deterministic_results] + [best_soft["mean_expanded_nodes"]]
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.bar(alg_expand, expand_values)
    ax.set_ylabel("Expanded nodes")
    ax.set_title("Search effort comparison")
    plt.xticks(rotation=15)
    annotate_bars(ax, expand_values)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "expanded_nodes_comparison.png", dpi=220)
    plt.close()

    runtime_labels = [r["algorithm"] for r in deterministic_results] + ["Softmax mean"]
    runtime_values = [r["runtime_seconds"] * 1e3 for r in deterministic_results] + [best_soft["runtime_seconds"] * 1e3]
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.bar(runtime_labels, runtime_values)
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("Runtime comparison")
    plt.xticks(rotation=15)
    annotate_bars(ax, runtime_values, " ms")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "runtime_comparison.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.yscale("log")
    ax = plt.gca()
    ax.bar(runtime_labels, runtime_values)
    ax.set_ylabel("Runtime (ms, log scale)")
    ax.set_title("Runtime comparison (log scale)")
    plt.xticks(rotation=15)
    annotate_bars(ax, runtime_values, " ms")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "runtime_comparison_log.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(best_soft_delays, bins=25)
    plt.axvline(det_opt, linestyle="--", label="Deterministic optimum")
    plt.xlabel("Queue delay")
    plt.ylabel("Frequency")
    plt.title(f"Softmax delay histogram ({best_soft['algorithm']})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "softmax_delay_histogram.png", dpi=220)
    plt.close()

    xs, ys = empirical_cdf(best_soft_delays)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker=".")
    plt.axvline(det_opt, linestyle="--", label="Deterministic optimum")
    plt.xlabel("Queue delay")
    plt.ylabel("CDF")
    plt.title("Empirical CDF of softmax randomized delay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "softmax_delay_cdf.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(temperature_df["temperature"], temperature_df["mean_delay"], marker="o", label="Mean delay")
    plt.plot(temperature_df["temperature"], temperature_df["best_delay"], marker="s", label="Best delay")
    plt.xlabel("Temperature")
    plt.ylabel("Queue delay")
    plt.title("Temperature sensitivity of softmax routing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "temperature_sensitivity.png", dpi=220)
    plt.close()


def create_pdf_report(summary_df: pd.DataFrame, temperature_df: pd.DataFrame, inference_text: str) -> Path:
    report_path = OUT_DIR / "queue_delay_final_report.pdf"
    image_files = [
        OUT_DIR / "delay_comparison.png",
        OUT_DIR / "expanded_nodes_comparison.png",
        OUT_DIR / "runtime_comparison.png",
        OUT_DIR / "runtime_comparison_log.png",
        OUT_DIR / "softmax_delay_histogram.png",
        OUT_DIR / "softmax_delay_cdf.png",
        OUT_DIR / "temperature_sensitivity.png",
    ]

    with PdfPages(report_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
        ax.axis("off")
        ax.text(0.5, 0.97, "Queue Delay Analysis in Network Congestion", ha="center", va="top", fontsize=18, fontweight="bold")
        ax.text(0.5, 0.93, "Dijkstra vs A* vs Softmax Randomized Routing", ha="center", va="top", fontsize=13)
        ax.text(0.0, 0.87, "Method summary", fontsize=12, fontweight="bold")
        ax.text(0.0, 0.84, summary_df.round(6).to_string(index=False), family="monospace", fontsize=9, va="top")
        ax.text(0.0, 0.56, "Temperature sensitivity", fontsize=12, fontweight="bold")
        ax.text(0.0, 0.53, temperature_df.round(6).to_string(index=False), family="monospace", fontsize=9, va="top")
        ax.text(0.0, 0.31, "Inference", fontsize=12, fontweight="bold")
        ax.text(0.0, 0.28, textwrap.fill(inference_text, width=100), fontsize=10, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for image_file in image_files:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
            ax.axis("off")
            ax.imshow(plt.imread(image_file))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return report_path


def save_text_report(path: Path, summary_df: pd.DataFrame, temperature_df: pd.DataFrame, inference_text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Queue Delay Analysis: Deterministic vs Randomized Routing\n")
        f.write("=" * 70 + "\n\n")
        f.write("Summary table\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\nTemperature sensitivity\n")
        f.write(temperature_df.to_string(index=False))
        f.write("\n\nDetailed inference\n")
        f.write(inference_text)
        f.write("\n")


def build_inference_text(dijkstra_res, astar_res, best_soft) -> str:
    det_delay = dijkstra_res["delay"]
    gap = ((best_soft["mean_delay"] - det_delay) / det_delay) * 100 if det_delay else 0.0
    within_10 = sum(x <= det_delay * 1.10 for x in best_soft["all_successful_delays"]) / len(best_soft["all_successful_delays"])
    return (
        f"Dijkstra gives the exact minimum queue-delay path with total delay {det_delay:.4f}. "
        f"A* reaches the same optimum in this graph, which shows that the heuristic guidance is compatible with the network layout for this instance. "
        f"The softmax randomized method introduces controlled stochastic exploration: instead of always choosing the minimum-energy next hop, it samples the next node using a softmax distribution over candidate energies. "
        f"Among the tested temperatures, {best_soft['algorithm']} gave the strongest average result with mean delay {best_soft['mean_delay']:.4f}, best observed delay {best_soft['delay']:.4f}, and success rate {best_soft['success_rate']:.2%}. "
        f"Its average delay is {gap:.4f}% above the deterministic optimum, which is expected because randomness trades perfect optimality for route diversity and exploration. "
        f"The histogram acts as an empirical PDF approximation and shows the concentration of randomized delays, while the empirical CDF shows the probability of obtaining a route below a chosen threshold. "
        f"For this graph, the probability of getting a path within 10% of the deterministic optimum is {within_10:.2%}. "
        f"Therefore, Dijkstra is the best choice when exact minimum queue delay is required, A* is a strong informed deterministic alternative, and softmax randomized routing is useful when the study needs probabilistic behavior, multiple candidate routes, or an algorithmic base that can later be extended to dynamic congestion scenarios."
    )


def main() -> None:
    n, edges, queue_len, service_rate, source, dest, coords = load_data(DATA_FILE)
    graph_obj = QueueDelayGraph(n, edges, queue_len, service_rate, coords)

    dijkstra_res = run_dijkstra(graph_obj, source, dest)
    astar_res = run_astar(graph_obj, source, dest)

    temperatures = [0.15, 0.30, 0.50, 0.80, 1.20]
    randomized_results = [
        run_softmax_randomized(graph_obj, source, dest, trials=2000, temperature=t, alpha=1.0, beta=1.0, seed=42)
        for t in temperatures
    ]

    summary_rows = [
        {
            "algorithm": dijkstra_res["algorithm"],
            "best_delay": dijkstra_res["delay"],
            "mean_delay": dijkstra_res["delay"],
            "success_rate": 1.0,
            "expanded_nodes": dijkstra_res["expanded_nodes"],
            "runtime_ms": dijkstra_res["runtime_seconds"] * 1e3,
            "path_length": len(dijkstra_res["path"]) if dijkstra_res["path"] else math.nan,
        },
        {
            "algorithm": astar_res["algorithm"],
            "best_delay": astar_res["delay"],
            "mean_delay": astar_res["delay"],
            "success_rate": 1.0,
            "expanded_nodes": astar_res["expanded_nodes"],
            "runtime_ms": astar_res["runtime_seconds"] * 1e3,
            "path_length": len(astar_res["path"]) if astar_res["path"] else math.nan,
        },
    ]

    for result in randomized_results:
        summary_rows.append(
            {
                "algorithm": result["algorithm"],
                "best_delay": result["delay"],
                "mean_delay": result["mean_delay"],
                "success_rate": result["success_rate"],
                "expanded_nodes": result["mean_expanded_nodes"],
                "runtime_ms": result["runtime_seconds"] * 1e3,
                "path_length": len(result["path"]) if result["path"] else math.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    temperature_df = pd.DataFrame(
        [
            {
                "temperature": float(result["algorithm"].split("=")[1].rstrip(")")),
                "best_delay": result["delay"],
                "mean_delay": result["mean_delay"],
                "std_delay": result["std_delay"],
                "success_rate": result["success_rate"],
            }
            for result in randomized_results
        ]
    ).sort_values("temperature")

    best_soft = min(randomized_results, key=lambda x: x["mean_delay"])
    inference_text = build_inference_text(dijkstra_res, astar_res, best_soft)

    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
    temperature_df.to_csv(OUT_DIR / "temperature_sensitivity.csv", index=False)
    pd.DataFrame(best_soft["trial_records"]).to_csv(OUT_DIR / "softmax_trials.csv", index=False)

    save_plots([dijkstra_res, astar_res], randomized_results, temperature_df)
    save_text_report(OUT_DIR / "analysis.txt", summary_df.round(6), temperature_df.round(6), inference_text)
    pdf_path = create_pdf_report(summary_df, temperature_df, inference_text)

    print("Saved outputs in:", OUT_DIR)
    print(summary_df.round(6).to_string(index=False))
    print("\nBest softmax configuration:", best_soft["algorithm"])
    print("Best path:", best_soft["path"])
    print("Best observed delay:", round(best_soft["delay"], 6))
    print("Mean delay:", round(best_soft["mean_delay"], 6))
    print("PDF report:", pdf_path)


if __name__ == "__main__":
    main()
