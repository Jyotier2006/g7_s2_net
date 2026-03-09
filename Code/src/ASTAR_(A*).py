import heapq
import math
import time

def heuristic(coords, node, dest):
    x1, y1 = coords[node]
    x2, y2 = coords[dest]
    return abs(x1 - x2) + abs(y1 - y2)

def run_astar(graph_obj, source, dest):
    n = graph_obj.n
    graph = graph_obj.graph
    coords = graph_obj.coords
    g = [float("inf")] * n
    parent = [-1] * n
    g[source] = 0.0
    pq = [(heuristic(coords, source, dest), 0.0, source)]
    expanded = 0
    start = time.perf_counter()

    while pq:
        _, curr_g, u = heapq.heappop(pq)

        if curr_g > g[u]:
            continue

        expanded += 1

        if u == dest:
            break

        for v, w in graph[u]:
            new_g = curr_g + w

            if new_g < g[v]:
                g[v] = new_g
                parent[v] = u
                new_f = new_g + heuristic(coords, v, dest)
                heapq.heappush(pq, (new_f, new_g, v))
            elif new_g == g[v]:
                if parent[v] == -1 or u < parent[v]:
                    parent[v] = u

    runtime = time.perf_counter() - start
    path = graph_obj.reconstruct_path(parent, source, dest)

    return {
        "algorithm": "A*",
        "path": path,
        "delay": g[dest] if path else float("inf"),
        "expanded_nodes": expanded,
        "runtime_seconds": runtime
    }
