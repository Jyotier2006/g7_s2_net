import heapq
import time

def run_dijkstra(graph_obj, source, dest):
    n = graph_obj.n
    graph = graph_obj.graph
    dist = [float("inf")] * n
    parent = [-1] * n
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

        for v, w in graph[u]:
            new_cost = curr_cost + w

            if new_cost < dist[v]:
                dist[v] = new_cost
                parent[v] = u
                heapq.heappush(pq, (new_cost, v))
            elif new_cost == dist[v]:
                if parent[v] == -1 or u < parent[v]:
                    parent[v] = u

    runtime = time.perf_counter() - start
    path = graph_obj.reconstruct_path(parent, source, dest)

    return {
        "algorithm": "Dijkstra",
        "path": path,
        "delay": dist[dest] if path else float("inf"),
        "expanded_nodes": expanded,
        "runtime_seconds": runtime
    }
