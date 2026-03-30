# Probabilistic-Modeling-and-Analysis-of-Network-Congestion-and-Queue-Delay
This project studies network congestion and queue delay using probabilistic models. Packet arrivals and service times are treated as random events. Queueing models like M/M/1 are simulated in Python to analyze waiting time, delay, throughput, and packet loss under different traffic loads.

# Queue Delay Analysis in Networks

A congestion-aware routing project that compares **deterministic shortest-path algorithms** with a **softmax-based randomized routing strategy** on a structured network graph.

This project models a communication network as a graph where each node has a **queue length** and **service rate**. The routing cost is based on a simple queue-delay approximation:

\[
\text{Node Delay} = \frac{\text{Queue Length}}{\text{Service Rate}}
\]

Using this delay model, the project evaluates how different routing strategies behave in terms of:

- total delay
- runtime
- search effort
- route quality
- probabilistic behavior under randomized decision-making

---

## Project Overview

In real networks, shortest-hop routing is not always the best choice when congestion exists. A path with fewer hops may still experience larger delay if packets must wait in long queues. This project studies that idea by assigning queue-based delay costs to the graph and comparing three routing styles:

- **Dijkstra** – exact deterministic shortest path for non-negative costs
- **A\*** – heuristic-guided shortest path using node coordinates
- **Softmax Randomized Routing** – probabilistic next-hop selection for controlled exploration

The final goal is to understand when deterministic routing is best and when a probabilistic strategy can still produce near-optimal results while offering route diversity.

---

## Dataset and Network Model

The project uses a **synthetic 10 × 10 grid-style network** for controlled experimentation.

### Input data includes

- **100 nodes**
- **360 directed edges**
- queue lengths for each node
- service rates for each node
- source node and destination node
- node coordinates for visualization and heuristic design

### Why synthetic data?

A synthetic dataset was used because it provides:

- reproducibility
- controlled congestion settings
- fair algorithm comparison
- easy interpretation during analysis

The main dataset is stored in:

```text
Code/Data/Data.txt
```

---

## Delay Model

Each node is assigned a queueing cost:

\[
\text{delay}(v) = \frac{q_v}{\mu_v}
\]

where:

- \(q_v\) = queue length at node \(v\)
- \(\mu_v\) = service rate at node \(v\)

This delay is used as the effective cost of entering a node. The model is intentionally simple and interpretable, making it suitable for algorithm comparison and viva presentation.

---

## Algorithms Implemented

### 1. Dijkstra

Dijkstra’s algorithm is used as the **exact baseline** because all routing costs are non-negative.

**Why chosen:**
- guarantees the minimum-delay path
- standard benchmark for shortest path problems
- reliable deterministic reference for comparison

### 2. A*

A* extends shortest-path search by using a heuristic estimate of remaining distance.

**Heuristic used:**
- Manhattan distance based on node coordinates

**Why chosen:**
- reduces search effort in grid-like graphs
- keeps path quality high
- allows comparison between uninformed and informed deterministic search

### 3. Softmax Randomized Routing

This method selects the next hop probabilistically using a **softmax distribution** over candidate energies.

\[
P(i) = \frac{e^{-E_i/T}}{\sum_j e^{-E_j/T}}
\]

where:

- \(E_i\) is the candidate move energy/cost
- \(T\) is temperature

**Why chosen:**
- introduces controlled exploration
- can produce multiple reasonable routes
- helps study delay distributions, not just one fixed answer

### Temperature interpretation

- **Low temperature** → more greedy, more deterministic behavior
- **High temperature** → more random exploration

---

## Final Experimental Results

The consolidated results from the final analysis are shown below.

| Algorithm | Best Delay | Mean Delay | Success Rate | Expanded Nodes | Runtime (ms) | Path Length |
|---|---:|---:|---:|---:|---:|---:|
| Dijkstra | 36.0 | 36.00000 | 1.0 | 84.000 | 0.091095 | 19 |
| A* | 36.0 | 36.00000 | 1.0 | 76.000 | 0.098943 | 19 |
| Softmax Randomized (T=0.15) | 36.0 | 36.01725 | 1.0 | 18.000 | 115.626981 | 19 |
| Softmax Randomized (T=0.30) | 36.0 | 36.08750 | 1.0 | 18.005 | 123.340650 | 19 |
| Softmax Randomized (T=0.50) | 36.0 | 36.26825 | 1.0 | 18.059 | 118.767696 | 19 |
| Softmax Randomized (T=0.80) | 36.0 | 38.33700 | 1.0 | 18.728 | 122.422879 | 19 |
| Softmax Randomized (T=1.20) | 36.0 | 44.32075 | 1.0 | 20.554 | 144.109968 | 19 |

### Key observations

- **Dijkstra** produced the exact minimum-delay route.
- **A*** found the same optimal path while expanding fewer nodes.
- **Softmax routing** also reached the optimum in some trials, but average delay increased as temperature increased.
- The best randomized performance was obtained at **T = 0.15**.

---

## Temperature Sensitivity

| Temperature | Best Delay | Mean Delay | Std. Delay | Success Rate |
|---|---:|---:|---:|---:|
| 0.15 | 36.0 | 36.01725 | 0.091255 | 1.0 |
| 0.30 | 36.0 | 36.08750 | 0.237368 | 1.0 |
| 0.50 | 36.0 | 36.26825 | 0.790359 | 1.0 |
| 0.80 | 36.0 | 38.33700 | 15.037833 | 1.0 |
| 1.20 | 36.0 | 44.32075 | 28.207242 | 1.0 |

### Inference

As the temperature increases, the randomized router becomes less selective and more exploratory. This causes a visible increase in average delay and variance. Therefore, low-temperature softmax is best when we want a balance between route diversity and near-optimal delay.

---

## Visual Analysis

The project includes plots and output files for visual interpretation:

- delay comparison bar chart
- expanded nodes comparison
- runtime comparison
- runtime comparison in log scale
- softmax delay histogram (empirical PDF-style view)
- softmax delay CDF
- temperature sensitivity plot

These outputs are available in:

```text
Code/Experiments/
```

Important generated files include:

```text
Code/Experiments/delay_comparison.png
Code/Experiments/expanded_nodes_comparison.png
Code/Experiments/runtime_comparison.png
Code/Experiments/runtime_comparison_log.png
Code/Experiments/softmax_delay_histogram.png
Code/Experiments/softmax_delay_cdf.png
Code/Experiments/temperature_sensitivity.png
Code/Experiments/queue_delay_final_report.pdf
```

---

## Project Structure

```text
g7_s2_net-main/
├── Base-Paper/
│   └── Queuing-Network-Models-for-Delay-Analysis-of-1-Mul.pdf
├── Code/
│   ├── Data/
│   │   └── Data.txt
│   ├── Experiments/
│   │   ├── analysis.txt
│   │   ├── summary.csv
│   │   ├── temperature_sensitivity.csv
│   │   ├── delay_comparison.png
│   │   ├── expanded_nodes_comparison.png
│   │   ├── runtime_comparison.png
│   │   ├── runtime_comparison_log.png
│   │   ├── softmax_delay_histogram.png
│   │   ├── softmax_delay_cdf.png
│   │   ├── temperature_sensitivity.png
│   │   └── queue_delay_final_report.pdf
│   ├── Requirements.txt
│   └── src/
│       ├── Graph_Model.py
│       ├── Dijkstra.py
│       ├── ASTAR_(A*).py
│       ├── Compare.py
│       └── final_queue_delay_analysis.py
└── Lecture_Scribe/
```

> Note: `Lecture_Scribe/` is separate from the routing project and is not part of the main queue-delay analysis.

---

## Main Files Explained

### `Graph_Model.py`
Builds the graph using queue-delay based edge weights.

### `Dijkstra.py`
Implements deterministic shortest-path routing.

### `ASTAR_(A*).py`
Implements heuristic-based shortest-path routing using node coordinates.

### `final_queue_delay_analysis.py`
This is the **main consolidated analysis file**. It performs:

- graph construction
- deterministic routing
- softmax randomized routing
- repeated trials
- result aggregation
- plot generation
- PDF-style report generation

### `Compare.py`
An intermediate comparison file from development stages.

---

## Installation

### Requirements

```text
python>=3.10
numpy>=1.26
pandas>=2.0
networkx>=3.0
matplotlib>=3.8
seaborn>=0.13
tqdm>=4.66
```

### Install dependencies

```bash
pip install -r Code/Requirements.txt
```

---

## How to Run

Run the final consolidated script:

```bash
python Code/src/final_queue_delay_analysis.py
```

This script performs the main experiment and generates the analysis outputs in the `Code/Experiments/` folder.

---

## Expected Output

After running the final script, you should obtain:

- summary tables in CSV/text format
- comparison plots
- histogram and CDF for randomized routing
- temperature sensitivity results
- final PDF report

---

## Why This Project Is Useful

This project helps demonstrate how routing decisions can change when congestion is included in the cost function instead of only hop count.

It is useful for understanding:

- congestion-aware routing
- queue-delay based path optimization
- deterministic vs probabilistic routing
- heuristic search in networks
- reliability analysis using empirical distributions

Possible application areas:

- packet routing studies
- educational networking experiments
- algorithm analysis coursework
- congestion-aware path selection research prototypes

---

## Limitations

This project uses a **simplified queue-delay approximation** rather than a full queueing-theoretic simulator.

### Current limitations

- synthetic rather than real traffic data
- static queue lengths and service rates
- no time-varying congestion
- no packet-level or event-level simulation
- heuristic not formally scaled for all possible weighted graphs

These limitations are acceptable for algorithm comparison and academic analysis, but future work can make the model more realistic.

---

## Future Work

Potential improvements include:

- use of real network traffic traces
- dynamic queue updates over time
- packet-level simulation
- larger and irregular topologies
- reinforcement learning or adaptive routing extensions
- comparison with Bellman-Ford, BFS-based baselines, or metaheuristics
- true queueing models such as M/M/1-based delay formulations

---

## Conclusion

This project shows that:

- **Dijkstra** is the strongest choice for exact minimum-delay routing with non-negative costs.
- **A*** can achieve the same optimal result with lower search effort when a useful heuristic is available.
- **Softmax randomized routing** provides near-optimal routing at low temperature and becomes more exploratory as temperature increases.

Overall, the work demonstrates a clear and practical comparison between deterministic and probabilistic routing under a queue-delay based cost model.

---

## Authors

Group project submission for networking / queue-delay analysis.

You can add your team member names, enrollment numbers, guide name, and institute here before pushing to GitHub.

---

## License

This repository is intended for academic and educational use.

