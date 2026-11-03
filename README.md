# 🚑 Emergency Vehicle Routing: Dijkstra's Algorithm vs. Bellman-Ford

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-00897B.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-455A64.svg)](https://opensource.org/licenses/MIT)

An interactive, high-performance web application built for the **DSA-II Case Study**:
> *"Fastest Routes for Emergency Vehicles: Dijkstra's Algorithm vs. Bellman-Ford on a City Road Map"*

---

## 📌 Executive Summary & Case Study Overview

In emergency medical and fire response, every second saved directly impacts human lives. Dispatch systems require deterministic, sub-millisecond shortest path algorithms capable of navigating dynamic municipal road networks, adapting instantly to road closures, and scaling gracefully across hundreds of city junctions.

This application provides an empirical comparison between two classic single-source shortest path (SSSP) algorithms:
1. **Dijkstra's Algorithm** (Greedy approach utilizing a Min-Heap priority queue)
2. **Bellman-Ford Algorithm** (Dynamic programming approach performing $|V|-1$ iterative edge relaxation sweeps)

---

## 🗺️ City Road Network Topology

The municipal graph represents key urban facilities and road junctions:
- **Facilities:**
  - `H` = Hospital (Teal)
  - `F` = Fire Station (Purple)
  - `X` = Accident Site (Red)
- **Junctions:**
  - `A`, `B`, `C`, `D`, `E` (Slate Grey)

### Road Segments & Travel Times (Undirected, in Minutes)
| Edge Segment | Travel Time | Edge Segment | Travel Time |
|:---:|:---:|:---:|:---:|
| **H — A** | 4 min | **F — D** | 3 min |
| **H — B** | 2 min | **C — D** | 2 min |
| **A — B** | 1 min | **C — E** | 6 min |
| **A — C** | 5 min | **D — E** | 3 min |
| **B — D** | 8 min | **D — X** | 9 min |
| **B — F** | 6 min | **E — X** | 2 min |

---

## 🎯 Case Study Benchmark Results

| Scenario | Algorithm | Optimal Route | Travel Time | Edge Relaxations |
|---|---|---|---|---|
| **Default Dispatch (H ➔ X)** | **Dijkstra** | `H ➔ B ➔ D ➔ E ➔ X` | **15 minutes** | **12 relaxations** |
| **Default Dispatch (H ➔ X)** | **Bellman-Ford** | `H ➔ B ➔ D ➔ E ➔ X` | **15 minutes** | **72 relaxations** |
| **Road D — E Closed (Hazard)** | **Dijkstra** | `H ➔ B ➔ A ➔ C ➔ E ➔ X` | **16 minutes** | **11 relaxations** |
| **Road D — E Closed (Hazard)** | **Bellman-Ford** | `H ➔ B ➔ A ➔ C ➔ E ➔ X` | **16 minutes** | **66 relaxations** |

> **Key Observation:** Dijkstra's min-heap requires **6x fewer edge relaxations** than Bellman-Ford on the base map (12 vs 72), yielding an identical 15-minute route. When road `D — E` is closed, the system dynamically reroutes traffic via Junctions `A` and `C` with an impact delay of only **+1 minute**.

---

## 🖥️ App Features & Structure

The web application is partitioned into 6 intuitive tabs:

1. **📍 Route Finder:**
   - Dropdown selection of origin and target destination (default: `H ➔ X`).
   - Algorithm toggles: *Dijkstra*, *Bellman-Ford*, or *Both (Comparative View)*.
   - Fixed layout NetworkX & Matplotlib visualization showing color-coded facilities, edge weights, and amber-highlighted shortest paths.
   - High-level metric summary cards for path, time, and computational relaxations.

2. **🚧 Road Closure:**
   - Multi-select interactive road blocking.
   - Closed roads are visually rendered as dashed red lines.
   - Immediate re-calculation displaying baseline route vs. re-routed path, with delay delta metrics.

3. **🔍 Dijkstra Step Trace:**
   - An interactive slider stepping through each settled node from the min-heap.
   - Complete distance estimation table (matching Report Table 2).
   - Node-level color changes showing settled status and active exploration.

4. **⚠️ Negative Weights Demo:**
   - Visual demonstration on directed graph: $A \to B$ (2), $A \to C$ (5), $C \to B$ (-4).
   - Shows Dijkstra failing greedily ($d=2$) and Bellman-Ford succeeding ($d=1$).
   - In-depth theoretical explanation of why physical urban road weights are always non-negative ($w(u, v) > 0$).

5. **📊 Performance Comparison:**
   - Graph size slider scaling from $V=50$ to $V=800$ vertices.
   - Synthetic connected road networks generated on-the-fly ($\approx 1.5$ edges/vertex).
   - Interactive Plotly figures showing relaxation counts (log-scale) and execution runtimes in milliseconds.

6. **📖 About & Technical Analysis:**
   - Formal asymptotic complexity matrix ($O((V+E)\log V)$ vs. $O(V \cdot E)$).
   - Case study findings, algorithm trade-offs, and final municipal recommendations.

---

## 📂 Project Structure

```text
├── .streamlit/
│   └── config.toml          # Custom Teal theme configuration
├── app.py                   # Streamlit UI & interactive visualization
├── algorithms.py            # Dijkstra, Bellman-Ford, and graph utilities
├── test_algorithms.py       # Unit test suite verifying exact case study metrics
├── requirements.txt         # Core dependencies
├── README.md                # Project documentation and deployment guide
└── .gitignore               # Standard git exclusions
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the unit test suite
```bash
python test_algorithms.py
```

### 5. Launch the Streamlit application
```bash
streamlit run app.py
```
The app will automatically launch in your default browser at `http://localhost:8501`.

---

## 🌐 How to Deploy to Streamlit Community Cloud

Deploying your app for public access on **Streamlit Community Cloud** takes under 2 minutes:

### Step 1: Push your code to a public GitHub repository
1. Go to [GitHub](https://github.com/new) and create a new **public** repository (e.g., `emergency-vehicle-routing`).
2. Run the following commands in your terminal:
   ```bash
   git init
   git add .
   git commit -m "feat: complete DSA-II emergency vehicle routing application"
   git branch -M main
   git remote add origin https://github.com/<your-github-username>/emergency-vehicle-routing.git
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Community Cloud
1. Navigate to **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
2. Click the **"New app"** button in the top right corner.
3. In the deployment modal, select:
   - **Repository:** `<your-github-username>/emergency-vehicle-routing`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (optional):** Enter a custom subdomain slug (e.g., `emergency-routing-dsa`).
4. Click **"Deploy!"**.

### Step 3: Access your live app
Streamlit will install `requirements.txt` and launch your application. Your live public link will follow this structure:
```text
https://<custom-slug>.streamlit.app
# Or default:
https://<your-username>-emergency-vehicle-routing-app-<hash>.streamlit.app
```

---

## 🔬 Algorithmic Complexity Reference

| Property | Dijkstra's Algorithm | Bellman-Ford Algorithm |
|---|:---:|:---:|
| **Time Complexity (General)** | $\mathcal{O}((V + E) \log V)$ | $\mathcal{O}(V \cdot E)$ |
| **Time Complexity (Sparse $E \approx 1.5V$)** | $\mathcal{O}(V \log V)$ | $\mathcal{O}(V^2)$ |
| **Space Complexity** | $\mathcal{O}(V + E)$ | $\mathcal{O}(V + E)$ |
| **Paradigm** | Greedy with Min-Heap | Dynamic Programming |
| **Negative Edges** | ❌ Not supported ($w \ge 0$) | ✅ Supported (no negative cycles) |
| **Production Suitability** | High-performance real-time dispatch | Arbitrage / Energy networks |

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
