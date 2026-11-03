"""
app.py - Streamlit Web Application for DSA-II Case Study:
'Fastest Routes for Emergency Vehicles: Dijkstra's Algorithm vs Bellman-Ford on a City Road Map'
"""

import os
import tempfile
import time
from typing import Dict, List, Tuple, Optional, Any

# Ensure Matplotlib can write cache files in sandboxed/cloud environments
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from algorithms import (
    PLACES,
    PLACE_LABELS,
    DEFAULT_ROADS,
    FIXED_LAYOUT,
    NODE_THEME,
    PATH_COLOR,
    EDGE_COLOR,
    CLOSED_ROAD_COLOR,
    make_graph,
    dijkstra_search,
    bellman_ford_search,
    rebuild_path,
    calculate_path_time,
    generate_random_connected_graph,
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Emergency Vehicle Routing: Dijkstra vs Bellman-Ford",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished cards, metrics, and badges
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        color: #475569;
        font-weight: 600;
        font-size: 0.85rem;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #00897B;
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    /* Custom Alert / Callout Badges */
    .emergency-banner {
        background: linear-gradient(135deg, #00897B 0%, #00695C 100%);
        color: white;
        padding: 1.2rem 1.6rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 137, 123, 0.2);
    }
    .emergency-banner h1 {
        color: white !important;
        margin: 0 0 0.4rem 0;
        font-size: 1.8rem;
    }
    .emergency-banner p {
        color: #E0F2F1 !important;
        margin: 0;
        font-size: 0.95rem;
    }
    .route-pill {
        display: inline-block;
        background-color: #E0F2F1;
        color: #004D40;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-family: monospace;
        margin-right: 6px;
    }
    .legend-dot {
        height: 12px;
        width: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.markdown(
    """
    <div class="emergency-banner">
        <h1>🚑 Emergency Dispatch: Dijkstra vs. Bellman-Ford</h1>
        <p>DSA-II Case Study • Fastest Routes for Emergency Vehicles on a City Road Map</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# GRAPH VISUALIZATION HELPER
# -----------------------------------------------------------------------------
def plot_city_network(
    city: Dict[str, Dict[str, int]],
    path: Optional[List[str]] = None,
    blocked_edges: Optional[List[Tuple[str, str]]] = None,
    step_highlight: Optional[Dict[str, Any]] = None,
    title: str = "City Road Map & Emergency Route"
) -> plt.Figure:
    """
    Renders the city road graph using Matplotlib and NetworkX
    with the exact report layout and color hierarchy.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=160)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')

    G = nx.Graph()
    # Add all places
    for p in PLACES:
        G.add_node(p)

    # Add all original roads to graph
    for u, v, w in DEFAULT_ROADS:
        G.add_edge(u, v, weight=w)

    blocked_set = set()
    if blocked_edges:
        for b in blocked_edges:
            blocked_set.add((b[0], b[1]))
            blocked_set.add((b[1], b[0]))

    # Categorize edges: regular active, blocked, and path edges
    path_edges = []
    if path and len(path) >= 2:
        for i in range(len(path) - 1):
            path_edges.append((path[i], path[i + 1]))
            path_edges.append((path[i + 1], path[i]))
    path_set = set(path_edges)

    active_edges = []
    closed_edges = []
    for u, v in G.edges():
        if (u, v) in blocked_set:
            closed_edges.append((u, v))
        else:
            active_edges.append((u, v))

    # 1. Draw regular active edges
    nx.draw_networkx_edges(
        G,
        FIXED_LAYOUT,
        edgelist=active_edges,
        ax=ax,
        edge_color=EDGE_COLOR,
        width=2.5,
        alpha=0.85
    )

    # 2. Draw blocked / closed roads as dashed red lines
    if closed_edges:
        nx.draw_networkx_edges(
            G,
            FIXED_LAYOUT,
            edgelist=closed_edges,
            ax=ax,
            edge_color=CLOSED_ROAD_COLOR,
            width=2.8,
            style='dashed',
            alpha=0.9
        )

    # 3. Draw shortest path in prominent amber
    if path and len(path) >= 2:
        highlight_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(
            G,
            FIXED_LAYOUT,
            edgelist=highlight_edges,
            ax=ax,
            edge_color=PATH_COLOR,
            width=6.0,
            alpha=0.95
        )

    # 4. Determine node colors
    node_colors = []
    for node in G.nodes():
        if step_highlight and 'settled_nodes' in step_highlight:
            # Highlight settled vs unsettled in step trace
            if node == step_highlight.get('current_node'):
                node_colors.append('#F59E0B')  # Current settled node in gold
            elif node in step_highlight['settled_nodes']:
                node_colors.append(NODE_THEME[node])
            else:
                node_colors.append('#94A3B8')  # Unsettled grey
        else:
            node_colors.append(NODE_THEME[node])

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        FIXED_LAYOUT,
        ax=ax,
        node_color=node_colors,
        node_size=1200,
        edgecolors='#FFFFFF',
        linewidths=2.8
    )

    # If step highlight, draw outer ring around current settled node
    if step_highlight and step_highlight.get('current_node'):
        curr = step_highlight['current_node']
        nx.draw_networkx_nodes(
            G,
            FIXED_LAYOUT,
            nodelist=[curr],
            ax=ax,
            node_color='none',
            node_size=1700,
            edgecolors='#D97706',
            linewidths=3.0
        )

    # 5. Draw node labels
    nx.draw_networkx_labels(
        G,
        FIXED_LAYOUT,
        ax=ax,
        font_size=12,
        font_color='#FFFFFF',
        font_weight='bold',
        font_family='sans-serif'
    )

    # 6. Draw edge weight labels (travel times in minutes)
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        is_closed = (u, v) in blocked_set
        w_val = d['weight']
        edge_labels[(u, v)] = f"{w_val}m ✕" if is_closed else f"{w_val} min"

    label_colors = {}
    for edge in G.edges():
        if edge in blocked_set or (edge[1], edge[0]) in blocked_set:
            label_colors[edge] = CLOSED_ROAD_COLOR
        elif edge in path_set:
            label_colors[edge] = '#92400E'
        else:
            label_colors[edge] = '#1E293B'

    # Matplotlib edge label drawing
    for edge, text in edge_labels.items():
        is_closed = edge in blocked_set or (edge[1], edge[0]) in blocked_set
        bbox_style = dict(
            boxstyle='round,pad=0.25',
            facecolor='#FEE2E2' if is_closed else '#FFFFFF',
            edgecolor=CLOSED_ROAD_COLOR if is_closed else '#CBD5E1',
            alpha=0.95
        )
        nx.draw_networkx_edge_labels(
            G,
            FIXED_LAYOUT,
            edge_labels={edge: text},
            ax=ax,
            font_size=8.5,
            font_color=label_colors[edge],
            font_weight='bold',
            bbox=bbox_style
        )

    # Subtitle descriptions below nodes
    descriptions = {
        'H': 'Hospital',
        'F': 'Fire Station',
        'X': 'Accident',
    }
    for node, desc in descriptions.items():
        x, y = FIXED_LAYOUT[node]
        offset_y = -0.32 if node != 'F' else -0.32
        ax.text(
            x, y + offset_y, desc,
            fontsize=8.5, fontweight='bold', color=NODE_THEME[node],
            ha='center', va='top',
            bbox=dict(boxstyle='square,pad=0.15', facecolor='#FFFFFF', edgecolor='none', alpha=0.85)
        )

    ax.set_title(title, fontsize=13, fontweight='bold', color='#0F172A', pad=14)
    ax.set_xlim(-0.3, 9.6)
    ax.set_ylim(-0.3, 4.3)
    ax.axis('off')
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Dispatch Settings")

# Source and Destination Dropdowns (Default: Hospital H -> Accident Site X)
col_src, col_dst = st.sidebar.columns(2)
with col_src:
    source_node = st.selectbox(
        "Source Origin",
        options=PLACES,
        index=PLACES.index('H'),
        format_func=lambda x: f"{x} - {PLACE_LABELS[x].split(' ')[0]}"
    )
with col_dst:
    dest_node = st.selectbox(
        "Target Destination",
        options=PLACES,
        index=PLACES.index('X'),
        format_func=lambda x: f"{x} - {PLACE_LABELS[x].split(' ')[0]}"
    )

# Algorithm Choice
selected_algo = st.sidebar.radio(
    "Routing Algorithm",
    options=["Dijkstra", "Bellman-Ford", "Both (Comparative)"],
    index=0,
    help="Select which shortest path algorithm to execute on the city graph."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Map Legend:**
    - <span class="legend-dot" style="background-color:#00897B;"></span> **Teal**: Hospital (H)
    - <span class="legend-dot" style="background-color:#8E24AA;"></span> **Purple**: Fire Station (F)
    - <span class="legend-dot" style="background-color:#E53935;"></span> **Red**: Accident Site (X)
    - <span class="legend-dot" style="background-color:#455A64;"></span> **Slate**: Road Junctions (A-E)
    - <span class="legend-dot" style="background-color:#FFB300;"></span> **Amber Line**: Shortest Route
    - <span class="legend-dot" style="background-color:#D32F2F;"></span> **Dashed Red**: Blocked Road
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# APPLICATION TABS
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "📍 Route Finder",
    "🚧 Road Closure",
    "🔍 Dijkstra Step Trace",
    "⚠️ Negative Weights Demo",
    "📊 Performance Comparison",
    "📖 About & Analysis"
])


# =============================================================================
# TAB 1: ROUTE FINDER
# =============================================================================
with tabs[0]:
    st.subheader("Urban Emergency Vehicle Routing")
    st.caption("Calculate optimal navigation paths and monitor algorithmic relaxation efficiency.")

    city_graph = make_graph(DEFAULT_ROADS)

    # Compute Dijkstra
    d_best, d_parent, d_rel, _ = dijkstra_search(city_graph, source_node)
    d_path = rebuild_path(d_parent, dest_node, start=source_node)
    d_time = d_best[dest_node] if d_path else None

    # Compute Bellman-Ford
    bf_best, bf_parent, bf_rel = bellman_ford_search(city_graph, source_node)
    bf_path = rebuild_path(bf_parent, dest_node, start=source_node)
    bf_time = bf_best[dest_node] if bf_path else None

    # Handle Reachability
    if not d_path or d_time == float('inf'):
        st.error(f"❌ Destination **{dest_node}** is unreachable from **{source_node}**.")
    else:
        path_str = " ➔ ".join([f"**{p}**" for p in d_path])

        if selected_algo == "Dijkstra":
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(label="Optimal Route", value=f"{' ➔ '.join(d_path)}")
            with c2:
                st.metric(label="Total Travel Time", value=f"{int(d_time)} Minutes")
            with c3:
                st.metric(label="Dijkstra Relaxations", value=f"{d_rel} Edges Examined")

        elif selected_algo == "Bellman-Ford":
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(label="Optimal Route", value=f"{' ➔ '.join(bf_path)}")
            with c2:
                st.metric(label="Total Travel Time", value=f"{int(bf_time)} Minutes")
            with c3:
                st.metric(label="Bellman-Ford Relaxations", value=f"{bf_rel} Edge Relaxations")

        else:  # Both (Comparative)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric(label="Optimal Route", value=f"{' ➔ '.join(d_path)}")
            with c2:
                st.metric(label="Travel Time", value=f"{int(d_time)} Minutes")
            with c3:
                st.metric(label="Dijkstra Relaxations", value=f"{d_rel}")
            with c4:
                st.metric(label="Bellman-Ford Relaxations", value=f"{bf_rel}", delta=f"{bf_rel - d_rel} extra", delta_color="inverse")

            st.info(
                f"💡 **Case Study Insight:** Both algorithms identified the identical optimal route (**{' ➔ '.join(d_path)}** in {int(d_time)} min). "
                f"However, Bellman-Ford performed **{bf_rel}** edge evaluations compared to Dijkstra's **{d_rel}** relaxations "
                f"— a **{bf_rel / d_rel:.1f}x reduction** in computational work achieved by Dijkstra's min-heap priority queue."
            )

        # Plot network
        fig = plot_city_network(
            city_graph,
            path=d_path,
            title=f"Emergency Route from {PLACE_LABELS[source_node]} to {PLACE_LABELS[dest_node]}"
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# =============================================================================
# TAB 2: ROAD CLOSURE
# =============================================================================
with tabs[1]:
    st.subheader("Dynamic Road Closures & Re-routing")
    st.caption("Simulate traffic accidents, construction, or hazard blockages and observe live emergency re-routing.")

    road_options = [f"{u} — {v} ({w} min)" for u, v, w in DEFAULT_ROADS]
    road_map = {f"{u} — {v} ({w} min)": (u, v) for u, v, w in DEFAULT_ROADS}

    # Case study quick selector
    c_btn1, c_btn2 = st.columns([1, 3])
    with c_btn1:
        load_case_study = st.button("🚨 Block Road D — E (Case Study)")

    default_selected = ["D — E (3 min)"] if load_case_study else []

    selected_closure_labels = st.multiselect(
        "Select Roads to Block / Close:",
        options=road_options,
        default=default_selected,
        help="Choose one or more road segments to simulate complete closure."
    )

    blocked_tuples = [road_map[label] for label in selected_closure_labels]

    # Baseline graph & route
    base_graph = make_graph(DEFAULT_ROADS)
    base_best, base_parent, _, _ = dijkstra_search(base_graph, source_node)
    base_path = rebuild_path(base_parent, dest_node, start=source_node)
    base_time = base_best[dest_node] if base_path else None

    # Closed graph & route
    closed_graph = make_graph(DEFAULT_ROADS, blocked=blocked_tuples)
    new_best, new_parent, new_rel, _ = dijkstra_search(closed_graph, source_node)
    new_path = rebuild_path(new_parent, dest_node, start=source_node)
    new_time = new_best[dest_node] if new_path else None

    # Comparison Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        if base_path:
            st.metric("Baseline Route", " ➔ ".join(base_path), f"{int(base_time)} min")
        else:
            st.metric("Baseline Route", "Unreachable")

    with m2:
        if new_path:
            st.metric("Re-routed Path", " ➔ ".join(new_path), f"{int(new_time)} min")
        else:
            st.metric("Re-routed Path", "BLOCKED / NO ROUTE", "Disconnected", delta_color="inverse")

    with m3:
        if base_time is not None and new_time is not None and new_time != float('inf'):
            diff = int(new_time - base_time)
            st.metric("Impact Delay", f"+{diff} min" if diff > 0 else "0 min", delta=f"{diff} min delay" if diff > 0 else "No delay", delta_color="inverse")
        else:
            st.metric("Impact Delay", "Infinite (Unreachable)", delta_color="inverse")

    if new_path and base_path:
        if new_path != base_path:
            st.warning(
                f"⚠️ **Re-routing Active:** Path diverted from `{' ➔ '.join(base_path)}` ({int(base_time)} min) to "
                f"`{' ➔ '.join(new_path)}` ({int(new_time)} min), adding **+{int(new_time - base_time)} min** to response time."
            )
        else:
            st.success("✅ **Closure Impact:** The selected closure does not affect the optimal shortest path.")
    elif not new_path:
        st.error(f"🚨 **Emergency Alert:** No viable route exists between {source_node} and {dest_node} due to road closures!")

    # Plot graph with closed roads dashed in red
    fig_closed = plot_city_network(
        closed_graph,
        path=new_path,
        blocked_edges=blocked_tuples,
        title=f"Re-routed Emergency Response ({len(selected_closure_labels)} Closed Road{'s' if len(selected_closure_labels) != 1 else ''})"
    )
    st.pyplot(fig_closed, use_container_width=True)
    plt.close(fig_closed)


# =============================================================================
# TAB 3: DIJKSTRA STEP TRACE
# =============================================================================
with tabs[2]:
    st.subheader("Dijkstra's Algorithm: Step-by-Step Distance Trace")
    st.caption("Inspect the exact distance table snapshots and trace the priority queue relaxation after each settled node.")

    city_trace = make_graph(DEFAULT_ROADS)
    _, _, _, trace_log = dijkstra_search(city_trace, source_node, keep_log=True)

    if trace_log:
        max_step = len(trace_log)
        selected_step = st.slider(
            "Step through settled nodes one by one:",
            min_value=1,
            max_value=max_step,
            value=max_step,
            help="Drag the slider to view the algorithm state after each node is settled from the min-heap."
        )

        current_log = trace_log[selected_step - 1]
        settled_node = current_log['Settled Node']
        settled_dist = current_log['Settled Dist']
        settled_set = current_log['Settled Nodes Set']

        # Step Highlight Info
        step_col1, step_col2 = st.columns([1, 2])
        with step_col1:
            st.metric("Settled Node at Step", f"{settled_node} ({PLACE_LABELS[settled_node].split(' ')[0]})")
            st.metric("Settled Minimum Distance", f"{int(settled_dist)} min")
            st.write(f"**Settled Vertices so far ({len(settled_set)}/{len(PLACES)}):**")
            st.markdown(" ".join([f"<span class='route-pill'>{n}</span>" for n in sorted(settled_set)]), unsafe_allow_html=True)

        with step_col2:
            # Build current distance estimates table for display
            curr_distances = {n: (f"{int(current_log[n])} min" if current_log[n] != float('inf') else "∞") for n in PLACES}
            df_curr = pd.DataFrame([curr_distances], index=[f"Step {selected_step}"])
            st.markdown(f"**Tentative Distance Estimates after Step {selected_step}:**")
            st.dataframe(df_curr, use_container_width=True)

        # Draw graph at this step
        fig_trace = plot_city_network(
            city_trace,
            path=None,
            step_highlight={'settled_nodes': settled_set, 'current_node': settled_node},
            title=f"Step {selected_step}/{max_step}: Node '{settled_node}' Settled (Distance = {int(settled_dist)} min)"
        )
        st.pyplot(fig_trace, use_container_width=True)
        plt.close(fig_trace)

        # Full Distance Matrix Table (matching Table 2 of the report)
        st.markdown("### 📋 Complete Distance Table (Report Table 2)")
        st.caption("Evolution of tentative distance values after settling each node from source origin.")

        rows = []
        for entry in trace_log:
            row = {
                'Step': entry['Step'],
                'Settled Node': f"{entry['Settled Node']} ({entry['Settled Dist']:.0f}m)",
            }
            for p in PLACES:
                val = entry[p]
                row[p] = f"{int(val)}" if val != float('inf') else "∞"
            rows.append(row)

        table_df = pd.DataFrame(rows)
        st.dataframe(table_df, use_container_width=True)


# =============================================================================
# TAB 4: NEGATIVE WEIGHTS DEMO
# =============================================================================
with tabs[3]:
    st.subheader("Algorithm Correctness on Negative Weights")
    st.caption("Demonstrating why Dijkstra's greedy invariant fails on negative edges while Bellman-Ford succeeds.")

    # Directed Graph: A -> B (2), A -> C (5), C -> B (-4)
    neg_graph = {
        'A': {'B': 2, 'C': 5},
        'B': {},
        'C': {'B': -4}
    }

    # Run Dijkstra
    d_neg_best, d_neg_parent, d_neg_rel, _ = dijkstra_search(neg_graph, 'A')
    # Run Bellman-Ford
    bf_neg_best, bf_neg_parent, bf_neg_rel = bellman_ford_search(neg_graph, 'A')

    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.error("❌ **Dijkstra's Algorithm (Greedy Failure)**")
        st.metric("Reported Distance to B", f"{int(d_neg_best['B'])}", delta="Incorrect (Sub-optimal)", delta_color="inverse")
        st.write(f"**Identified Path:** `A ➔ B`")
        st.caption("Dijkstra greedily settles B with cost 2 first (2 < 5). Because B is marked as settled, Dijkstra never revisits B when the path through C yields a lower total cost.")

    with col_res2:
        st.success("✅ **Bellman-Ford Algorithm (Optimal Convergence)**")
        st.metric("Reported Distance to B", f"{int(bf_neg_best['B'])}", delta="Optimal (Correct)", delta_color="normal")
        st.write(f"**Identified Path:** `A ➔ C ➔ B` (Cost: 5 - 4 = 1)")
        st.caption("Bellman-Ford relaxes all edges across |V|-1 iterations regardless of greedy ordering, correctly capturing the negative-edge reduction.")

    # Visual representation of the 3-node graph
    fig_neg, ax_neg = plt.subplots(figsize=(8, 3.2), dpi=150)
    fig_neg.patch.set_facecolor('#FFFFFF')
    ax_neg.set_facecolor('#F8FAFC')

    G_neg = nx.DiGraph()
    G_neg.add_node('A', pos=(1, 2))
    G_neg.add_node('B', pos=(5, 2))
    G_neg.add_node('C', pos=(3, 3.5))

    pos_neg = {'A': (1.0, 1.5), 'B': (5.0, 1.5), 'C': (3.0, 3.0)}
    G_neg.add_edge('A', 'B', weight=2)
    G_neg.add_edge('A', 'C', weight=5)
    G_neg.add_edge('C', 'B', weight=-4)

    # Draw nodes
    nx.draw_networkx_nodes(G_neg, pos_neg, ax=ax_neg, node_color=['#00897B', '#E53935', '#455A64'], node_size=1300)
    nx.draw_networkx_labels(G_neg, pos_neg, ax=ax_neg, font_size=12, font_color='white', font_weight='bold')

    # Draw directed edges
    nx.draw_networkx_edges(
        G_neg, pos_neg, ax=ax_neg,
        arrows=True, arrowsize=20, arrowstyle='-|>',
        edge_color=['#94A3B8', '#94A3B8', '#E53935'],
        connectionstyle='arc3,rad=0.08',
        width=2.5
    )

    # Edge labels
    neg_edge_labels = {
        ('A', 'B'): '+2',
        ('A', 'C'): '+5',
        ('C', 'B'): '-4 (Negative Weight!)'
    }
    nx.draw_networkx_edge_labels(
        G_neg, pos_neg, edge_labels=neg_edge_labels, ax=ax_neg,
        font_size=10, font_weight='bold', font_color='#0F172A',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFFFFF', edgecolor='#CBD5E1')
    )

    ax_neg.set_xlim(0.2, 5.8)
    ax_neg.set_ylim(0.8, 3.8)
    ax_neg.axis('off')
    ax_neg.set_title("Directed Graph Counterexample: A ➔ B (2), A ➔ C (5), C ➔ B (-4)", fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    st.pyplot(fig_neg, use_container_width=True)
    plt.close(fig_neg)

    st.markdown(
        """
        ### 🧠 Theoretical Explanation
        - **Dijkstra's Non-Negative Invariant:** Dijkstra's mathematical correctness relies on the assumption that adding an edge to a path can never decrease the path's total weight: $\\forall (u, v), w(u, v) \\ge 0$. Under this invariant, once a node is popped from the min-heap, its tentative distance is finalized and guaranteed to be optimal.
        - **Why Greedy Fails:** When $w(C, B) = -4$, the property is violated. Path $A \\to C$ costs $5 > 2$, but the subsequent reduction yields $5 + (-4) = 1 < 2$. Because Dijkstra settled $B$ at cost $2$ and marked it complete, it failed to update $B$.
        - **Application to Emergency Vehicles:** Physical travel times on city road networks are **strictly non-negative** ($t > 0$). Negative weights do not exist in urban road navigation, rendering Dijkstra's algorithm 100% sound and optimal for emergency routing.
        """
    )


# =============================================================================
# TAB 5: PERFORMANCE COMPARISON
# =============================================================================
with tabs[4]:
    st.subheader("Empirical Performance: Dijkstra vs. Bellman-Ford")
    st.caption("Benchmarking computational relaxation overhead and runtime scaling across synthetic city graphs.")

    col_bench1, col_bench2 = st.columns([2, 1])
    with col_bench1:
        max_vertices = st.slider(
            "Maximum Number of Vertices (V):",
            min_value=50,
            max_value=800,
            value=400,
            step=50,
            help="Benchmark graph sizes from 50 up to this selected maximum."
        )
    with col_bench2:
        st.write("")
        st.write("")
        run_benchmark_btn = st.button("🚀 Run Scalability Benchmark", type="primary")

    @st.cache_data(show_spinner=False)
    def compute_scalability_benchmark(max_v: int) -> pd.DataFrame:
        # Determine test sizes
        sizes = [50]
        step = 50 if max_v <= 300 else 100
        curr = 100
        while curr <= max_v:
            sizes.append(curr)
            curr += step
        if sizes[-1] != max_v:
            sizes.append(max_v)

        data = []
        for v in sizes:
            # Generate synthetic city network with ~1.5 edges per vertex
            g = generate_random_connected_graph(v, edge_factor=1.5, min_weight=1, max_weight=20, seed=42 + v)
            total_edges = sum(len(neighbors) for neighbors in g.values()) // 2

            # Time Dijkstra
            t0 = time.perf_counter()
            _, _, d_rel, _ = dijkstra_search(g, 0)
            d_time_ms = (time.perf_counter() - t0) * 1000

            # Time Bellman-Ford
            t0 = time.perf_counter()
            _, _, bf_rel = bellman_ford_search(g, 0)
            bf_time_ms = (time.perf_counter() - t0) * 1000

            data.append({
                'Vertices (V)': v,
                'Edges (E)': total_edges,
                'Dijkstra Relaxations': d_rel,
                'Bellman-Ford Relaxations': bf_rel,
                'Dijkstra Time (ms)': round(d_time_ms, 3),
                'Bellman-Ford Time (ms)': round(bf_time_ms, 3),
                'Relaxation Ratio (BF/Dijkstra)': round(bf_rel / max(1, d_rel), 1),
            })

        return pd.DataFrame(data)

    if run_benchmark_btn or 'bench_df' not in st.session_state:
        with st.spinner("Generating connected graphs and benchmarking algorithms..."):
            st.session_state.bench_df = compute_scalability_benchmark(max_vertices)

    bench_df = st.session_state.bench_df

    # Display Benchmark Table
    st.markdown("### 📊 Benchmark Results Table")
    st.dataframe(bench_df, use_container_width=True)

    # Plotly Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_rel = go.Figure()
        fig_rel.add_trace(go.Scatter(
            x=bench_df['Vertices (V)'],
            y=bench_df['Dijkstra Relaxations'],
            mode='lines+markers',
            name="Dijkstra's Algorithm",
            line=dict(color='#00897B', width=3),
            marker=dict(size=7)
        ))
        fig_rel.add_trace(go.Scatter(
            x=bench_df['Vertices (V)'],
            y=bench_df['Bellman-Ford Relaxations'],
            mode='lines+markers',
            name="Bellman-Ford",
            line=dict(color='#E53935', width=3, dash='dash'),
            marker=dict(size=7)
        ))
        fig_rel.update_layout(
            title="<b>Edge Relaxations vs. Graph Size</b> (Log Scale)",
            xaxis_title="Number of Vertices (V)",
            yaxis_title="Total Relaxations (Log Scale)",
            yaxis_type="log",
            template="plotly_white",
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
            height=380,
            margin=dict(l=40, r=30, t=50, b=40)
        )
        st.plotly_chart(fig_rel, use_container_width=True)

    with chart_col2:
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=bench_df['Vertices (V)'],
            y=bench_df['Dijkstra Time (ms)'],
            mode='lines+markers',
            name="Dijkstra's Algorithm",
            line=dict(color='#00897B', width=3),
            marker=dict(size=7)
        ))
        fig_time.add_trace(go.Scatter(
            x=bench_df['Vertices (V)'],
            y=bench_df['Bellman-Ford Time (ms)'],
            mode='lines+markers',
            name="Bellman-Ford",
            line=dict(color='#E53935', width=3, dash='dash'),
            marker=dict(size=7)
        ))
        fig_time.update_layout(
            title="<b>Execution Runtime (ms) vs. Graph Size</b>",
            xaxis_title="Number of Vertices (V)",
            yaxis_title="Execution Time (Milliseconds)",
            template="plotly_white",
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
            height=380,
            margin=dict(l=40, r=30, t=50, b=40)
        )
        st.plotly_chart(fig_time, use_container_width=True)

    st.markdown(
        """
        > **Key Takeaway:** Because urban road maps are sparse graphs ($E \\approx 1.5V$), Dijkstra with a binary min-heap executes in $\\mathcal{O}((V + E) \\log V) \\approx \\mathcal{O}(V \\log V)$ time. In contrast, Bellman-Ford performs $|V|-1$ passes across all edges, requiring $\\mathcal{O}(V \\cdot E) \\approx \\mathcal{O}(V^2)$ time. As city size scales from $V=50$ to $V=800$, Bellman-Ford's relaxation count surges by orders of magnitude.
        """
    )


# =============================================================================
# TAB 6: ABOUT & ANALYSIS
# =============================================================================
with tabs[5]:
    st.subheader("DSA-II Case Study: Technical Report & Analysis")
    st.markdown(
        """
        ### 🚑 Case Study Context
        When an emergency vehicle (ambulance or fire engine) is dispatched, navigation software must calculate the guaranteed fastest route in sub-millisecond time. Road conditions are dynamic: accidents, heavy traffic, and infrastructure failures frequently cause sudden road closures.

        This case study evaluates two single-source shortest path algorithms on a model municipal road network:
        1. **Dijkstra's Algorithm** (Greedy approach with a Binary Min-Heap)
        2. **Bellman-Ford Algorithm** (Dynamic programming / edge relaxation passes)
        """
    )

    st.markdown("### 📊 Algorithmic Complexity Comparison")
    comp_data = {
        "Metric / Property": [
            "Time Complexity (General)",
            "Time Complexity (Sparse Graph: E ≈ 1.5 V)",
            "Space Complexity",
            "Algorithmic Paradigm",
            "Data Structures Used",
            "Handles Negative Edge Weights?",
            "Detects Negative-Weight Cycles?",
            "Early Termination Support",
            "Emergency Routing Suitability"
        ],
        "Dijkstra's Algorithm": [
            "O((V + E) log V)",
            "O(V log V)",
            "O(V + E)",
            "Greedy Method",
            "Binary Min-Heap (heapq) + Adjacency List",
            "No (requires non-negative edge weights)",
            "No (assumes w(u,v) ≥ 0)",
            "Yes (stops when target is settled)",
            "⭐⭐⭐⭐⭐ Optimal for Real-Time Dispatch"
        ],
        "Bellman-Ford Algorithm": [
            "O(V · E)",
            "O(V²)",
            "O(V + E)",
            "Dynamic Programming",
            "Edge List / Adjacency List",
            "Yes (arbitrary real edge weights)",
            "Yes (verifies on |V|-th pass)",
            "Yes (early exit if no relaxations in pass)",
            "⭐⭐ Inefficient for Large Road Networks"
        ]
    }
    st.table(pd.DataFrame(comp_data))

    st.markdown(
        """
        ### 🏁 Final Conclusions & Recommendations
        1. **Runtime Efficiency:** On the 8-node case study road map, Dijkstra identifies the optimal path ($H \\to B \\to D \\to E \\to X$ in 15 min) with just **12 edge relaxations**, whereas Bellman-Ford requires **72 relaxations** (a 6x difference). On larger city graphs ($V \\ge 400$), Dijkstra outperforms Bellman-Ford by over 20x.
        2. **Dynamic Adaptability:** When road $D-E$ is blocked, both algorithms promptly re-route via $H \\to B \\to A \\to C \\to E \\to X$ (16 min, adding +1 min delay). Road closures are seamlessly handled in $O(1)$ by removing the edge from the adjacency list.
        3. **Soundness:** Since travel times on physical roads cannot be negative, Dijkstra's non-negative requirement is always satisfied.
        4. **Production Recommendation:** **Dijkstra's algorithm** is the clear choice for emergency dispatch engines, navigation systems (OpenStreetMap / Google Maps), and real-time municipal routing.
        """
    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.85rem;'>"
    "DSA-II Case Study • Emergency Vehicle Routing • Built with Streamlit, NetworkX, Matplotlib & Plotly"
    "</div>",
    unsafe_allow_html=True
)
