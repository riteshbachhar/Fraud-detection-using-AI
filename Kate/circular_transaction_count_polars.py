import rustworkx as rx
import polars as pl

"""
Compute circular_transaction_count using a 
calendar‑month sliding window, ensuring rustworkx is installed 

"""

def circular_transaction_feature(df:pl.DataFrame):
    # Iterate over monthly groups
    results = []
    for (year, month), group in df.group_by(["year", "month"]):
        res = circular_count_monthly(group, year, month)
        if res.height > 0:
            results.append(res)

    # Combine all results
    out_rx = pl.concat(results, how="vertical") if results else pl.DataFrame()

    # Join back to original df
    df_result = (
        df.join(out_rx, on=["Sender_account", "year", "month"], how="left")
        .with_columns(
            pl.col("circular_transaction_count").fill_null(0)
        )
    )
    return df_result

def circular_count_monthly(pdf, year, month):
    edges = list(zip(pdf["Sender_account"], pdf["Receiver_account"]))
    if not edges:
        return empty_month_frame()

    G = rx.PyDiGraph()
    node_idx = {}
    for u, v in edges:
        if u not in node_idx:
            node_idx[u] = G.add_node(u)
        if v not in node_idx:
            node_idx[v] = G.add_node(v)
        G.add_edge(node_idx[u], node_idx[v], None)

    cycles = rx.simple_cycles(G)

    counter = {}
    for cyc in cycles:
        cyc_nodes = [G[node] for node in cyc]
        for node in cyc_nodes:
            counter[node] = counter.get(node, 0) + 1

    return pl.DataFrame({
        "Sender_account": list(counter.keys()),
        "circular_transaction_count": list(counter.values()),
        "year": [year] * len(counter),
        "month": [month] * len(counter)
    })