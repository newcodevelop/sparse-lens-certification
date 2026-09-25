import glob
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

BOUND_DIR = "./bounds/"
FIDELITY_DIR = "./final_ops/"
OUT_DIR = "./final_ops/"

# Number of post-observation doublings to show for layers whose asymptotic
# floor never falls below the uninformed baseline.
MAX_EXTRAP_DOUBLINGS = 10


def human_n(x, pos=None):
    x = float(x)
    if x >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    if x >= 1_000:
        return f"{x / 1_000:.0f}k"
    return f"{int(x)}"


def exact_log_binom(m: int, p: int) -> float:
    """Return ln(C(m,p)) stably using log-gamma."""
    m = int(m)
    p = int(p)
    if p < 0 or p > m:
        raise ValueError(f"p must satisfy 0 <= p <= m, got p={p}, m={m}")
    if p == 0 or p == m:
        return 0.0
    p = min(p, m - p)
    return (
        math.lgamma(m + 1.0)
        - math.lgamma(p + 1.0)
        - math.lgamma(m - p + 1.0)
    )


def uniform_baseline_from_B_Delta(B: float, Delta: float) -> float:
    """Recover R_unif = log2(V) from the stored smoothed-loss constants.

    B = log2(V/alpha)
    Delta = log2(1 + (1-alpha)V/alpha)

    Hence V = 1 + 2**B - 2**Delta.
    """
    vocab_size = 1.0 + (2.0 ** float(B)) - (2.0 ** float(Delta))
    if vocab_size <= 1.0:
        raise ValueError(
            f"Could not recover a valid vocabulary size from B={B}, Delta={Delta}."
        )
    return math.log2(vocab_size)


def row_log_complexity(row) -> float:
    """Read exact ln(C(m,P)) if present; otherwise recompute it exactly."""
    if "exact_log_C_m_P" in row.index and pd.notna(row["exact_log_C_m_P"]):
        return float(row["exact_log_C_m_P"])
    if "Exact_log_C_m_P" in row.index and pd.notna(row["Exact_log_C_m_P"]):
        return float(row["Exact_log_C_m_P"])
    return exact_log_binom(int(row["m"]), int(row["P"]))


def certificate_from_row(row) -> float:
    """Compute the revised empirical certificate at one observed checkpoint."""
    N = float(row["N"])
    R = float(row["R_hat_hG"])
    eps = float(row["eps_loss_hat"])
    eta = float(row["eta_hat"])
    Delta = float(row["Delta"])
    delta = float(row["delta"])
    log_C = row_log_complexity(row)

    t_eta = Delta * math.sqrt(math.log(3.0 / delta) / (2.0 * N))
    t_occ = Delta * math.sqrt(
        (log_C + math.log(3.0 / delta)) / (2.0 * N)
    )
    t_eps = Delta * math.sqrt(math.log(6.0 / delta) / (2.0 * N))

    return R + eps + eta * Delta + t_eta + t_occ + t_eps


def terminal_extrapolation_parameters(last):
    """Freeze terminal empirical terms and return C,A for U(N)=C+A/sqrt(N)."""
    R = float(last["R_hat_hG"])
    eps = float(last["eps_loss_hat"])
    eta = float(last["eta_hat"])
    Delta = float(last["Delta"])
    delta = float(last["delta"])
    log_C = row_log_complexity(last)

    C = R + eps + eta * Delta
    A = Delta * (
        math.sqrt(math.log(3.0 / delta) / 2.0)
        + math.sqrt((log_C + math.log(3.0 / delta)) / 2.0)
        + math.sqrt(math.log(6.0 / delta) / 2.0)
    )
    return C, A


def load_bound_curves():
    records = []
    paths = sorted(
        glob.glob(os.path.join(BOUND_DIR, "llama3_8b_layers_*.csv")),
        key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
    )

    if not paths:
        raise FileNotFoundError(
            f"No layerwise bound CSVs found in {BOUND_DIR!r} matching "
            "'llama3_8b_layers_*.csv'."
        )

    for path in paths:
        layer = int(os.path.basename(path).split("_")[-1].split(".")[0])
        df = pd.read_csv(path).sort_values("N").reset_index(drop=True)

        required = {
            "N",
            "R_hat_hG",
            "eps_loss_hat",
            "eta_hat",
            "B",
            "Delta",
            "delta",
            "m",
            "P",
        }
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        # Genuine observed curve: every checkpoint uses its own empirical
        # R_hat(h_G), epsilon_hat, eta_hat and its own concentration terms.
        observed = df.copy()
        observed["Bound"] = observed.apply(certificate_from_row, axis=1)

        last = df.iloc[-1]
        N_last = float(last["N"])
        R = float(last["R_hat_hG"])
        eps = float(last["eps_loss_hat"])
        eta = float(last["eta_hat"])
        B = float(last["B"])
        Delta = float(last["Delta"])
        m = int(last["m"])
        P = int(last["P"])
        log_C = row_log_complexity(last)
        baseline = uniform_baseline_from_B_Delta(B, Delta)

        # Descriptive terminal-parameter extrapolation:
        # freeze the final empirical quantities and extrapolate only the
        # analytically shrinking 1/sqrt(N) concentration terms.
        asym_floor, A = terminal_extrapolation_parameters(last)

        nstar = None
        if asym_floor < baseline:
            nstar = (A / (baseline - asym_floor)) ** 2

        def terminal_bound_at_n(n):
            return asym_floor + A / math.sqrt(float(n))

        extrap_rows = []
        n = N_last

        if nstar is not None and nstar > N_last:
            # Continue by doublings until the curve has passed the exact
            # descriptive crossing. We intentionally do NOT insert, mark,
            # annotate, or otherwise display a special crossing point.
            for _ in range(MAX_EXTRAP_DOUBLINGS):
                n *= 2.0
                extrap_rows.append(
                    {"Layer": layer, "N": n, "Bound": terminal_bound_at_n(n)}
                )
                if n >= nstar:
                    break
        elif nstar is None:
            # No finite crossing: show the asymptotic approach for a fixed
            # number of doublings, just as in the original figure logic.
            for _ in range(MAX_EXTRAP_DOUBLINGS):
                n *= 2.0
                extrap_rows.append(
                    {"Layer": layer, "N": n, "Bound": terminal_bound_at_n(n)}
                )

        extrap = pd.DataFrame(extrap_rows, columns=["Layer", "N", "Bound"])

        records.append(
            {
                "layer": layer,
                "observed": observed[["N", "Bound"]],
                "extrap": extrap,
                "P": P,
                "m": m,
                "P_over_m": P / m,
                "log_C_m_P": log_C,
                "R": R,
                "eps": eps,
                "eta": eta,
                "Delta": Delta,
                "bound_at_last_observed_N": float(observed["Bound"].iloc[-1]),
                "asym_floor": asym_floor,
                "nstar": nstar,
                "baseline": baseline,
            }
        )

    # All layers use the same base model/vocabulary, so the baseline should
    # agree up to floating-point noise.
    baselines = [rec["baseline"] for rec in records]
    if max(baselines) - min(baselines) > 1e-8:
        raise ValueError(f"Inconsistent Llama uniform baselines across layers: {baselines}")

    return records


def make_bound_figure(records):
    fig, ax = plt.subplots(figsize=(10.2, 6.4))

    for rec in records:
        obs = rec["observed"]
        extra = rec["extrap"]
        layer = rec["layer"]

        # Observed and extrapolated portions are deliberately rendered as one
        # continuous, same-style curve. There is no visual marker indicating
        # the start of extrapolation and no crossing-point marker/annotation.
        if not extra.empty:
            x = obs["N"].tolist() + extra["N"].tolist()
            y = obs["Bound"].tolist() + extra["Bound"].tolist()
        else:
            x = obs["N"].tolist()
            y = obs["Bound"].tolist()

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            label=f"Layer {layer}",
        )

    baseline = records[0]["baseline"]
    ax.axhline(
        baseline,
        linestyle=":",
        linewidth=2.0,
        color="black",
        label=f"Baseline = {baseline:.2f} bits",
    )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(human_n))
    ax.set_xlabel("Sample size $N$")
    ax.set_ylabel("Total bound (bits)")
    ax.set_title("LLaMA-3-8B bounds across layers")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=3, fontsize=8.5, frameon=True)

    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "llama_bound_layers.png")
    out_pdf = os.path.join(OUT_DIR, "llama_bound_layers.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


def load_fidelity_curves(records):
    # Keep the original flat-tail fidelity extrapolation. The tail is extended
    # only to the descriptive N* implied by the bound extrapolation.
    nstar_map = {
        rec["layer"]: rec["nstar"]
        for rec in records
        if rec["nstar"] is not None
    }

    layers = [4, 8, 12, 16, 20, 24, 28, 30]
    fidelities = []
    for layer in layers:
        path = os.path.join(FIDELITY_DIR, f"output_fidelity_layer_{layer}.csv")
        if not os.path.exists(path):
            alt = os.path.join(
                FIDELITY_DIR, f"output_fidelity_layer_{layer} (1).csv"
            )
            path = alt if os.path.exists(alt) else path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing fidelity CSV for layer {layer}: {path}")

        df = pd.read_csv(path).sort_values("N").reset_index(drop=True)

        nstar = nstar_map.get(layer, None)
        extrap = []
        if nstar is not None and nstar > float(df["N"].iloc[-1]):
            n = float(df["N"].iloc[-1])
            last = df.iloc[-1]
            while n < nstar:
                n *= 2.0
                row = last.copy()
                row["N"] = min(n, nstar)
                extrap.append(row)

        extrap_df = pd.DataFrame(extrap)
        fidelities.append({"layer": layer, "observed": df, "extrap": extrap_df})

    return fidelities


def make_fidelity_figure(fidelities):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), sharex=True)
    panels = [
        ("KL_M_vs_SoM", r"KL$(M\,\|\,S\circ M)$"),
        ("Top1Agree_M_vs_SoM", r"Top-1 agreement $(M, S\circ M)$"),
        ("AbsGoldLogProbDiff_M_vs_SoM", r"$|\Delta \log p_{\mathrm{gold}}|$"),
        ("Loss_SoM", r"Loss$(S\circ M)$"),
    ]

    for ax, (col, title) in zip(axes.flatten(), panels):
        for item in fidelities:
            layer = item["layer"]
            obs = item["observed"]
            extra = item["extrap"]

            # Preserve the requested flat-tail fidelity extrapolation and
            # render it seamlessly with the observed portion.
            if not extra.empty:
                x = obs["N"].tolist() + extra["N"].tolist()
                y = obs[col].tolist() + extra[col].tolist()
            else:
                x = obs["N"].tolist()
                y = obs[col].tolist()

            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.2,
                markersize=5.2,
                label=f"Layer {layer}",
            )

        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(FuncFormatter(human_n))

    axes[1, 0].set_xlabel("Sample size $N$")
    axes[1, 1].set_xlabel("Sample size $N$")
    axes[0, 0].set_ylabel("Value")
    axes[1, 0].set_ylabel("Value")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("LLaMA-3-8B late-layer output fidelity", y=1.06, fontsize=13)
    fig.tight_layout()

    out_png = os.path.join(OUT_DIR, "llama_output_fidelity_layers.png")
    out_pdf = os.path.join(OUT_DIR, "llama_output_fidelity_layers.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    bound_records = load_bound_curves()
    out1 = make_bound_figure(bound_records)

    fidelities = load_fidelity_curves(bound_records)
    out2 = make_fidelity_figure(fidelities)

    print("Saved:", out1, out2)
    print("\nLayerwise terminal/extrapolation summary:")
    for rec in bound_records:
        nstar_text = "Never" if rec["nstar"] is None else f"{rec['nstar']:.2f}"
        print(
            f"Layer {rec['layer']:>2}: "
            f"P/m={rec['P_over_m']:.6f}, "
            f"lnC={rec['log_C_m_P']:.3f}, "
            f"Bound@last={rec['bound_at_last_observed_N']:.4f}, "
            f"AsymFloor={rec['asym_floor']:.4f}, "
            f"Descriptive N*={nstar_text}"
        )


if __name__ == "__main__":
    main()
