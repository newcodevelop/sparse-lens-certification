import gc
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens import SAE


print("GPT-2 layerwise exact-certificate experiment")


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

topk = 64

CONFIG = {
    # Same clean calibration budget as the revised main experiments.
    "CALIBRATION_TOKENS": 2_240_000,

    # Figure-5-like held-out sample checkpoints.
    # N denotes the number of 32-token sequences.
    "N_CHECKPOINTS": [10_000, 30_000, 50_000, 70_000, 140_000],

    "ALPHA": 0.5,
    "CONFIDENCE_DELTA": 0.05,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "TOP_K": topk,
    "SEQ_LEN": 32,
    "TOKENIZER_BUFFER_SIZE": 32_000,

    # GPT-2 layerwise sweep used in the existing appendix analysis.
    "LAYERS": [0, 2, 4, 6, 8, 10],

    "MODEL_NAME": "gpt2-small",
    "SAE_RELEASE": "gpt2-small-res-jb",
    "SAE_ID_TEMPLATE": "blocks.{layer}.hook_resid_pre",
    "HOOK_NAME_TEMPLATE": "blocks.{layer}.hook_resid_pre",
    "BATCH_SIZE": 16,

    # Produce TWO separate layerwise figures:
    #   (i)  U_SAE: fixed-pool SAE-mediated certificate
    #   (ii) U_SAE_Occam: exact-P uniform proxy-risk extension
    "PLOT_BOUNDS": ["U_SAE", "U_SAE_Occam"],

    "OUTPUT_DIR": "./gpt2_layerwise_exact_final",
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def exact_log_binom(m: int, p: int) -> float:
    """Return ln(C(m,p)) stably using log-gamma."""
    m = int(m)
    p = int(p)

    if p < 0 or p > m:
        raise ValueError(
            f"p must satisfy 0 <= p <= m, got p={p}, m={m}"
        )

    if p == 0 or p == m:
        return 0.0

    # C(m,p) = C(m,m-p); this is especially useful near saturation.
    q = min(p, m - p)

    return (
        math.lgamma(m + 1.0)
        - math.lgamma(q + 1.0)
        - math.lgamma(m - q + 1.0)
    )


def direct_hoeffding_bound(
    empirical_risk: float,
    range_width: float,
    delta: float,
    N: float,
):
    """
    Fixed-predictor Hoeffding upper bound.

    For the smoothed log loss, concentration uses capital Delta, the
    width of the bounded loss interval, not the absolute ceiling B.
    """
    margin = range_width * math.sqrt(
        math.log(1.0 / delta) / (2.0 * N)
    )

    return empirical_risk + margin, margin


def apply_topk_dense(activations, k):
    """
    Match the GPT-2 / sae_lens Top-k logic in the supplied final script:
    select the k largest activation VALUES, zero everything else.
    """
    k = min(k, activations.shape[-1])

    topk_vals, topk_inds = torch.topk(
        activations,
        k=k,
        dim=-1,
    )

    mask = torch.zeros_like(
        activations,
        dtype=torch.bool,
    )
    mask.scatter_(
        -1,
        topk_inds,
        True,
    )

    return activations * mask


def smoothed_bpd_loss(
    logits,
    tokens,
    alpha,
    vocab_size,
    reduction="mean",
):
    probs = torch.softmax(
        logits.float(),
        dim=-1,
    )

    probs_shifted = probs[:, :-1, :]
    tokens_shifted = tokens[:, 1:]

    true_probs = torch.gather(
        probs_shifted,
        -1,
        tokens_shifted.unsqueeze(-1),
    ).squeeze(-1)

    smoothed_probs = (
        (1.0 - alpha) * true_probs
        + alpha / vocab_size
    )

    log_probs = -torch.log2(
        smoothed_probs
    )

    loss_per_seq = log_probs.mean(
        dim=-1
    )

    if reduction == "mean":
        return loss_per_seq.mean()

    return loss_per_seq


def get_tokens_generator(
    model,
    batch_size,
    device,
    mode,
    calibration_limit=0,
):
    """
    Same disjoint calibration/evaluation partition used by the supplied
    delta/exact-Hoeffding script.

    A shuffled C4 stream with seed 42 is restarted for each layer.
    Calibration consumes the initial prefix. Evaluation skips exactly that
    prefix and begins with the first subsequent batch.
    """
    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
    )

    dataset = dataset.shuffle(
        seed=42,
        buffer_size=CONFIG["TOKENIZER_BUFFER_SIZE"],
    )

    iterator = iter(dataset)

    tokens_processed_global = 0
    skip_needed = (mode == "evaluation")

    while True:
        batch_texts = []

        try:
            for _ in range(batch_size):
                item = next(iterator)
                text = (
                    item["text"]
                    if "text" in item
                    else item["content"]
                )
                batch_texts.append(text)

        except StopIteration:
            if not batch_texts:
                break

        if not batch_texts:
            break

        tokens = model.to_tokens(
            batch_texts
        )

        if tokens.shape[1] < CONFIG["SEQ_LEN"]:
            continue

        tokens = tokens[:, : CONFIG["SEQ_LEN"]]
        num_tok = tokens.numel()

        if skip_needed:
            # Mirror the calibration split exactly.
            if tokens_processed_global < calibration_limit:
                tokens_processed_global += num_tok
                continue

            skip_needed = False

        elif mode == "calibration":
            if tokens_processed_global >= calibration_limit:
                break

            tokens_processed_global += num_tok

        yield tokens.to(device)


def extract_pre_acts(encode_out):
    """Normalize SAE encode outputs."""
    if torch.is_tensor(encode_out):
        return encode_out

    if hasattr(encode_out, "pre_acts"):
        return encode_out.pre_acts

    if isinstance(encode_out, dict) and "pre_acts" in encode_out:
        return encode_out["pre_acts"]

    if isinstance(encode_out, (list, tuple)) and len(encode_out) > 0:
        return encode_out[0]

    raise TypeError(
        f"Unsupported SAE encode output type: {type(encode_out)}"
    )


class SAEAdapter:
    def __init__(self, sae, device):
        self.sae = sae
        self.device = device

    def encode_pre_acts(self, flat_act):
        return extract_pre_acts(
            self.sae.encode(flat_act)
        )

    def decode_topk(
        self,
        feature_acts_raw,
        k,
    ):
        sparse_acts = apply_topk_dense(
            feature_acts_raw,
            k,
        )

        recons = self.sae.decode(
            sparse_acts
        )

        # Preserve the same support convention as the supplied final script.
        active_mask = sparse_acts > 0

        return recons, active_mask

    def get_latent_dim(self):
        return int(self.sae.cfg.d_sae)


def load_model():
    model = HookedTransformer.from_pretrained(
        CONFIG["MODEL_NAME"],
        device=CONFIG["DEVICE"],
        dtype=torch.float32,
    )

    model.eval()
    return model


def load_sae_for_layer(layer):
    sae_id = CONFIG["SAE_ID_TEMPLATE"].format(
        layer=layer
    )
    hook_name = CONFIG["HOOK_NAME_TEMPLATE"].format(
        layer=layer
    )

    sae, _, _ = SAE.from_pretrained(
        release=CONFIG["SAE_RELEASE"],
        sae_id=sae_id,
        device=CONFIG["DEVICE"],
    )

    sae.eval()

    adapter = SAEAdapter(
        sae=sae,
        device=CONFIG["DEVICE"],
    )

    return adapter, hook_name


def measure_pool_and_p(
    model,
    sae_adapter,
    token_gen,
    hook_name,
    target_tokens,
    k,
    device,
):
    print(
        f"  --> Calibration: measuring P on "
        f"{target_tokens:,} tokens with k={k}"
    )

    d_sae = sae_adapter.get_latent_dim()

    active_indices = torch.zeros(
        d_sae,
        dtype=torch.bool,
        device=device,
    )

    total_tokens = 0

    pbar = tqdm(
        total=target_tokens,
        desc="Calibration",
    )

    for tokens in token_gen:
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name],
            )

            act = cache[hook_name]
            flat_act = act.reshape(
                -1,
                act.shape[-1],
            )

            feature_acts_raw = (
                sae_adapter.encode_pre_acts(
                    flat_act
                )
            )

            _, active_mask = (
                sae_adapter.decode_topk(
                    feature_acts_raw,
                    k,
                )
            )

            batch_active = active_mask.any(
                dim=0
            )

            active_indices |= batch_active

        num_tok = tokens.numel()
        total_tokens += num_tok
        pbar.update(num_tok)

        del (
            tokens,
            cache,
            act,
            flat_act,
            feature_acts_raw,
            active_mask,
            batch_active,
        )

        if total_tokens >= target_tokens:
            break

    pbar.close()

    P = int(
        active_indices.sum().item()
    )

    print(
        f"  <-- Calibration complete: "
        f"m={d_sae:,}, P={P:,}, P/m={P/d_sae:.6f}"
    )

    return active_indices, P, d_sae


# ============================================================
# 3. CERTIFICATE CALCULATION
# ============================================================

def compute_certificates(
    R_hat_M,
    R_hat_hG,
    eps_loss_hat,
    eta_hat,
    N,
    vocab_size,
    m,
    P,
):
    alpha = CONFIG["ALPHA"]
    delta = CONFIG["CONFIDENCE_DELTA"]

    # Absolute upper endpoint of smoothed loss.
    B = math.log2(
        vocab_size / alpha
    )

    # Capital Delta: WIDTH of the smoothed-loss range.
    Delta = math.log2(
        1.0
        + (
            (1.0 - alpha)
            * vocab_size
            / alpha
        )
    )

    # Correct uninformed baseline.
    R_unif = math.log2(
        vocab_size
    )

    # Exact class log-size in nats.
    log_C = exact_log_binom(
        m,
        P,
    )

    # Empirical transfer terms.
    mismatch_empirical = (
        Delta * eta_hat
    )

    # Direct fixed-predictor Hoeffding diagnostics.
    direct_U_M, direct_margin = (
        direct_hoeffding_bound(
            R_hat_M,
            Delta,
            delta,
            N,
        )
    )

    direct_U_hG, _ = (
        direct_hoeffding_bound(
            R_hat_hG,
            Delta,
            delta,
            N,
        )
    )

    # --------------------------------------------------------
    # Theorem 1: fixed-pool SAE-mediated certificate
    # --------------------------------------------------------
    fixed_proxy_conc = (
        Delta
        * math.sqrt(
            math.log(3.0 / delta)
            / (2.0 * N)
        )
    )

    eta_conc = (
        Delta
        * math.sqrt(
            math.log(3.0 / delta)
            / (2.0 * N)
        )
    )

    eps_conc = (
        Delta
        * math.sqrt(
            math.log(6.0 / delta)
            / (2.0 * N)
        )
    )

    U_SAE = (
        R_hat_hG
        + eps_loss_hat
        + mismatch_empirical
        + fixed_proxy_conc
        + eta_conc
        + eps_conc
    )

    # --------------------------------------------------------
    # Corollary 1: exact-P uniform proxy-risk extension
    # --------------------------------------------------------
    exact_P_proxy_conc = (
        Delta
        * math.sqrt(
            (
                log_C
                + math.log(3.0 / delta)
            )
            / (2.0 * N)
        )
    )

    # IMPORTANT:
    # exact_P_proxy_conc REPLACES fixed_proxy_conc.
    U_SAE_Occam = (
        R_hat_hG
        + eps_loss_hat
        + mismatch_empirical
        + eta_conc
        + exact_P_proxy_conc
        + eps_conc
    )

    return {
        "B": B,
        "Delta": Delta,
        "Random Baseline": R_unif,
        "Exact_log_C_m_P": log_C,
        "P_over_m": P / m,

        "Mismatch_Delta_eta": mismatch_empirical,

        "Fixed_Proxy_Concentration": fixed_proxy_conc,
        "Eta_Concentration": eta_conc,
        "Reconstruction_Concentration": eps_conc,
        "Exact_P_Proxy_Concentration": exact_P_proxy_conc,

        "Direct_U_M": direct_U_M,
        "Direct_U_hG": direct_U_hG,
        "Hoeffding_Margin": direct_margin,

        "U_SAE": U_SAE,
        "U_SAE_Occam": U_SAE_Occam,
        "Occam_Increment": U_SAE_Occam - U_SAE,
    }


# ============================================================
# 4. RUN ONE GPT-2 LAYER
# ============================================================

def run_experiment_for_layer(
    model,
    layer,
):
    print(
        "\n"
        + "=" * 80
        + f"\nGPT-2 SMALL | LAYER {layer}\n"
        + "=" * 80
    )

    torch.cuda.empty_cache()
    gc.collect()

    try:
        sae_adapter, hook_name = (
            load_sae_for_layer(
                layer
            )
        )

    except Exception as exc:
        print(
            f"Error loading GPT-2 SAE for "
            f"layer {layer}: {exc}"
        )
        return []

    vocab_size = int(
        model.cfg.d_vocab
    )

    top_k = CONFIG["TOP_K"]
    cal_tokens_limit = (
        CONFIG["CALIBRATION_TOKENS"]
    )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------
    cal_gen = get_tokens_generator(
        model,
        CONFIG["BATCH_SIZE"],
        CONFIG["DEVICE"],
        mode="calibration",
        calibration_limit=cal_tokens_limit,
    )

    pool_mask, P, m = measure_pool_and_p(
        model=model,
        sae_adapter=sae_adapter,
        token_gen=cal_gen,
        hook_name=hook_name,
        target_tokens=cal_tokens_limit,
        k=top_k,
        device=CONFIG["DEVICE"],
    )

    # --------------------------------------------------------
    # Evaluation on disjoint C4 stream
    # --------------------------------------------------------
    print(
        "  --> Evaluation: measuring revised certificates"
    )

    eval_gen = get_tokens_generator(
        model,
        CONFIG["BATCH_SIZE"],
        CONFIG["DEVICE"],
        mode="evaluation",
        calibration_limit=cal_tokens_limit,
    )

    results = {
        "loss_M": [],
        "loss_h_G": [],
        "epsilon_loss": [],
        "pool_violation": [],
    }

    plot_points = []

    total_sequences = 0
    current_checkpoint_idx = 0

    max_N = max(
        CONFIG["N_CHECKPOINTS"]
    )

    pbar = tqdm(
        total=max_N,
        desc=f"Layer {layer} evaluation sequences",
    )

    for tokens in eval_gen:
        with torch.no_grad():
            # ------------------------------------------------
            # A. Base model
            # ------------------------------------------------
            orig_logits, cache = (
                model.run_with_cache(
                    tokens,
                    names_filter=[hook_name],
                )
            )

            loss_M = smoothed_bpd_loss(
                orig_logits,
                tokens,
                CONFIG["ALPHA"],
                vocab_size,
                reduction="none",
            )

            act = cache[hook_name]

            flat_act = act.reshape(
                -1,
                act.shape[-1],
            )

            feature_acts_raw = (
                sae_adapter.encode_pre_acts(
                    flat_act
                )
            )

            # ------------------------------------------------
            # B. Unrestricted SAE proxy S o M
            # ------------------------------------------------
            (
                recons_unrestricted,
                active_mask_unrestricted,
            ) = sae_adapter.decode_topk(
                feature_acts_raw,
                top_k,
            )

            recons_unrestricted = (
                recons_unrestricted
                .reshape(act.shape)
                .to(act.dtype)
            )

            # ------------------------------------------------
            # C. Pool-restricted SAE proxy h_G*
            # ------------------------------------------------
            masked_acts = (
                feature_acts_raw
                * pool_mask.unsqueeze(0)
            )

            recons_restricted, _ = (
                sae_adapter.decode_topk(
                    masked_acts,
                    top_k,
                )
            )

            recons_restricted = (
                recons_restricted
                .reshape(act.shape)
                .to(act.dtype)
            )

            # ------------------------------------------------
            # D. Support mismatch eta
            # ------------------------------------------------
            violation_mask = (
                active_mask_unrestricted
                & (~pool_mask.unsqueeze(0))
            )

            seq_has_violation = (
                violation_mask
                .any(dim=-1)
                .reshape(
                    tokens.shape[0],
                    -1,
                )
                .any(dim=-1)
                .float()
            )

            # ------------------------------------------------
            # E. Run unrestricted proxy
            # ------------------------------------------------
            def hook_unrestricted(
                activations,
                hook,
            ):
                return recons_unrestricted

            logits_Som = (
                model.run_with_hooks(
                    tokens,
                    fwd_hooks=[
                        (
                            hook_name,
                            hook_unrestricted,
                        )
                    ],
                )
            )

            loss_Som = smoothed_bpd_loss(
                logits_Som,
                tokens,
                CONFIG["ALPHA"],
                vocab_size,
                reduction="none",
            )

            # ------------------------------------------------
            # F. Run pool-restricted proxy
            # ------------------------------------------------
            def hook_restricted(
                activations,
                hook,
            ):
                return recons_restricted

            logits_hG = (
                model.run_with_hooks(
                    tokens,
                    fwd_hooks=[
                        (
                            hook_name,
                            hook_restricted,
                        )
                    ],
                )
            )

            loss_hG = smoothed_bpd_loss(
                logits_hG,
                tokens,
                CONFIG["ALPHA"],
                vocab_size,
                reduction="none",
            )

            gap = torch.abs(
                loss_M - loss_Som
            )

            results["loss_M"].extend(
                loss_M.detach()
                .cpu()
                .float()
                .numpy()
            )

            results["loss_h_G"].extend(
                loss_hG.detach()
                .cpu()
                .float()
                .numpy()
            )

            results["epsilon_loss"].extend(
                gap.detach()
                .cpu()
                .float()
                .numpy()
            )

            results["pool_violation"].extend(
                seq_has_violation.detach()
                .cpu()
                .float()
                .numpy()
            )

        batch_sequences = int(
            tokens.shape[0]
        )

        total_sequences += batch_sequences
        pbar.update(batch_sequences)

        # Evaluate all newly reached predetermined checkpoints.
        while (
            current_checkpoint_idx
            < len(CONFIG["N_CHECKPOINTS"])
            and total_sequences
            >= CONFIG["N_CHECKPOINTS"][
                current_checkpoint_idx
            ]
        ):
            requested_N = (
                CONFIG["N_CHECKPOINTS"][
                    current_checkpoint_idx
                ]
            )

            # Because batch size is 16, actual N can exceed the requested
            # checkpoint by at most 15. We report the actual number.
            N = len(
                results["loss_h_G"]
            )

            R_hat_M = float(
                np.mean(
                    results["loss_M"]
                )
            )

            R_hat_hG = float(
                np.mean(
                    results["loss_h_G"]
                )
            )

            eps_loss_hat = float(
                np.mean(
                    results["epsilon_loss"]
                )
            )

            eta_hat = float(
                np.mean(
                    results["pool_violation"]
                )
            )

            cert = compute_certificates(
                R_hat_M=R_hat_M,
                R_hat_hG=R_hat_hG,
                eps_loss_hat=eps_loss_hat,
                eta_hat=eta_hat,
                N=N,
                vocab_size=vocab_size,
                m=m,
                P=P,
            )

            point = {
                "Model": "GPT-2 Small",
                "Layer": layer,

                "Requested_N": requested_N,
                "N": N,

                "m": m,
                "P": P,

                "R_hat_M": R_hat_M,
                "R_hat_hG": R_hat_hG,
                "eps_loss_hat": eps_loss_hat,
                "eta_hat": eta_hat,

                **cert,
            }

            plot_points.append(
                point
            )

            print(
                f"  Layer {layer:2d} | "
                f"N={N:7,d} | "
                f"P={P:,} | "
                f"P/m={P/m:.6f} | "
                f"lnC={cert['Exact_log_C_m_P']:.3f} | "
                f"Rhat(M)={R_hat_M:.3f} | "
                f"Rhat(hG)={R_hat_hG:.3f} | "
                f"eps={eps_loss_hat:.3f} | "
                f"eta={eta_hat:.5f} | "
                f"U_SAE={cert['U_SAE']:.3f} | "
                f"U_SAE+Occam={cert['U_SAE_Occam']:.3f} | "
                f"R_unif={cert['Random Baseline']:.3f}"
            )

            current_checkpoint_idx += 1

        del (
            tokens,
            orig_logits,
            cache,
            loss_M,
            act,
            flat_act,
            feature_acts_raw,
            recons_unrestricted,
            active_mask_unrestricted,
            masked_acts,
            recons_restricted,
            violation_mask,
            seq_has_violation,
            logits_Som,
            loss_Som,
            logits_hG,
            loss_hG,
            gap,
        )

        if (
            current_checkpoint_idx
            >= len(CONFIG["N_CHECKPOINTS"])
        ):
            break

    pbar.close()

    # --------------------------------------------------------
    # Descriptive first observed crossings for BOTH certificates.
    # These are summaries over predetermined nested prefixes only;
    # no simultaneous-confidence stopping-time interpretation is implied.
    # --------------------------------------------------------
    for metric in CONFIG["PLOT_BOUNDS"]:
        first_cross = None

        for point in plot_points:
            if point[metric] < point["Random Baseline"]:
                first_cross = point["N"]
                break

        field_name = (
            "First_Observed_Nonvacuous_U_SAE"
            if metric == "U_SAE"
            else "First_Observed_Nonvacuous_U_SAE_Occam"
        )

        for point in plot_points:
            point[field_name] = first_cross
            point["Crossing_Interpretation"] = (
                "descriptive over predetermined nested prefixes"
            )

        if first_cross is None:
            print(
                f"Layer {layer}: no observed non-vacuous "
                f"{metric} checkpoint through N={max_N:,}."
            )
        else:
            print(
                f"Layer {layer}: first observed "
                f"{metric} < R_unif at N={first_cross:,} "
                "(descriptive)."
            )

    del sae_adapter, pool_mask

    gc.collect()
    torch.cuda.empty_cache()

    return plot_points


# ============================================================
# 5. RUNNER
# ============================================================

def run_all_layers():
    os.makedirs(
        CONFIG["OUTPUT_DIR"],
        exist_ok=True,
    )

    print(
        f"Top-k = {CONFIG['TOP_K']}\n"
        f"Layers = {CONFIG['LAYERS']}\n"
        f"Calibration tokens = "
        f"{CONFIG['CALIBRATION_TOKENS']:,}\n"
        f"Evaluation checkpoints = "
        f"{CONFIG['N_CHECKPOINTS']}\n"
        f"Plotted bounds = "
        f"{CONFIG['PLOT_BOUNDS']}"
    )

    model = load_model()

    all_data = []

    for layer in CONFIG["LAYERS"]:
        all_data.extend(
            run_experiment_for_layer(
                model,
                layer,
            )
        )

    df = pd.DataFrame(
        all_data
    )

    if df.empty:
        raise RuntimeError(
            "No GPT-2 layer completed successfully."
        )

    csv_path = os.path.join(
        CONFIG["OUTPUT_DIR"],
        "gpt2_layerwise_delta_exact_hoeffding_results.csv",
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    print(
        f"Saved results: {csv_path}"
    )

    return model, df


# ============================================================
# 6. PLOTTING
# ============================================================

def plot_layerwise_metric(df, metric):
    """
    Plot one certificate across GPT-2 layers.

    Two separate figures are generated by main():
        - U_SAE
        - U_SAE_Occam
    """
    if metric not in {"U_SAE", "U_SAE_Occam"}:
        raise ValueError(
            "metric must be U_SAE or U_SAE_Occam"
        )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(
        figsize=(10.5, 6.2)
    )

    for layer in CONFIG["LAYERS"]:
        group = (
            df[df["Layer"] == layer]
            .sort_values("N")
            .reset_index(drop=True)
        )

        if group.empty:
            continue

        x = group["N"].to_numpy()
        y = group[metric].to_numpy()

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            label=f"Layer {layer}",
        )

    baseline = float(
        df["Random Baseline"].iloc[0]
    )

    ax.axhline(
        y=baseline,
        color="black",
        linestyle=":",
        linewidth=2.0,
        label=(
            f"Uniform baseline = "
            f"{baseline:.2f} bits"
        ),
    )

    # Match the visual spirit of the old Figure 5.
    ax.set_xscale(
        "log",
        base=2,
    )

    xticks = [
        16_000,
        32_000,
        64_000,
        128_000,
    ]

    ax.set_xticks(xticks)
    ax.set_xticklabels(
        ["16k", "32k", "64k", "128k"]
    )

    if metric == "U_SAE":
        y_label = r"$U_{\mathrm{SAE}}$ (bits)"
        title = (
            "GPT-2 Small: Layerwise Fixed-Pool "
            "SAE-Mediated Certificate"
        )
        filename_tag = "u_sae"

    else:
        y_label = r"$U_{\mathrm{SAE+Occam}}$ (bits)"
        title = (
            "GPT-2 Small: Layerwise SAE-Mediated "
            "Certificate (Exact-$P$ Extension)"
        )
        filename_tag = "u_sae_occam"

    ax.set_xlabel(
        r"Held-out sample size $N$"
    )
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.grid(
        True,
        which="both",
        linestyle="-",
        alpha=0.18,
    )

    ax.legend(
        ncol=3,
        frameon=True,
    )

    plt.tight_layout()

    png_path = os.path.join(
        CONFIG["OUTPUT_DIR"],
        f"gpt2_bounds_across_layers_{filename_tag}.png",
    )

    pdf_path = os.path.join(
        CONFIG["OUTPUT_DIR"],
        f"gpt2_bounds_across_layers_{filename_tag}.pdf",
    )

    plt.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


def save_crossing_summary(df):
    """
    Save descriptive first-observed non-vacuity checkpoints for BOTH
    U_SAE and U_SAE_Occam.
    """
    rows = []

    for metric in CONFIG["PLOT_BOUNDS"]:
        for layer in CONFIG["LAYERS"]:
            group = (
                df[df["Layer"] == layer]
                .sort_values("N")
            )

            if group.empty:
                continue

            baseline = float(
                group["Random Baseline"].iloc[0]
            )

            hits = group[
                group[metric] < baseline
            ]

            if hits.empty:
                crossing_N = np.nan
                bound_at_cross = np.nan
            else:
                first = hits.iloc[0]
                crossing_N = first["N"]
                bound_at_cross = first[metric]

            rows.append(
                {
                    "Layer": layer,
                    "Metric": metric,
                    "First_observed_nonvacuous_N": crossing_N,
                    "Bound_at_crossing": bound_at_cross,
                    "R_unif": baseline,
                    "Interpretation": (
                        "descriptive over predetermined nested prefixes"
                    ),
                }
            )

    summary = pd.DataFrame(rows)

    path = os.path.join(
        CONFIG["OUTPUT_DIR"],
        "gpt2_layerwise_crossing_summary_both_bounds.csv",
    )

    summary.to_csv(
        path,
        index=False,
    )

    print(f"Saved crossing summary: {path}")
    print("\nDescriptive crossing summary:\n")
    print(summary.to_string(index=False))


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":
    model, df = run_all_layers()

    for metric in CONFIG["PLOT_BOUNDS"]:
        plot_layerwise_metric(
            df,
            metric,
        )

    save_crossing_summary(
        df
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
