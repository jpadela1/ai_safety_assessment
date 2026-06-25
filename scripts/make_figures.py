"""
make_figures.py — regenerate the result figures (Fig. 2-4) from results/*.csv.

    python scripts/make_figures.py

Writes figures/fig2_text_calibration.png, figures/fig3_text_410m.{png,pdf},
figures/fig4_tabular_dose.png. Fig. 1 (framework flow) is a hand-built diagram
shipped directly in figures/ and is not regenerated here.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)


def boot_ci(x, y, B=10000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x); y = np.asarray(y); n = len(x); rs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if np.std(x[i]) > 0 and np.std(y[i]) > 0:
            rs.append(np.corrcoef(x[i], y[i])[0, 1])
    return np.corrcoef(x, y)[0, 1], np.percentile(rs, [2.5, 97.5])


def text_arm():
    cal = pd.read_csv(RES / "civilcomments_doseresponse.csv")
    o410 = pd.read_csv(RES / "civilcomments_outcome_410m.csv").drop_duplicates(["injected_p", "seed"], keep="last")
    o160 = pd.read_csv(RES / "civilcomments_outcome_160m.csv").drop_duplicates(["injected_p", "seed"], keep="last")
    m410, m160 = cal.merge(o410, on=["injected_p", "seed"]), cal.merge(o160, on=["injected_p", "seed"])

    # Fig 2 — calibration
    g = cal.groupby("injected_p")["harm_content_density"].agg(["mean", "std"])
    r = np.corrcoef(cal["injected_p"], cal["harm_content_density"])[0, 1]
    plt.figure(figsize=(5.2, 4))
    plt.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=3, color="#1f77b4")
    plt.xlabel("injected toxic proportion"); plt.ylabel("harm_content_density (proxy)")
    plt.title(f"Text-arm proxy calibration: r = {r:.4f}")
    plt.tight_layout(); plt.savefig(FIG / "fig2_text_calibration.png", dpi=200); plt.close()

    # Fig 3 — 410M predictive (left) + scale comparison (right)
    r924, ci = boot_ci(m410["harm_content_density"], m410["generated_toxicity"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    x, y = m410["harm_content_density"], m410["generated_toxicity"]
    ax[0].scatter(x, y, c="#244", s=34, alpha=.85)
    b, a = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50); ax[0].plot(xs, b * xs + a, "--", c="#c33")
    ax[0].set_xlabel("harm-content density (pre-training proxy)"); ax[0].set_ylabel("generated toxicity (EMT)")
    ax[0].set_title(f"H2 (Pythia-410M): r = {r924:.3f}, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
    for d, lab, c in [(m160, "160M (underpowered)", "#bbb"), (m410, "410M (confirmed)", "#244")]:
        gg = d.groupby("injected_p")["generated_toxicity"].agg(["mean", "std"])
        ax[1].errorbar(gg.index, gg["mean"], yerr=gg["std"], marker="o", capsize=3, label=lab, color=c)
    ax[1].set_xlabel("injected toxic proportion"); ax[1].set_ylabel("generated toxicity (EMT)")
    ax[1].set_title("H4 — effect emerges with model scale"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "fig3_text_410m.png", dpi=200); plt.savefig(FIG / "fig3_text_410m.pdf"); plt.close()


def tabular_arm():
    df = pd.read_csv(RES / "diabetes_outcome.csv")
    g = df.groupby("injected_p").agg(proxy=("proxy_noise", "mean"), proxy_sd=("proxy_noise", "std"),
                                     auc=("auc", "mean"), auc_sd=("auc", "std"),
                                     ece=("ece", "mean"), ece_sd=("ece", "std"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].errorbar(g.index, g["proxy"], yerr=g["proxy_sd"], marker="o", capsize=3, color="#244", label="integrity proxy")
    ax[0].plot([0, .4], [0, .4], "--", c="#aaa", lw=1, label="identity")
    ax[0].set_xlabel("injected label-noise fraction"); ax[0].set_ylabel("estimated noise (proxy)")
    ax[0].set_title("Tabular calibration: r = +0.98"); ax[0].legend(fontsize=8)
    ax2, ax3 = ax[1], ax[1].twinx()
    l1 = ax2.errorbar(g.index, g["auc"], yerr=g["auc_sd"], marker="o", capsize=3, color="#244", label="AUC (down)")
    l2 = ax3.errorbar(g.index, g["ece"], yerr=g["ece_sd"], marker="s", capsize=3, color="#c33", label="ECE (up)")
    ax2.set_xlabel("injected label-noise fraction"); ax2.set_ylabel("downstream AUC", color="#244")
    ax3.set_ylabel("calibration error (ECE)", color="#c33"); ax2.set_title("Tabular predictive validity (Diabetes 130)")
    ax2.legend(handles=[l1, l2], fontsize=8, loc="center left")
    plt.tight_layout(); plt.savefig(FIG / "fig4_tabular_dose.png", dpi=200); plt.close()


if __name__ == "__main__":
    text_arm(); tabular_arm()
    print("wrote figures to", FIG)
