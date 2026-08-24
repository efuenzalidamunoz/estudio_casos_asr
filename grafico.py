import csv
import numpy as np
import matplotlib.pyplot as plt

INPUT_CSV = "scripts_out/R2/c/datos_precision.csv"
OUTPUT_SVG = "scripts_out/R2/c/wer_porcentual_ordenado.svg"

datos = []
with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        datos.append({
            "formato": row["formato"],
            "bits": float(row["bits_efectivos"]),
            "wer": float(row["wer_promedio"])
        })

datos.sort(key=lambda x: x["wer"])

formatos = [d["formato"] for d in datos]
bits = np.array([d["bits"] for d in datos])
wer = np.array([d["wer"] for d in datos])

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})

x = np.arange(len(formatos))

fig, ax = plt.subplots(
    figsize=(6.6, 3.8)
)

ax.plot(
    x,
    wer,
    marker="o",
    linewidth=1.8,
    markersize=6
)

ax.set_title(
    "Error de transcripción según representación numérica"
)

ax.set_ylabel(
    "WER promedio (%)"
)

ax.set_xlabel(
    "Formato y bits efectivos por peso"
)

ax.set_xticks(x)

ax.set_xticklabels([
    f"{formato}\n{bit:g} bits/peso"
    for formato, bit in zip(formatos, bits)
])

# Margen automático para que las etiquetas no choquen
margen = max((wer.max() - wer.min()) * 0.18, 0.5)

ax.set_ylim(
    wer.min() - margen,
    wer.max() + margen
)

ax.grid(
    axis="y",
    alpha=0.25
)


for xi, yi in zip(x, wer):

    ax.annotate(
        f"{yi:.2f}%",
        xy=(xi, yi),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(
    axis="x",
    length=0
)

fig.tight_layout(
    pad=1.4
)

fig.savefig(
    OUTPUT_SVG,
    bbox_inches="tight"
)

plt.show()