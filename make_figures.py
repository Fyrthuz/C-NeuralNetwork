#!/usr/bin/env python3
"""
make_figures.py — Genera las figuras del benchmark (comparativa.csv) para el README.

Uso:
    uv run --with matplotlib python3 make_figures.py
    # o, si tienes matplotlib instalado:
    python3 make_figures.py

Entrada:  comparativa.csv  (lo genera ./benchmark.sh)
Salida:   docs/figures/*.png

Las figuras se pueden regenerar en cualquier momento; si repites el benchmark,
vuelve a ejecutar este script para mantener las imágenes sincronizadas con los datos.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "comparativa.csv"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "docs/figures"

CONFIGS = [("cuda", "f32"), ("cuda", "f64"), ("cpu", "f32"), ("cpu", "f64")]
COLORS = {
    ("cuda", "f32"): "#1f77b4",   # azul      = GPU
    ("cuda", "f64"): "#a8cbe8",
    ("cpu", "f32"): "#d62728",    # rojo      = CPU
    ("cpu", "f64"): "#f0b0b0",
}
LABELS = {
    ("cuda", "f32"): "GPU · F32",
    ("cuda", "f64"): "GPU · F64",
    ("cpu", "f32"): "CPU · F32",
    ("cpu", "f64"): "CPU · F64",
}


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            sizes = [int(s) for s in r["arquitectura"].split(",")]
            rows.append({
                "arch": r["arquitectura"],
                "sizes": sizes,
                "depth": len(sizes) - 2,          # numero de capas ocultas
                "device": r["device"],
                "dtype": r["dtype"],
                "tiempo": float(r["tiempo_s"]),
                "s_epoca": float(r["s_epoca"]),
                "acc_train": float(r["acc_train"]),
                "acc_test": float(r["acc_test"]),
            })
    rows.sort(key=lambda r: r["depth"])
    return rows


def get(rows, arch, device, dtype):
    for r in rows:
        if r["arch"] == arch and r["device"] == device and r["dtype"] == dtype:
            return r
    raise KeyError((arch, device, dtype))


def arch_ticks(rows, archs):
    labels = []
    for a in archs:
        r = get(rows, a, "cuda", "f32")
        arrow = "→".join(str(s) for s in r["sizes"])
        n = r["depth"]
        capa = "oculta" if n == 1 else "ocultas"
        labels.append(f"{arrow}\n({n} {capa})")
    return labels


def style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    })


def fig_tiempo_total(rows, archs, out):
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(archs))
    w = 0.2
    for i, cfg in enumerate(CONFIGS):
        vals = [get(rows, a, *cfg)["tiempo"] for a in archs]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, color=COLORS[cfg],
                      edgecolor="black", linewidth=0.4, label=LABELS[cfg])
        ax.bar_label(bars, fmt="%.1f", fontsize=7.5, padding=1.5)
    ax.set_yscale("log")
    ax.set_ylim(2, 900)
    ax.set_ylabel("Tiempo de entrenamiento (s, escala log)")
    ax.set_title("Tiempo total de entrenamiento por arquitectura\n"
                 "(15 épocas · 60 000 muestras · batch 64)")
    ax.set_xticks(x)
    ax.set_xticklabels(arch_ticks(rows, archs))
    ax.legend(ncols=4, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_aceleracion(rows, archs, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(archs))
    w = 0.35
    for i, dt in enumerate(("f32", "f64")):
        vals = [get(rows, a, "cpu", dt)["tiempo"] / get(rows, a, "cuda", dt)["tiempo"]
                for a in archs]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=COLORS[("cuda", dt)],
                      edgecolor="black", linewidth=0.4, label=f"F{dt[1:]}")
        ax.bar_label(bars, fmt="%.1f×", fontsize=9, padding=2)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.text(3.42, 1.4, "1× = igual rendimiento", fontsize=8, color="gray", ha="right")
    ax.set_ylim(0, 24)
    ax.set_ylabel("Aceleración  (tiempo CPU / tiempo GPU)")
    ax.set_title("Factor de aceleración de la GPU frente a la CPU")
    ax.set_xticks(x)
    ax.set_xticklabels(arch_ticks(rows, archs))
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_precision(rows, archs, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharex=True)
    x = np.arange(len(archs))
    w = 0.2
    for ax, key, name in ((axes[0], "acc_train", "Precisión en entrenamiento"),
                          (axes[1], "acc_test", "Precisión en test")):
        for i, cfg in enumerate(CONFIGS):
            vals = [get(rows, a, *cfg)[key] for a in archs]
            bars = ax.bar(x + (i - 1.5) * w, vals, w, color=COLORS[cfg],
                          edgecolor="black", linewidth=0.4, label=LABELS[cfg])
            ax.bar_label(bars, fmt="%.1f", fontsize=6.5, padding=1.5)
        ax.set_ylim(88, 103)
        ax.set_title(name)
        ax.set_xticks(x)
        ax.set_xticklabels(arch_ticks(rows, archs), fontsize=8.5)
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].legend(fontsize=8, loc="lower right", ncols=2)
    fig.suptitle("Precisión por arquitectura (mismos hiperparámetros)", y=0.99)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_escalado(rows, archs, out):
    # desplazamientos por configuracion para que las anotaciones no se pisen
    offsets = {("cuda", "f32"): (8, 8), ("cuda", "f64"): (8, -16),
               ("cpu", "f32"): (8, 8), ("cpu", "f64"): (8, -16)}
    # versiones oscuras para el texto (las claras apenas contrastan en blanco)
    annot_colors = {("cuda", "f32"): "#1f77b4", ("cuda", "f64"): "#2e6da4",
                    ("cpu", "f32"): "#d62728", ("cpu", "f64"): "#b02020"}
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for cfg in CONFIGS:
        xs = [get(rows, a, *cfg)["depth"] for a in archs]
        ys = [get(rows, a, *cfg)["s_epoca"] for a in archs]
        ax.plot(xs, ys, marker="o", linewidth=2, color=COLORS[cfg], label=LABELS[cfg])
        ax.annotate(f"{ys[-1]:.2f} s", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=offsets[cfg], fontsize=8, color=annot_colors[cfg],
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.6))
    ax.set_yscale("log")
    ax.set_xlim(-0.15, 3.55)
    ax.set_xlabel("Número de capas ocultas")
    ax.set_ylabel("Tiempo por época (s, escala log)")
    ax.set_title("Escalado del tiempo por época con la profundidad")
    ax.set_xticks([0, 1, 2, 3])
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    style()
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_rows(CSV_PATH)

    archs = []
    for r in rows:
        if r["arch"] not in archs:
            archs.append(r["arch"])          # rows ya vienen ordenadas por profundidad

    for name, fn in (("tiempo_total.png", fig_tiempo_total),
                     ("aceleracion_gpu_cpu.png", fig_aceleracion),
                     ("precision.png", fig_precision),
                     ("escalado_profundidad.png", fig_escalado)):
        path = os.path.join(OUT_DIR, name)
        fn(rows, archs, path)
        print(f"[OK] {path}")

    print("\nFactores de aceleracion (CPU / GPU):")
    for a in archs:
        for dt in ("f32", "f64"):
            sp = get(rows, a, "cpu", dt)["tiempo"] / get(rows, a, "cuda", dt)["tiempo"]
            print(f"  {a:>18}  {dt}: {sp:5.1f}x")


if __name__ == "__main__":
    main()
