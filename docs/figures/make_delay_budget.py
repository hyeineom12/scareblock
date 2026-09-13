"""논문 그림 1 — 지연 예산 타임라인.

    python3 docs/figures/make_delay_budget.py

무엇을 보이는가: 탐지는 **라이브 신호**에서 일어나고 개입은 **사용자 화면**에서
일어난다. 두 시각의 간격이 개입에 쓸 수 있는 예산이고, 지연 L이 없으면 음수가 된다.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK, GRAY, LATE, BUDGET = "#1a1a1a", "#8a8a8a", "#b3261e", "#1b5e20"

X0, X1 = 0.6, 10.4
ONSET, FIRE, EXPOSE = 2.6, 4.3, 9.2      # 25 ms를 보이게 하려고 벌려 그렸다
Y_LIVE, Y_SCREEN = 0.88, 0.10
Y_D, Y_L, Y_BUDGET = 1.18, 0.62, 0.30


def line(ax, y, dashed, name, sub):
    ax.plot([X0, X1], [y, y], color=INK, lw=2.2,
            ls=(0, (7, 5)) if dashed else "-", solid_capstyle="butt", zorder=2)
    ax.text(X0 - 0.3, y + 0.07, name, ha="right", va="center",
            fontsize=13, color=INK, fontweight="bold")
    ax.text(X0 - 0.3, y - 0.08, sub, ha="right", va="center", fontsize=10, color=GRAY)


def tick(ax, x, y, bold):
    ax.plot([x, x], [y - 0.11, y + 0.11], color=INK,
            lw=4.0 if bold else 1.5, ls="-" if bold else (0, (2, 2)), zorder=4)


def drop(ax, x, y_from, y_to):
    ax.plot([x, x], [y_to, y_from], color=GRAY, lw=0.9, ls=(0, (1, 3)), zorder=1)


def arrow(ax, xa, xb, y, color, both=True):
    ax.add_patch(FancyArrowPatch((xa, y), (xb, y),
                                 arrowstyle="<|-|>" if both else "-|>",
                                 mutation_scale=14, lw=2.0, color=color, zorder=3))


def panel(ax, title, l_zero):
    ax.set_xlim(-2.6, X1 + 0.4)
    ax.set_ylim(-0.30, 1.42)
    ax.axis("off")
    ax.text(-2.6, 1.40, title, fontsize=14.5, fontweight="bold", color=INK, va="top")

    line(ax, Y_LIVE, False, "라이브", "탐지기가 듣는 소리")
    line(ax, Y_SCREEN, True, "사용자 화면", "라이브보다 L초 늦다")

    # 사건 onset — 굵은 세로선
    tick(ax, ONSET, Y_LIVE, True)
    ax.text(ONSET - 0.15, Y_LIVE + 0.13, "사건 onset", ha="right", va="bottom",
            fontsize=12, color=INK, fontweight="bold")

    # 탐지 확신 — 얇은 세로 점선
    tick(ax, FIRE, Y_LIVE, False)
    ax.text(FIRE + 0.12, Y_LIVE - 0.14, "탐지 확신", ha="left", va="top",
            fontsize=11, color=INK)

    arrow(ax, ONSET, FIRE, Y_D, GRAY)
    ax.text(FIRE + 0.2, Y_D, "탐지 지연 d ≈ 25 ms", ha="left", va="center",
            fontsize=12, color=GRAY, fontweight="bold")

    drop(ax, ONSET, Y_LIVE - 0.11, Y_SCREEN + 0.11)
    drop(ax, FIRE, Y_LIVE - 0.11, Y_BUDGET - 0.12 if not l_zero else 0.30)

    if l_zero:
        tick(ax, ONSET, Y_SCREEN, True)
        ax.text(ONSET - 0.15, Y_SCREEN - 0.14, "사용자가 본다", ha="right", va="top",
                fontsize=12, color=INK, fontweight="bold")
        arrow(ax, FIRE, ONSET, Y_BUDGET + 0.06, LATE, both=False)
        ax.text(FIRE + 0.2, Y_BUDGET + 0.10, "예산 = -25 ms", ha="left", va="bottom",
                fontsize=13, color=LATE, fontweight="bold")
        ax.text(FIRE + 0.2, Y_BUDGET + 0.02, "확신했을 때 사용자는 이미 봤다",
                ha="left", va="top", fontsize=10.5, color=LATE)
    else:
        tick(ax, EXPOSE, Y_SCREEN, True)
        ax.text(EXPOSE + 0.15, Y_SCREEN - 0.14, "사용자가 본다", ha="center", va="top",
                fontsize=12, color=INK, fontweight="bold")
        drop(ax, EXPOSE, Y_LIVE - 0.11, Y_SCREEN + 0.11)

        arrow(ax, ONSET, EXPOSE, Y_L, GRAY)
        ax.text((ONSET + EXPOSE) / 2, Y_L - 0.05, "L = 3.0 s", ha="center",
                va="top", fontsize=12, color=GRAY, fontweight="bold")

        arrow(ax, FIRE, EXPOSE, Y_BUDGET, BUDGET)
        ax.text((FIRE + EXPOSE) / 2, Y_BUDGET + 0.05,
                "개입 준비 예산 = L - d ≈ 2.98 s", ha="center", va="bottom",
                fontsize=13, color=BUDGET, fontweight="bold")
        ax.text((FIRE + EXPOSE) / 2, Y_BUDGET - 0.07,
                "이 구간 안에 블러를 준비하면 사용자는 못 본다",
                ha="center", va="top", fontsize=10.5, color=BUDGET)


def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.2))
    fig.subplots_adjust(left=0.20, right=0.985, top=0.845, bottom=0.065, hspace=0.30)

    fig.text(0.012, 0.975,
             "그림 1. 지연 예산 — 탐지는 라이브에서, 개입은 화면에서 일어난다",
             fontsize=15.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.928,
             "실선 = 탐지기가 듣는 라이브 신호   ·   점선 = 사용자가 보는 화면(L초 늦다)   ·   "
             "굵은 세로선 = 사건   ·   얇은 세로 점선 = 시점 표시",
             fontsize=10.5, color=GRAY, va="top")
    fig.text(0.012, 0.893,
             "onset = 사건이 시작되는 시각(소리가 올라가기 시작하는 순간)   ·   "
             "d = onset부터 탐지기가 확신할 때까지   ·   L = 화면을 늦추는 양",
             fontsize=10.5, color=GRAY, va="top")

    panel(ax1, "(a) 지연 없음 (L = 0) — 개입할 시간이 없다", True)
    panel(ax2, "(b) 제안: 지연을 의도적으로 도입 (L = 3.0 s)", False)

    fig.text(0.012, 0.012,
             "가로축은 시간이지만 축척이 아니다 — 25 ms를 보이게 하려고 크게 벌려 그렸다 "
             "(실제 비율 25 ms : 3 s = 1 : 120).",
             fontsize=9.5, color=GRAY)

    out = Path(__file__).parent
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"delay-budget.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    print("저장 완료")


if __name__ == "__main__":
    main()
