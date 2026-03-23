import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def plot_acc_vs_round(
    MNIST_RES_PATH,
    EMNIST_PATH,
    CIFAR10_PATH,
    labels=None,
    figsize=(18, 5),
    save_path=None,
):
    """
    Plot test accuracy vs communication rounds with fixed x-axis limits.

    - MNIST / EMNIST: rounds <= 100
    - CIFAR-10: rounds <= 150
    """

    assert len(MNIST_RES_PATH) == len(EMNIST_PATH) == len(CIFAR10_PATH), \
        "All path lists must have the same length"

    num_lines = len(MNIST_RES_PATH)
    if labels is None:
        labels = [f"Method-{i+1}" for i in range(num_lines)]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].set_ylim(0.6, 1)
    axes[1].set_ylim(0.7,0.9)
    datasets = [
        ("MNIST", MNIST_RES_PATH, 100),
        ("EMNIST", EMNIST_PATH, 100),
        ("CIFAR-10", CIFAR10_PATH, 250),
    ]

    for ax, (title, path_list, max_round) in zip(axes, datasets):
        for path, label in zip(path_list, labels):
            df = pd.read_csv(path)

            # ===== 强制裁剪 =====
            df = df[df["round"] <= max_round]

            ax.plot(
                df["round"],
                df["test_acc"],
                linewidth=2,
                label=label,
            )

        ax.set_title(title)
        ax.set_xlabel("Communication Round")
        ax.set_xlim(1, max_round)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Test Accuracy")

    # ===== 全局图例（论文友好）=====
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=num_lines,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_acc_vs_round_new(
    MNIST_RES_PATH,
    EMNIST_PATH,
    CIFAR10_PATH,
    labels,
    save_path=None,
):
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    assert len(MNIST_RES_PATH) == len(EMNIST_PATH) == len(CIFAR10_PATH)

    # ===== 方法风格定义 =====
    style_map = {
        "FP-Central": dict(color="#8B4513", linestyle="--", linewidth=0.8, alpha=0.8),
        "Binary-Central": dict(color="#556B2F", linestyle="--", linewidth=0.8, alpha=0.8),
        "FP-FedAvg": dict(color="#1f77b4", linewidth=1.0),
        "Binary-Local": dict(color="#ff7f0e", linewidth=1.0),
        "Binary-FedAvg": dict(color="#9467bd", linewidth=1.0),
        "SR-BinAgg": dict(color="#DC143C", linewidth=1.5),
    }

    marker_map = {
        "FP-FedAvg": "o",
        "Binary-Local": "s",
        "Binary-FedAvg": "^",
        "SR-BinAgg": "D",
    }

    fig = plt.figure(figsize=(20, 5))
    gs = GridSpec(1, 4, figure=fig)
    ax_mnist = fig.add_subplot(gs[0, 0])
    ax_emnist = fig.add_subplot(gs[0, 1])
    ax_cifar = fig.add_subplot(gs[0, 2:])

    datasets = [
        ("MNIST", MNIST_RES_PATH, ax_mnist, 100, (0.82, 1.0)),
        ("FEMNIST", EMNIST_PATH, ax_emnist, 100, (0.8, 0.875)),
        ("CIFAR-10", CIFAR10_PATH, ax_cifar, 250, (0.2, 1.0)),
    ]

    for title, paths, ax, max_round, ylim in datasets:
        for path, label in zip(paths, labels):
            df = pd.read_csv(path)
            df = df[df["round"] <= max_round]

            # ===== 有选择的平滑 =====
            if title in ["FEMNIST"]:
                df["test_acc"] = df["test_acc"].rolling(window=5, min_periods=1).mean()
            if title in ["CIFAR-10"]:
                df["test_acc"] = df["test_acc"].rolling(window=10, min_periods=1).mean()

            style = style_map.get(label, {})
            marker = marker_map.get(label, None)

            mark_step = 25 if title == "CIFAR-10" else 10

            ax.plot(
                df["round"],
                df["test_acc"],
                label=label,
                markevery=mark_step if marker else None,
                marker=marker,
                markersize=4,
                **style,
            )

        ax.set_title(title)
        ax.set_xlabel("Communication Round")
        ax.set_xlim(1, max_round)
        ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", alpha=0.4)

    ax_mnist.set_ylabel("Test Accuracy")

    # ===== 全局图例 =====
    handles, labels = ax_mnist.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()



from matplotlib.gridspec import GridSpec


def plot_acc_vs_comm_new(
    MNIST_RES_PATH,
    EMNIST_PATH,
    CIFAR10_PATH,
    labels,
    comm_cost_map,
    save_path=None,
):
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    assert len(MNIST_RES_PATH) == len(EMNIST_PATH) == len(CIFAR10_PATH) == 4
    assert len(labels) == 4

    style_map = {
        "FP-FedAvg": dict(color="#1f77b4", linewidth=1.0),
        "Binary-Local": dict(color="#ff7f0e", linewidth=1.0),
        "Binary-FedAvg": dict(color="#9467bd", linewidth=1.0),
        "SR-BinAgg": dict(color="#DC143C", linewidth=1.5),
    }

    marker_map = {
        "FP-FedAvg": "o",
        "Binary-Local": "s",
        "Binary-FedAvg": "^",
        "SR-BinAgg": "D",
    }

    fig = plt.figure(figsize=(20, 5))
    gs = GridSpec(1, 4, figure=fig)
    ax_mnist = fig.add_subplot(gs[0, 0])
    ax_emnist = fig.add_subplot(gs[0, 1])
    ax_cifar = fig.add_subplot(gs[0, 2:])

    datasets = [
        ("MNIST", MNIST_RES_PATH, ax_mnist, 100, (0.82, 1.0)),
        ("FEMNIST", EMNIST_PATH, ax_emnist, 100, (0.8, 0.875)),
        ("CIFAR-10", CIFAR10_PATH, ax_cifar, 250, (0.2, 1.0)),
    ]

    for title, paths, ax, max_round, ylim in datasets:
        for path, label in zip(paths, labels):
            df = pd.read_csv(path)
            df = df[df["round"] <= max_round]

            # ===== 平滑（保持一致）=====
            if title == "FEMNIST":
                df["test_acc"] = df["test_acc"].rolling(window=5, min_periods=1).mean()
            if title == "CIFAR-10":
                df["test_acc"] = df["test_acc"].rolling(window=10, min_periods=1).mean()

            # ===== dataset-aware communication cost =====
            per_round_mb = comm_cost_map[title][label]
            df["comm_cost"] = df["round"] * per_round_mb

            style = style_map[label]
            marker = marker_map[label]
            mark_step = 25 if title == "CIFAR-10" else 10

            ax.plot(
                df["comm_cost"],
                df["test_acc"],
                label=label,
                marker=marker,
                markevery=mark_step,
                markersize=4,
                **style,
            )
            #ax.set_xlim(0, xlim)
        ax.set_title(title)
        ax.set_xlabel("Communication Cost (MB)")
        ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", alpha=0.4)

    ax_mnist.set_ylabel("Test Accuracy")


    # ===== MNIST inset =====
    axins = inset_axes(
        ax_mnist,
        width="70%",     # inset 宽度（相对主图）
        height="45%",
        loc="lower right",
        borderpad=2,
    )

    for path, label in zip(MNIST_RES_PATH, labels):
        df = pd.read_csv(path)
        df = df[df["round"] <= 100]

        per_round_mb = comm_cost_map["MNIST"][label]
        df["comm_cost"] = df["round"] * per_round_mb

        # 只画 inset 区域
        df = df[df["comm_cost"] <= 20]

        style = style_map[label]
        marker = marker_map[label]

        axins.plot(
            df["comm_cost"],
            df["test_acc"],
            marker=marker,
            markevery=5,
            markersize=3,
            **style,
        )

    axins.set_xlim(0, 20)
    axins.set_ylim(0.825, 0.98)
    axins.set_xticks([0, 10, 20])
    axins.set_yticks([0.85, 0.90, 0.95])
    axins.grid(True, linestyle="--", alpha=0.4)

    # 主图 ↔ inset 连线
    mark_inset(ax_mnist, axins, loc1=2, loc2=4, fc="none", ec="gray", lw=0.8,linestyle="--")

        # FEMNIST inset
    axins = inset_axes(ax_emnist, width="70%", height="45%", loc="lower right", borderpad=2)

    for path, label in zip(EMNIST_PATH, labels):
        df = pd.read_csv(path)
        df = df[df["round"] <= 100]

        per_round_mb = comm_cost_map["FEMNIST"][label]
        df["comm_cost"] = df["round"] * per_round_mb
        df = df[df["comm_cost"] <= 300]

        style = style_map[label]
        marker = marker_map[label]
        df["test_acc"] = df["test_acc"].rolling(
            window=5, min_periods=1
        ).mean()
        axins.plot(df["comm_cost"], df["test_acc"], marker=marker,markevery=5 , markersize=3, **style)

    axins.set_xlim(0, 320)
    axins.set_ylim(0.82, 0.863)
    axins.set_xticks([0, 160, 320])
    axins.grid(True, linestyle="--", alpha=0.4)

    mark_inset(ax_emnist, axins, loc1=2, loc2=4, fc="none", ec="gray", lw=0.8,linestyle="--")

    # CIFAR-10 inset
    axins = inset_axes(ax_cifar, width="70%", height="45%", loc="lower right", borderpad=2)

    for path, label in zip(CIFAR10_PATH, labels):
        df = pd.read_csv(path)
        df = df[df["round"] <= 250]

        per_round_mb = comm_cost_map["CIFAR-10"][label]
        df["comm_cost"] = df["round"] * per_round_mb
        df = df[df["comm_cost"] <= 1000]

        style = style_map[label]
        marker = marker_map[label]

        df["test_acc"] = df["test_acc"].rolling(
            window=5, min_periods=1
        ).mean()
        axins.plot(
            df["comm_cost"],
            df["test_acc"],
            marker=marker,
            markevery=5,
            markersize=3,
            **style,
        )
    axins.set_xlim(0, 800)
    axins.set_ylim(0.25, 0.65)
    axins.set_xticks([0, 400, 800])
    axins.grid(True, linestyle="--", alpha=0.4)

    mark_inset(ax_cifar, axins, loc1=2, loc2=4, fc="none", ec="gray", lw=0.8,linestyle="--")




    handles, legend_labels = ax_mnist.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_acc_round_and_comm_final(
    # --- top row (acc vs round): 6 lines ---
    MNIST_ROUND_PATHS,
    FEMNIST_ROUND_PATHS,
    CIFAR10_ROUND_PATHS,
    round_labels,   # len=6

    # --- bottom row (acc vs comm): 4 lines ---
    MNIST_COMM_PATHS,
    FEMNIST_COMM_PATHS,
    CIFAR10_COMM_PATHS,
    comm_labels,    # len=4
    comm_cost_map,  # dataset -> method -> per-round MB

    save_path=None,
):
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    assert len(MNIST_ROUND_PATHS) == len(FEMNIST_ROUND_PATHS) == len(CIFAR10_ROUND_PATHS) == len(round_labels)
    assert len(MNIST_COMM_PATHS)  == len(FEMNIST_COMM_PATHS)  == len(CIFAR10_COMM_PATHS)  == len(comm_labels)

    # ===== Style (统一一套) =====
    style_map = {
        "FP-Central":    dict(color="#8B4513", linestyle="--", linewidth=0.8, alpha=0.8),
        "Binary-Central":dict(color="#556B2F", linestyle="--", linewidth=0.8, alpha=0.8),
        "FP-FedAvg":     dict(color="#1f77b4", linewidth=1.0),
        "Binary-Local":  dict(color="#ff7f0e", linewidth=1.0),
        "Binary-FedAvg": dict(color="#9467bd", linewidth=1.0),
        "SR-BinAgg":     dict(color="#DC143C", linewidth=1.5),
    }

    marker_map = {
        "FP-FedAvg": "o",
        "Binary-Local": "s",
        "Binary-FedAvg": "^",
        "SR-BinAgg": "D",
        # central 两条通常不加 marker（你原来也没有）
    }

    # ===== Layout: 2 x 3 =====
    fig = plt.figure(figsize=(20, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.28, wspace=0.18)

    ax_r_mnist  = fig.add_subplot(gs[0, 0])
    ax_r_femnist= fig.add_subplot(gs[0, 1])
    ax_r_cifar  = fig.add_subplot(gs[0, 2])

    ax_c_mnist  = fig.add_subplot(gs[1, 0])
    ax_c_femnist= fig.add_subplot(gs[1, 1])
    ax_c_cifar  = fig.add_subplot(gs[1, 2])

    top_sets = [
        ("MNIST",   MNIST_ROUND_PATHS,  ax_r_mnist,   100, (0.82, 1.0)),
        ("FEMNIST", FEMNIST_ROUND_PATHS,ax_r_femnist, 100, (0.8, 0.875)),
        ("CIFAR-10",CIFAR10_ROUND_PATHS,ax_r_cifar,   250, (0.2, 1.0)),
    ]

    bot_sets = [
        ("MNIST",   MNIST_COMM_PATHS,   ax_c_mnist,   100, (0.82, 1.0)),
        ("FEMNIST", FEMNIST_COMM_PATHS, ax_c_femnist, 100, (0.8, 0.875)),
        ("CIFAR-10",CIFAR10_COMM_PATHS, ax_c_cifar,   250, (0.2, 1.0)),
    ]

    # =======================
    # Top row: Acc vs Round (6 lines)
    # =======================
    for title, paths, ax, max_round, ylim in top_sets:
        for path, label in zip(paths, round_labels):
            df = pd.read_csv(path)
            df = df[df["round"] <= max_round]

            # 平滑规则保持一致
            if title == "FEMNIST":
                df["test_acc"] = df["test_acc"].rolling(window=5, min_periods=1).mean()
            if title == "CIFAR-10":
                df["test_acc"] = df["test_acc"].rolling(window=10, min_periods=1).mean()

            marker = marker_map.get(label, None)
            mark_step = 25 if title == "CIFAR-10" else 10

            ax.plot(
                df["round"],
                df["test_acc"],
                label=label,
                marker=marker,
                markevery=mark_step if marker else None,
                markersize=4,
                **style_map.get(label, {}),
            )

        ax.set_title(title,fontsize = 17)
        ax.set_xlabel("Communication Round",fontsize = 16)
        ax.set_xlim(1, max_round)
        ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(axis="both", labelsize=14)

    ax_r_mnist.set_ylabel("Test Accuracy",fontsize = 16)

    # 只从第一张子图收集 legend（6条）
    handles, legend_labels = ax_r_mnist.get_legend_handles_labels()

    # =======================
    # Bottom row: Acc vs Comm (4 lines)
    # =======================
    for title, paths, ax, max_round, ylim in bot_sets:
        for path, label in zip(paths, comm_labels):
            df = pd.read_csv(path)
            df = df[df["round"] <= max_round]

            if title == "FEMNIST":
                df["test_acc"] = df["test_acc"].rolling(window=5, min_periods=1).mean()
            if title == "CIFAR-10":
                df["test_acc"] = df["test_acc"].rolling(window=10, min_periods=1).mean()

            df["comm_cost"] = df["round"] * comm_cost_map[title][label]

            ax.plot(
                df["comm_cost"],
                df["test_acc"],
                marker=marker_map[label],
                markevery=25 if title == "CIFAR-10" else 10,
                markersize=4,
                **style_map[label],
            )

        #ax.set_title(title)
        ax.set_xlabel("Communication Cost (MB)",fontsize = 16)
        ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(axis="both", labelsize=14)
    ax_c_mnist.set_ylabel("Test Accuracy",fontsize = 16)

    # =======================
    # Insets (只在 bottom row 上做)
    # =======================
    def draw_inset(ax_main, title, paths, x_max, y_lim, xticks):
        axins = inset_axes(ax_main, width="70%", height="45%", loc="lower right", borderpad=2)

        for path, label in zip(paths, comm_labels):
            df = pd.read_csv(path)
            df = df[df["round"] <= (250 if title == "CIFAR-10" else 100)]
            df["comm_cost"] = df["round"] * comm_cost_map[title][label]
            df = df[df["comm_cost"] <= x_max]

            # inset 里你之前对 FEMNIST/CIFAR 做了更强一点的平滑，这里保留
            if title in ["FEMNIST", "CIFAR-10"]:
                df["test_acc"] = df["test_acc"].rolling(window=5, min_periods=1).mean()

            axins.plot(
                df["comm_cost"],
                df["test_acc"],
                marker=marker_map[label],
                markevery=5,
                markersize=3,
                **style_map[label],
            )

        axins.set_xlim(0, x_max)
        axins.set_ylim(*y_lim)
        axins.set_xticks(xticks)
        axins.grid(True, linestyle="--", alpha=0.4)
        mark_inset(ax_main, axins, loc1=2, loc2=4, fc="none", ec="gray", lw=0.8, linestyle="--")
        axins.tick_params(axis="both", labelsize=12)
    draw_inset(ax_c_mnist,   "MNIST",   MNIST_COMM_PATHS,   x_max=20,   y_lim=(0.825, 0.98), xticks=[0, 10, 20])
    draw_inset(ax_c_femnist, "FEMNIST", FEMNIST_COMM_PATHS, x_max=320,  y_lim=(0.81, 0.863), xticks=[0, 160, 320])
    draw_inset(ax_c_cifar,   "CIFAR-10",CIFAR10_COMM_PATHS, x_max=800,  y_lim=(0.22, 0.7), xticks=[0, 400, 800])

    # ===== Global legend (用第一排的 6 条) =====
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
        fontsize = 16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()



def smooth_curve(y, window=5):
    """Simple moving average smoothing."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    y_pad = np.pad(y, (window // 2, window // 2), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def plot_emnist_dirichlet_comparison(
    DIR_05_PATH,
    DIR_01_PATH,
    labels=None,
    max_round=80,                     # (1) round 截断到 80
    figsize=(12, 4),
    save_path=None,
):
    """
    Plot EMNIST test accuracy vs communication rounds under different Dirichlet partitions.
    """

    assert len(DIR_01_PATH) == len(DIR_05_PATH) == 4, \
        "Each Dirichlet setting must contain exactly 4 methods"

    if labels is None:
        labels = [
            "FedAvg",
            "Binary-Local",
            "Binary-FedAvg",
            "SR-BinAgg",
        ]

    # (2) 每条线的颜色 & 线宽（论文友好）
    colors = {
        "FedAvg": "#1f77b4",              # 蓝
        "Binary-Local": "#ff7f0e", # 橙
        "Binary-FedAvg": "#9467bd",       # 绿
        "SR-BinAgg": "#DC143C",               # 红
    }

    linewidths = {
        "FedAvg": 1.0,
        "Binary-Local": 1.0,
        "Binary-FedAvg": 1.0,
        "SR-BinAgg": 1.5,                     # SR-FL 略粗，突出方法
    }

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    settings = [
        (r"FEMNIST Dirichlet Non-IID $\alpha=0.5$", DIR_05_PATH, (0.5, 0.9)),   # (3)
        (r"FEMNIST Dirichlet Non-IID $\alpha=0.1$", DIR_01_PATH, (0.3, 0.85)),  # (3)
        
    ]

    for ax, (title, path_list, ylim) in zip(axes, settings):
        for path, label in zip(path_list, labels):
            df = pd.read_csv(path)
            df = df[df["round"] <= max_round]
            df["test_acc"] = df["test_acc"].rolling(window=5, min_periods=1).mean()
            #y_smooth = smooth_curve(df["test_acc"].values, window=10)  # (4)

            ax.plot(
                df["round"].values,
                df['test_acc'].values,
                label=label,
                color=colors[label],
                linewidth=linewidths[label],
            )

        ax.set_title(title,fontsize = 17)
        ax.set_xlabel("Communication Round",fontsize = 16)
        ax.set_xlim(1, max_round)
        ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis="both", labelsize=14)
    axes[0].set_ylabel("Test Accuracy",fontsize = 16)
    axes[1].legend(
        loc="lower right",      # 右下角（子图内部）
        fontsize=14,
        frameon=True
    )
    # # 全局 legend（论文标准）
    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(
    #     handles,
    #     labels,
    #     loc="upper center",
    #     ncol=4,
    #     frameon=False,
    #     bbox_to_anchor=(0.5, 1.10),
    #     fontsize = 16
    # )

    #plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()




def plot_femnist_niid_by_writer(
    DIR_PATH1,
    DIR_PATH2,
    labels=None,
    max_round=80,
    figsize=(12, 5),
    save_path=None
):
    assert len(DIR_PATH1) == len(DIR_PATH2) == 4, \
        "Each setting must contain exactly 4 methods"

    if labels is None:
        labels = [
            "FedAvg",
            "Binary-Local-FedAvg",  # local_bin
            "Binary-FedAvg",
            "SR-BinAgg",
        ]

    # 颜色 & 线宽
    colors = {
        "FedAvg": "#1f77b4",
        "Binary-Local-FedAvg": "#ff7f0e",
        "Binary-FedAvg": "#9467bd",
        "SR-BinAgg": "#DC143C",
    }
    linewidths = {
        "FedAvg": 1.0,
        "Binary-Local-FedAvg": 1.0,
        "Binary-FedAvg": 1.0,
        "SR-BinAgg": 1.5,
    }

    # 2x2 布局（下排更扁）
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], hspace=0.08, wspace=0.15)

    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_bl = fig.add_subplot(gs[1, 0], sharex=ax_tl)
    ax_br = fig.add_subplot(gs[1, 1], sharex=ax_tr)

    # 上排标题 & y 轴范围

    ax_tl.set_title(r"FEMNIST Writer-Level Non-IID (20-5)",fontsize = 17)
    ax_tr.set_title(r"FEMNIST Writer-Level Non-IID (100-10)",fontsize = 17)
    ax_tl.set_ylim(0.7, 0.85)
    ax_tr.set_ylim(0.55, 0.825)
    ax_tl.tick_params(axis="both", labelsize=14)
    ax_tr.tick_params(axis="both", labelsize=14)
    # smoothing 相关
    SMOOTH_W = 5
    START_IDX = SMOOTH_W - 1

    # ─────────────────────────────
    # 上排：主结果（不画 local_bin）
    # ─────────────────────────────
    def plot_main(ax, path_list, show_ylabel=True):
        for path, label in zip(path_list, labels):
            if label == "Binary-Local-FedAvg":
                continue

            df = pd.read_csv(path)
            df = df[df["round"] <= max_round]

            x = df["round"].values
            y = smooth_curve(df["test_acc"].values, window=SMOOTH_W)

            ax.plot(
                x[START_IDX:],
                y[START_IDX:],
                label=label,
                color=colors[label],
                linewidth=linewidths[label],
            )

        ax.set_xlim(SMOOTH_W, max_round)
        ax.grid(True, linestyle="--", alpha=0.5)

        if show_ylabel:
            ax.set_ylabel("Test Accuracy",fontsize = 16)
        else:
            ax.set_ylabel("")

        # 上排不显示 x 轴刻度文字
        ax.tick_params(labelbottom=False)

    plot_main(ax_tl, DIR_PATH1, show_ylabel=True)
    plot_main(ax_tr, DIR_PATH2, show_ylabel=False)
    legend_handles = [
        Line2D(
            [0], [0],
            color=colors["FedAvg"],
            lw=linewidths["FedAvg"],
            label="FedAvg"
        ),
        Line2D(
            [0], [0],
            color=colors["Binary-Local-FedAvg"],
            lw=linewidths["Binary-Local-FedAvg"],
            label="Binary-Local-FedAvg"
        ),
        Line2D(
            [0], [0],
            color=colors["Binary-FedAvg"],
            lw=linewidths["Binary-FedAvg"],
            label="Binary-FedAvg"
        ),
        Line2D(
            [0], [0],
            color=colors["SR-BinAgg"],
            lw=linewidths["SR-BinAgg"],
            label="SR-BinAgg"
        ),
    ]

    ax_tr.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=14,
        frameon=True
    )
    # ─────────────────────────────
    # 下排：local_bin（左右各一张）
    # ─────────────────────────────
    def plot_local(ax, path):
        df = pd.read_csv(path)
        df = df[df["round"] <= max_round]

        x = df["round"].values
        y = smooth_curve(df["test_acc"].values, window=SMOOTH_W)

        ax.plot(
            x[START_IDX:],
            y[START_IDX:],
            color=colors["Binary-Local-FedAvg"],
            linewidth=linewidths["Binary-Local-FedAvg"],
        )

        ax.set_xlim(SMOOTH_W, max_round)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlabel("Communication Round",fontsize = 16)
        ax.set_ylabel("")  # 下排全部不显示 y 轴标签

    plot_local(ax_bl, DIR_PATH1[1])
    plot_local(ax_br, DIR_PATH2[1])

    # 下排 y 轴范围（可单独调）
    ax_bl.set_ylim(0.4, 0.8)
    ax_br.set_ylim(0.3, 0.7)

    # ─────────────────────────────
    # 全局 legend（包含 local_bin）
    # ─────────────────────────────
    # legend_elements = [
    #     Line2D([0], [0], color=colors["FedAvg"], lw=linewidths["FedAvg"], label="FedAvg"),
    #     Line2D([0], [0], color=colors["Binary-Local"], lw=linewidths["Binary-Local"],
    #            label="Binary-Local"),
    #     Line2D([0], [0], color=colors["Binary-FedAvg"], lw=linewidths["Binary-FedAvg"],
    #            label="Binary-FedAvg"),
    #     Line2D([0], [0], color=colors["SR-BinAgg"], lw=linewidths["SR-BinAgg"], label="SR-BinAgg"),
    # ]

    # fig.legend(
    #     handles=legend_elements,
    #     loc="upper center",
    #     ncol=4,
    #     frameon=False,
    #     bbox_to_anchor=(0.5, 1.02),
    # )

    #splt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

from typing import Tuple, Dict, Any, Optional,List
import os
def plot_test_acc_from_dir(
    csv_dir: str,
    max_round: int | None = None,
    title: str | None = None,
    save_path: str | None = None,
    figsize=(7, 5),
):
    """
    Plot test_acc vs round for all csv files in a directory.

    Args:
        csv_dir (str): directory containing csv files
        max_round (int | None): only plot rounds <= max_round
        title (str | None): figure title
        save_path (str | None): if set, save figure to this path
        figsize (tuple): matplotlib figure size
    """
    csv_files = sorted([
        f for f in os.listdir(csv_dir)
        if f.endswith(".csv")
    ])

    if len(csv_files) == 0:
        raise ValueError(f"No csv files found in {csv_dir}")

    plt.figure(figsize=figsize)

    for fname in csv_files:
        path = os.path.join(csv_dir, fname)
        df = pd.read_csv(path)

        if "round" not in df or "test_acc" not in df:
            raise ValueError(f"{fname} does not contain required columns")

        rounds = df["round"]
        test_acc = df["test_acc"]

        if max_round is not None:
            mask = rounds <= max_round
            rounds = rounds[mask]
            test_acc = test_acc[mask]

        label = os.path.splitext(fname)[0]
        plt.plot(rounds, test_acc, label=label)

    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy")
    if title is not None:
        plt.title(title)

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[Saved] {save_path}")
    else:
        plt.show()



def plot_test_acc_from_csv_list(
    csv_paths: List[str],
    max_round: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize=(7, 5),
):
    """
    Plot test_acc vs round for multiple csv files.

    Args:
        csv_paths (List[str]): list of csv file paths
        max_round (int, optional): only plot rounds <= max_round
        title (str, optional): figure title
        save_path (str, optional): if set, save figure
        figsize (tuple): figure size
    """
    if len(csv_paths) == 0:
        raise ValueError("csv_paths is empty")

    plt.figure(figsize=figsize)

    for path in csv_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        if "round" not in df or "test_acc" not in df:
            raise ValueError(f"{path} does not contain required columns")

        rounds = df["round"]
        test_acc = df["test_acc"]

        if max_round is not None:
            mask = rounds <= max_round
            rounds = rounds[mask]
            test_acc = test_acc[mask]

        label = os.path.splitext(os.path.basename(path))[0]
        plt.plot(rounds, test_acc, label=label)

    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy")

    if title is not None:
        plt.title(title)

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[Saved] {save_path}")
    else:
        plt.show()


def main1():
    print("PLOT")
    labels = [
        "FP-Central",
        "Binary-Central",
        "FP-FedAvg",
        "Binary-Local",
        "Binary-FedAvg",
        "SR-BinAgg",
    ]
    MNIST_RES_PATH1 = [
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_central/mnist/MLP/FP-Central/seed_0/metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_central/mnist/BinaryMLP/Binary-Central/seed_0/metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/mnist/MLP/fp_fedavg/iid/seed_0/1metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/mnist/BinaryMLP/bin_local_fedavg/iid/seed_0/1metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/mnist/BinaryMLP/bin_fedavg/iid/seed_0/1metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/mnist/BinaryMLP/cab_fl/iid/seed_0/metrics.csv'
    ]
    EMNIST_RES_PATH1 = [
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_central/femnist/LeNetBN/FP-Central005_100_decay/seed_0/metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_central/femnist/LeNetBN/FP-Central/seed_0/metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_central/femnist/LeNetBN/FP-Central005_100_decay/seed_0/metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_central/femnist/LeNetBN/FP-Central001bd/seed_0/metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_central/femnist/LeNetBN/FP-Centralhyperdecay/seed_0/metrics.csv',
    

        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_central/femnist/BinaryLeNetBN/Binary-Central01/seed_0/metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_central/femnist/BinaryLeNetBN/Binary-Central001_100_decay/seed_0/metrics.csv',

        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/LeNetBN/fp_fedavg/iid/seed_0/1metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/LeNetBN/fp_fedavg/iid/seed_0/0.01_1_20_5decay0.03-0.01_metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/LeNetBN/fp_fedavg/iid/seed_0/0.01_1_20_5_metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/LeNetBN/fp_fedavg/iid/seed_0/0.03_1_20_5_metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/LeNetBN/fp_fedavg/iid/seed_10/0.005_1_20_5_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/LeNetBN/fp_fedavg/iid/seed_25/0.005_1_20_5dc0.005 0.001_metrics.csv',

        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_local_fedavg/iid/seed_0/1metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/BinaryLeNetBN/bin_local_fedavg/iid/seed_0/0.01_1_20_5dacay0.03-0.01_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/BinaryLeNetBN/bin_local_fedavg/iid/seed_0/0.01_1_20_5_metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/BinaryLeNetBN/bin_local_fedavg/iid/seed_0/0.03_1_20_5_metrics.csv',

        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/BinaryLeNetBN/bin_fedavg/iid/seed_0/0.01_1_20_5decay0.03-0.01_metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/BinaryLeNetBN/bin_fedavg/iid/seed_0/0.01_1_20_5_metrics.csv',
        #尾巴上没有前两个好，可能从0.05下来更好？
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_fedavg/iid/seed_0/1metrics.csv',
        
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/BinaryLeNetBN/cab_fl/iid/seed_0/0.01_1_20_5decay0.03-0.01_metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/BinaryLeNetBN/cab_fl/iid/seed_0/0.01_1_20_5_metrics.csv',
        #确定最后一条最好，不用decay，可能0.05下来效果好
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/cab_fl/iid/seed_0/1metrics.csv'
    ]
    CIFAR10_RES_PATH2 = [
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_central/cifar10/ResNet18/FP_Central_250/seed_0/metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_central/cifar10/BinaryResNet18/Binary_Central_250/seed_0/metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/cifar10/ResNet18/fp_fedavg/iid/seed_0/0.05_3_20_5_250_decay0.05-0.005_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/cifar10/BinaryResNet18/bin_local_fedavg/iid/seed_0/0.05_3_20_5250_decay0.05-0.005_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/cifar10/BinaryResNet18/bin_fedavg/iid/seed_0/0.05_320_5_250round_decay_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/cifar10/BinaryResNet18/cab_fl/iid/seed_0/0.05_3_20_5_250round_decay_metrics.csv'
    ]
    plot_acc_vs_round_new(
        MNIST_RES_PATH1,
        EMNIST_RES_PATH1,
        CIFAR10_RES_PATH2,
        labels=labels,
        save_path="acc_vs_round_final.pdf",
    )

def main2():
    DIR_01_PATH = [
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/LeNetBN/fp_fedavg/dirichlet_a0.1/seed_0/0.05_3_20_5metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_local_fedavg/dirichlet_a0.1/seed_0/0.05_3_20_5metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_fedavg/dirichlet_a0.1/seed_0/metrics.csv',       
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/cab_fl/dirichlet_a0.1/seed_0/metrics.csv',
    ]
    DIR_05_PATH = [
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/LeNetBN/fp_fedavg/dirichlet_a0.5/seed_0/0.05_3_20_5metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_local_fedavg/dirichlet_a0.5/seed_0/0.05_3_20_5metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_fedavg/dirichlet_a0.5/seed_0/0.05_3_metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/cab_fl/dirichlet_a0.5/seed_0/0.05_3_20_5_metrics.csv'
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/LeNetBN/fp_fedavg/dirichlet_a0.5/seed_0/0.03_1_20_5_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/BinaryLeNetBN/bin_local_fedavg/dirichlet_a0.5/seed_0/0.03_1_20_5_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/BinaryLeNetBN/bin_fedavg/dirichlet_a0.5/seed_0/0.03_120_5_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/BinaryLeNetBN/cab_fl/dirichlet_a0.5/seed_0/0.03_1_20_5_metrics.csv'
    ]
    
    plot_emnist_dirichlet_comparison(
        DIR_05_PATH,
        DIR_01_PATH,
        max_round=80,
        save_path="dirichlet_acclfinal.pdf",
    )

def main3():
    #femnist niid by writer
    DIR_20_5_PATH = [
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/LeNetBN/fp_fedavg/femnist/seed_0/0.03_1_20_5[0, 1, 2, 3, 4, 5, 6]_metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/LeNetBN/fp_fedavg/femnist/seed_0/0.03_1_20_5_metrics.csv',
        
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_local_fedavg/femnist/seed_0/0.03_1_20_5_metrics.csv',
        
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_fedavg/femnist/seed_0/0.05_1_20_5[0, 1, 2, 3, 4, 5, 6]_metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_fedavg/femnist/seed_0/0.02_120_5_metrics.csv',
        
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/cab_fl/femnist/seed_0/0.1_1_20_5[0, 1, 2, 3, 4, 5, 6]_metrics.csv'
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/cab_fl/femnist/seed_0/0.02_1_20_5_metrics.csv'
    ]

    DIR_SUP = [
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/LeNetBN/fp_fedavg/femnist/seed_0/0.03_1_20_5_metrics.csv',
        
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_local_fedavg/femnist/seed_0/0.03_1_20_5_metrics.csv',
        
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_fedavg/femnist/seed_0/0.02_120_5_metrics.csv',
        
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/cab_fl/femnist/seed_0/0.02_1_20_5_metrics.csv'
    ]
    DIR_100_10_PATH = [
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/LeNetBN/fp_fedavg/femnist/seed_0/0.05_1_100_10[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/LeNetBN/fp_fedavg/femnist/seed_0/0.1_1_100_10[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]_metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/LeNetBN/fp_fedavg/femnist/seed_0/0.08_1_100_10[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]_metrics.csv',

        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_local_fedavg/femnist/seed_0/0.03_1_20_5_metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_local_fedavg/femnist/seed_0/0.01_1_20_5[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]_metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_local_fedavg/femnist/seed_0/0.01_1_20_5[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_local_fedavg/femnist/seed_0/0.03_1_20_5[0, 1, 2, 3, 4, 5]decay0.03-0.0001_metrics.csv',
        
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_fedavg/femnist/seed_0/0.05_1_100_10[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]_metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_fedavg/femnist/seed_0/0.02_1_100_10[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]_metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_fedavg/femnist/seed_0/0.1_1_100_10[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]_metrics.csv',

        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/cab_fl/femnist/seed_0/1110.1_1_100_10[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]_metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/cab_fl/femnist/seed_0/0.05_1_100_10[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]_metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/cab_fl/femnist/seed_0/0.02_1_100_10[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]decay0.05-0.0001_metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/cab_fl/femnist/seed_0/0.02_1_100_10[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]_metrics.csv',
    ]
    plot_femnist_niid_by_writer(
        DIR_20_5_PATH,
        DIR_100_10_PATH,
        #DIR_SUP,
        max_round=80,
        save_path="femnist_niid.pdf",
    )
def main4():
    labels = [
        "FP-FedAvg",
        "Binary-Local",
        "Binary-FedAvg",
        "SR-BinAgg",
    ]
    MNIST_RES_PATH = [
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/mnist/MLP/fp_fedavg/iid/seed_0/1metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/mnist/BinaryMLP/bin_local_fedavg/iid/seed_0/1metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/mnist/BinaryMLP/bin_fedavg/iid/seed_0/1metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/mnist/BinaryMLP/cab_fl/iid/seed_0/metrics.csv'
    ]
    FEMNIST_RES_PATH = [
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/LeNetBN/fp_fedavg/iid/seed_25/0.005_1_20_5dc0.005 0.001_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/BinaryLeNetBN/bin_local_fedavg/iid/seed_0/0.01_1_20_5_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_fedavg/iid/seed_0/1metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/cab_fl/iid/seed_0/1metrics.csv'
    ]
    CIFAR10_RES_PATH = [
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/cifar10/ResNet18/fp_fedavg/iid/seed_0/0.05_3_20_5_250_decay0.05-0.005_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/cifar10/BinaryResNet18/bin_local_fedavg/iid/seed_0/0.05_3_20_5250_decay0.05-0.005_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/cifar10/BinaryResNet18/bin_fedavg/iid/seed_0/0.05_320_5_250round_decay_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/cifar10/BinaryResNet18/cab_fl/iid/seed_0/0.05_3_20_5_250round_decay_metrics.csv'
    ]

    comm_cost_map = {
        "MNIST": {
            "FP-FedAvg": 3.5970,
            "Binary-Local": 3.5970,
            "Binary-FedAvg": 0.1466,
            "SR-BinAgg": 0.1466,
        },
        "FEMNIST": {
            "FP-FedAvg": 49.6160,
            "Binary-Local": 49.6160,
            "Binary-FedAvg": 1.7657,
            "SR-BinAgg": 1.7657,
        },
        "CIFAR-10": {
            "FP-FedAvg": 85.3240,
            "Binary-Local": 85.3240,
            "Binary-FedAvg": 2.8598,
            "SR-BinAgg": 2.8600,
        },
    }

    plot_acc_vs_comm_new(
        MNIST_RES_PATH,
        FEMNIST_RES_PATH,
        CIFAR10_RES_PATH,
        labels,
        comm_cost_map,
        save_path="acc_vs_comm.pdf",
    )

def main5():
    labels1 = [
        "FP-Central",
        "Binary-Central",
        "FP-FedAvg",
        "Binary-Local",
        "Binary-FedAvg",
        "SR-BinAgg",
    ]
    MNIST_RES_PATH1 = [
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_central/mnist/MLP/FP-Central/seed_0/metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_central/mnist/BinaryMLP/Binary-Central/seed_0/metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/mnist/MLP/fp_fedavg/iid/seed_0/1metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/mnist/BinaryMLP/bin_local_fedavg/iid/seed_0/1metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/mnist/BinaryMLP/bin_fedavg/iid/seed_0/1metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/mnist/BinaryMLP/cab_fl/iid/seed_0/metrics.csv'
    ]
    FEMNIST_RES_PATH1 = [
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_central/femnist/LeNetBN/FP-Central001bd/seed_0/metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_central/femnist/BinaryLeNetBN/Binary-Central01/seed_0/metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_central/femnist/BinaryLeNetBN/Binary-Central001_100_decay/seed_0/metrics.csv',

        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/LeNetBN/fp_fedavg/iid/seed_25/0.005_1_20_5dc0.005 0.001_metrics.csv',

        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/BinaryLeNetBN/bin_local_fedavg/iid/seed_0/0.01_1_20_5_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_fedavg/iid/seed_0/1metrics.csv',
        #确定最后一条最好，不用decay，可能0.05下来效果好
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/cab_fl/iid/seed_0/1metrics.csv'
    ]
    CIFAR10_RES_PATH1 = [
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_central/cifar10/ResNet18/FP_Central_250/seed_0/metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_central/cifar10/BinaryResNet18/Binary_Central_250/seed_0/metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/cifar10/ResNet18/fp_fedavg/iid/seed_0/0.05_3_20_5_250_decay0.05-0.005_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/cifar10/BinaryResNet18/bin_local_fedavg/iid/seed_0/0.05_3_20_5250_decay0.05-0.005_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/cifar10/BinaryResNet18/bin_fedavg/iid/seed_0/0.05_3_20_5decay01-0001_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/cifar10/BinaryResNet18/cab_fl/iid/seed_0/0.1_3_20_5decay01-0001_metrics.csv'
    ]

    labels = [
        "FP-FedAvg",
        "Binary-Local",
        "Binary-FedAvg",
        "SR-BinAgg",
    ]
    MNIST_RES_PATH = [
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/mnist/MLP/fp_fedavg/iid/seed_0/1metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/mnist/BinaryMLP/bin_local_fedavg/iid/seed_0/1metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/mnist/BinaryMLP/bin_fedavg/iid/seed_0/1metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/mnist/BinaryMLP/cab_fl/iid/seed_0/metrics.csv'
    ]
    FEMNIST_RES_PATH = [
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/LeNetBN/fp_fedavg/iid/seed_25/0.005_1_20_5dc0.005 0.001_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/BinaryLeNetBN/bin_local_fedavg/iid/seed_0/0.01_1_20_5_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/bin_fedavg/iid/seed_0/1metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/femnist/BinaryLeNetBN/cab_fl/iid/seed_0/1metrics.csv'
    ]
    CIFAR10_RES_PATH = [
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/cifar10/ResNet18/fp_fedavg/iid/seed_0/0.05_3_20_5_250_decay0.05-0.005_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/cifar10/BinaryResNet18/bin_local_fedavg/iid/seed_0/0.05_3_20_5250_decay0.05-0.005_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/cifar10/BinaryResNet18/bin_fedavg/iid/seed_0/0.05_3_20_5decay01-0001_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/cifar10/BinaryResNet18/cab_fl/iid/seed_0/0.1_3_20_5decay01-0001_metrics.csv'
    ]

    comm_cost_map = {
        "MNIST": {
            "FP-FedAvg": 3.5970,
            "Binary-Local": 3.5970,
            "Binary-FedAvg": 0.1466,
            "SR-BinAgg": 0.1466,
        },
        "FEMNIST": {
            "FP-FedAvg": 49.6160,
            "Binary-Local": 49.6160,
            "Binary-FedAvg": 1.7657,
            "SR-BinAgg": 1.7657,
        },
        "CIFAR-10": {
            "FP-FedAvg": 85.3240,
            "Binary-Local": 85.3240,
            "Binary-FedAvg": 2.8598,
            "SR-BinAgg": 2.8600,
        },
    }

    plot_acc_round_and_comm_final(
    MNIST_RES_PATH1, FEMNIST_RES_PATH1, CIFAR10_RES_PATH1, labels1,
    MNIST_RES_PATH,  FEMNIST_RES_PATH,  CIFAR10_RES_PATH,  labels,
    comm_cost_map,
    save_path="main_figure.pdf",
    )



def main6():
    Flie_list = [
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/BinaryLeNetBN/cab_fl/dirichlet_a0.1/seed_0/0.03_1_20_15_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/BinaryLeNetBN/cab_fl/dirichlet_a0.1/seed_0/0.03_1_20_15mu0.4lam0_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/BinaryLeNetBN/cab_fl/dirichlet_a0.1/seed_0/0.03_1_20_15mu0.4lam1-3_metrics.csv',
        '/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/BinaryLeNetBN/cab_fl/dirichlet_a0.1/seed_0/0.03_1_20_15mu0.4lam1-4_metrics.csv',
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/BinaryLeNetBN/cab_fl/dirichlet_a0.1/seed_0/0.03_1_20_15mu0.4lam5-4_metrics.csv',
    
        #'/home/train_accounting/workspace/wyn/IJCNN2026/runs_fl/emnist/BinaryLeNetBN/bin_fedavg/dirichlet_a0.1/seed_0/0.03_1_20_15_metrics.csv'
    ]
    plot_test_acc_from_csv_list(
        Flie_list,
        100,
        'Test20-15',
        '/home/train_accounting/workspace/wyn/IJCNN2026/Test20-15.pdf',
        figsize=(7, 5),
    )

if __name__ == '__main__':
    #main1()
    #main2()
    #main3()
    #main4()
    #main5()
    main6()