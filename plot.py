import argparse
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from pathlib import Path

from typing import Tuple, Union


# List of markers for plotting
MARKERS = [
    '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 
    '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X'
]


def pareto_front(
    costs: np.array,
) -> np.array:
    """Function to calculate the pareto front from a pandas Series
    of Results, i.e., Mapping[str, float] objects.
    """
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(costs[is_pareto] < c, axis=1)
            is_pareto[i] = True
    return is_pareto


def plot_pareto_front(
    path: Union[str, Path],
    objectives: Tuple[str, str] = ("objective_1", "objective_2")
) -> None:
    """Plot the pareto front from a given path."""
    if isinstance(path, str):
        path = Path(path)
    assert path.is_dir(), f"Path {path} is not a directory"
    assert (path / "summary" / "full.csv").exists(), \
        f"Path {path} does not contain a summary/full.csv file. " \
        f"Either incomplete run or parameter not set."
    
    # Load data
    _df = pd.read_csv(path / "summary" / "full.csv")
    costs = np.vstack(_df["objective_to_minimize"].values)
    # assert costs.shape[1] == 2, \
    #     "Can only plot pareto front for 2 objectives"
    is_pareto = pareto_front(costs)
    pareto_costs = costs[is_pareto]
    pareto_costs = pareto_costs[np.argsort(pareto_costs[:, 0])]

    # Plot
    plt.step(
        pareto_costs[:, 0], 
        pareto_costs[:, 1],
        where="post",
    )
    plt.xlabel(objectives[0])
    plt.ylabel(objectives[1])
    plt.title("Pareto Front")
    plt.grid()
    plt.tight_layout()


def get_incumbent_trace_over_time(
    path: Union[str | Path] = None,
    use_neps_cost: bool = False,
    reduce_footprint: bool = False
) -> Tuple[list[float], list[float]]:
    """Get incumbent trace over time from a given path.

    Args:
        path: Union[str, Path]
            Path to the directory containing the results.
        use_neps_cost: bool
            Use NEPS cost instead of cumulative cost returned from the training objective.
    
    Returns:
        Tuple[list[float], list[float]]
            Incumbent trace of performance and cost.
    """
    if isinstance(path, str):
        path = Path(path)
    assert path.is_dir(), f"Path {path} is not a directory"
    assert (path / "summary_csv").exists(), \
        f"Path {path} does not contain a summary_csv directory. " \
        f"Either incomplete run or parameter not set."
    assert (path / "summary_csv" / "config_data.csv").is_file(), \
        f"Path {path} does not contain a config_data.csv file. " \
        f"Check run with a small run budget."
    
    # Load data
    df = pd.read_csv(path / "summary_csv" / "config_data.csv")
    df = df.loc[~(df.status.values == "pending")]
    df = df.sort_values(by=["metadata.time_end", "metadata.time_sampled"])

    # Calculate costs
    if use_neps_cost:
        cost = df["metadata.time_end"] - df["metadata.time_sampled"].values[0]
    else:
        cost = df["result.cost"].cumsum()
    
    # Calculate incumbent trace of performance
    # TODO: incumbent trace tracks the best performance seen till any time t
    performance = df["result.loss"].cummin().values

    # Reduce footprint
    if reduce_footprint:
        _df = pd.Series(performance, index=cost)
        _df = _df.loc[~_df.duplicated(keep='first') | (_df.index == _df.index[-1])]
        performance = _df.values
        cost = _df.index.values
    
    return performance, cost


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_directory",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        type=str,
        help="The list of algos to compare. Pass the output folder name that represents the algo.",
    )
    parser.add_argument(
        "--use_neps_cost",
        action="store_true",
        help="Use NEPS cost instead of cumulative cost returned from the training objective."
    )
    parser.add_argument(
        "--log_x",
        action="store_true",
        help="Use log scale for x-axis."
    )
    parser.add_argument(
        "--log_y",
        action="store_true",
        help="Use log scale for y-axis."
    )
    parser.add_argument(
        "--save_directory",
        default=None,
        help="If specified, save the plot to this directory, else, show the plot."
    )
    parser.add_argument(
        "--filename",
        default="incumbent_trace.png",
        help="Filename to save the plot."
    )
    parser.add_argument(
        "--do",
        default="incumbent_trace",
        choices=["incumbent_trace", "pareto_front"],
        help="What to plot. "
            "If 'incumbent_trace', plots the incumbent trace over time. "
            "If 'pareto_front', plots the pareto front from the results."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # Checks
    args.root_directory = Path(args.root_directory)
    assert args.root_directory.exists(), f"Path {args.root_directory} does not exist!"


    plt.figure(figsize=(8, 5))
    match args.do:
        case "pareto_front":
            plot_pareto_front(
                args.root_directory,
            )
        case "incumbent_trace":
            assert args.algos is not None, "Please specify the algos to compare."
            for algo in args.algos:
                assert (args.root_directory / algo).exists(), \
                    f"Path {args.root_directory / algo} does not exist!"

            # Plot
            for i, algo in enumerate(args.algos):
                inc_performance, cost = get_incumbent_trace_over_time(
                    args.root_directory / algo,
                    use_neps_cost=args.use_neps_cost,
                    reduce_footprint=True
                )
                plt.step(
                    cost,
                    inc_performance,
                    label=algo,
                    marker=MARKERS[i],
                    markersize=6,
                    linewidth=2,
                    where="post"
                )
            if args.log_x:
                plt.xscale("log")
            if args.log_y:
                plt.yscale("log")
            plt.xlabel("Wallclock time (in s)", fontsize=15)
            plt.ylabel("Loss", fontsize=15)
            # Formatting y-axis
            formatter = ticker.FuncFormatter(lambda y, _: '{:.2e}'.format(y))
            plt.gca().yaxis.set_major_formatter(formatter)
            plt.legend(prop={"size": 15})
            plt.tight_layout()
        case _:
            raise ValueError(
                f"Unknown value for --do: {args.do}. "
                f"Expected 'incumbent_trace' or 'pareto_front'."
            )

    # Saving plot
    save_directory = (
        Path(args.save_directory) if args.save_directory is not None else args.root_directory
    )
    plt.savefig(save_directory / args.filename, dpi=300)
    if args.save_directory is None:
        # Show plot if save_directory is not specified
        plt.show()
