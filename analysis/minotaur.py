import multiprocessing
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

MINOTAUR_OUTPUT_DIR = Path.cwd().parent / 'minotaur' / 'output'

DATASETS = ['CAL500',
            'emotions',
            'scene',
            'synthetic0',
            'synthetic1',
            'synthetic2',
            'synthetic3',
            'yeast']

RUN_COUNT = 30
FOLD_COUNT = 10

FIGURE_OUTPUT_PATH = Path.cwd().parent / 'temp'
FIGURE_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def _get_output_path(run: int, dataset: str, fold: int) -> Path:
    return MINOTAUR_OUTPUT_DIR / f"run-{run}-dataset-{dataset}-fold-{fold}-output.csv"


def parse_minotaur_output(run: int, dataset: str, fold: int) -> pd.DataFrame:
    filename = _get_output_path(run=run, dataset=dataset, fold=fold)
    df = pd.read_csv(filepath_or_buffer=filename, index_col=False)
    df = df[['Individual Id', 'MultiLabelFScore', 'MultiLabelFScore.1', 'RuleCount.1']]
    df = df.rename(columns={'Individual Id': 'individual_id',
                            'MultiLabelFScore': 'fscore_train',
                            'MultiLabelFScore.1': 'fscore_test',
                            'RuleCount.1': 'rule_count'})
    df['run'] = run
    df['dataset'] = dataset
    df['fold'] = fold

    # MINOTAUR tries to maximize all objectives, so internally
    # the values of rule counts are stored as negative numbers.
    # Here, we are multiplying by -1 to obtain the actual rule counts
    df['rule_count'] = -1 * df['rule_count']
    return df


def get_best_model_stats(df: pd.DataFrame) -> Tuple[float, int]:
    df = df.sort_values(by=['fscore_train', 'rule_count'],
                        ascending=[False, True])
    row = df.iloc[0]
    fscore = float(row['fscore_test'])
    rule_count = int(row['rule_count'])
    return fscore, rule_count


def get_run_stats(run: int, dataset: str):
    fscores = []
    rule_counts = []
    for fold in range(FOLD_COUNT):
        df = parse_minotaur_output(run=run, dataset=dataset, fold=fold)
        fs, rc = get_best_model_stats(df)
        fscores.append(fs)
        rule_counts.append(rc)

    return {'avg_fscore': np.mean(fscores),
            'std_fscore': np.std(fscores),
            'avg_rule_count': np.mean(rule_counts),
            'std_rule_count': np.std(rule_counts)}


def get_dataset_stats(dataset: str):
    avg_fscores = []
    std_fscores = []
    avg_rule_count = []
    std_rule_count = []

    for run in range(RUN_COUNT):
        stats = get_run_stats(run=run, dataset=dataset)

        avg_fscores.append(stats['avg_fscore'])
        std_fscores.append(stats['std_fscore'])

        avg_rule_count.append(stats['avg_rule_count'])
        std_rule_count.append(stats['std_rule_count'])

    return {'mm_fscore': np.mean(avg_fscores),
            'md_fscore': np.mean(std_fscores),
            'dm_fscore': np.std(avg_fscores),
            'dd_fscore': np.std(std_fscores),

            'mm_rule_count': np.mean(avg_rule_count),
            'md_rule_count': np.mean(std_rule_count),
            'dm_rule_count': np.std(avg_rule_count),
            'dd_rule_count': np.std(std_rule_count)}


def print_stats():
    pool = multiprocessing.Pool()
    stats = pool.map(get_dataset_stats, DATASETS)
    stats = dict(zip(DATASETS, stats))
    df = pd.DataFrame.from_dict(data=stats, orient='index')
    with pd.option_context('display.max_columns', None):
        print(df)


def is_dominated(target_index: int, pts: np.ndarray) -> bool:
    target = pts[target_index]
    pts_count, dimension_count = pts.shape
    for i in range(pts_count):
        other = pts[i]
        if (target <= other).all() and (other > target).any():
            return True

    return False


def compute_domination_status(pts: np.ndarray) -> np.ndarray:
    pts_count = pts.shape[0]
    dominated = [is_dominated(target_index=i, pts=pts)
                 for i in range(pts_count)]
    return np.asarray(dominated)


def plot_non_dominated_averaged(dataset: str):
    dfs = []
    for run in range(RUN_COUNT):
        for fold in range(FOLD_COUNT):
            df = parse_minotaur_output(run=run, dataset=dataset, fold=fold)
            df = df[['fscore_test', 'rule_count']]
            df['rule_count'] = 1 / df['rule_count']
            df = df.sort_values(by=['fscore_test', 'rule_count'], ascending=[False, True])
            dfs.append(df)

    data = [df.values for df in dfs]
    data = sum(data) / len(data)
    df = pd.DataFrame(data=data, columns=['fscore_test', 'rule_count'])

    dominated = compute_domination_status(df.values)
    non_dominated = dominated == False
    df = df[non_dominated]

    # We must call figure, because matplotlib utilized a global state machine...
    ax = plt.figure()
    ax = sns.scatterplot(x=df['rule_count'], y=df['fscore_test'], color='black')
    ax.set(xlabel='Interpretability', ylabel='Predictive power')
    ax.set(xlim=(0, 1))
    # ax.set_title(f"Dataset={dataset}")
    plt.savefig(fname=str(FIGURE_OUTPUT_PATH / dataset) + '.pdf', bbox_inches='tight', dpi=300)


def generate_figures():
    pool = multiprocessing.Pool()
    pool.map(plot_non_dominated_averaged, DATASETS)
    return


def main():
    print_stats()
    generate_figures()
    print("Done!")
    return


if __name__ == '__main__':
    main()
