import itertools
import multiprocessing
import platform
import subprocess
from pathlib import Path

_SYSTEM = platform.system()
if _SYSTEM == 'Windows':
    MINOTAUR_PATH = Path.cwd() / 'bin' / 'minotaur-win10-x64.exe'
elif _SYSTEM == 'Linux':
    MINOTAUR_PATH = Path.cwd() / 'bin' / 'minotaur-linux-x64'
else:
    raise Exception("Unknown system / paths.")

DATASETS_DIR = Path.cwd().parent / 'datasets'

RUN_COUNT = 30
FOLD_COUNT = 10

OUTPUT_BASE_DIR = Path.cwd() / 'output'
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

CFSBE_VALUES = {'CAL500': 64,
                'emotions': 128,
                'scene': 512,
                'synthetic0': 4096,
                'synthetic1': 32,
                'synthetic2': 2048,
                'synthetic3': 1024,
                'yeast': 512}

DATASET_NAMES = list(CFSBE_VALUES.keys())


def get_train_data_path(dataset_name: str, fold_nr: int) -> Path:
    return DATASETS_DIR / dataset_name / 'folds' / str(fold_nr) / 'train-data.csv'


def get_train_labels_path(dataset_name: str, fold_nr: int) -> Path:
    return DATASETS_DIR / dataset_name / 'folds' / str(fold_nr) / 'train-labels.csv'


def get_test_data_path(dataset_name: str, fold_nr: int) -> Path:
    return DATASETS_DIR / dataset_name / 'folds' / str(fold_nr) / 'test-data.csv'


def get_test_labels_path(dataset_name: str, fold_nr: int) -> Path:
    return DATASETS_DIR / dataset_name / 'folds' / str(fold_nr) / 'test-labels.csv'


def get_output_path(run: int, dataset: str, fold: int) -> Path:
    return OUTPUT_BASE_DIR / f"run-{run}-dataset-{dataset}-fold-{fold}-output.csv"


def get_stdout_redirection_path(run: int, dataset: str, fold: int) -> Path:
    return OUTPUT_BASE_DIR / f"run-{run}-dataset-{dataset}-fold-{fold}-stdout-redirection.txt"


def run_minotaur(dataset: str, fold: int, output_path: Path, stdout_redirection_path: Path):
    cfsbe_value = CFSBE_VALUES[dataset]

    args = [MINOTAUR_PATH,
            '--train-data', str(get_train_data_path(dataset, fold)),
            '--train-labels', str(get_train_labels_path(dataset, fold)),
            '--test-data', str(get_test_data_path(dataset, fold)),
            '--test-labels', str(get_test_labels_path(dataset, fold)),
            '--output-filename', str(output_path),
            '--classification-type', 'multilabel',
            '--fittest-selection', 'nsga2',
            '--fitness-metrics', 'fscore',
            '--fitness-metrics', 'rule-count',
            '--max-generations', '200',
            '--population-size', '80',
            '--mutants-per-generation', '40',
            '--cfsbe-target-instance-coverage', str(cfsbe_value),
            '--expensive-sanity-checks', 'false']

    with stdout_redirection_path.open(mode='wt') as stdout_redirection:
        print(f"Running 'MINOTAUR' on '{dataset}'-fold-'{fold}''...", flush=True)
        subprocess.run(args=args, stdout=stdout_redirection)


def create_dirs_and_run_minotaur(run: int, dataset: str, fold: int):
    output_path = get_output_path(run=run, dataset=dataset, fold=fold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_redirection_path = get_stdout_redirection_path(run=run, dataset=dataset, fold=fold)
    stdout_redirection_path.parent.mkdir(parents=True, exist_ok=True)

    run_minotaur(dataset=dataset, fold=fold, output_path=output_path, stdout_redirection_path=stdout_redirection_path)


def main():
    parameters = itertools.product(range(RUN_COUNT), DATASET_NAMES, range(FOLD_COUNT))
    p = multiprocessing.Pool(3)
    p.starmap(func=create_dirs_and_run_minotaur, iterable=parameters)


if __name__ == '__main__':
    main()
