import argparse
import os
import yaml
import csv
import matplotlib.pyplot as plt
import numpy as np

from glob import glob
from collections import OrderedDict

METRIC_NAMES = OrderedDict({
    'mse': ('MSE', 4, min, 0.025, 0.037, 0.002),
    'psnr': ('PSNR', 2, max, 14.4, 16.2, 0.2),
    'ssim': ('SSIM', 4, max, 0.48, 0.69, 0.03),
    'bce': ('BCE', 4, min, 0.091, 0.134, 0.005),
    'wbce': ('WBCE', 4, min, 0.056, 0.09, 0.005)
 })

METRIC_NAMES_WBCE = OrderedDict({
    'mse': ('MSE', 4, min, 0.07, 0.101, 0.004),
    'psnr': ('PSNR', 2, max, 10.1, 11.8, 0.2),
    'ssim': ('SSIM', 4, max, 0.24, 0.54, 0.04),
    'bce': ('BCE', 4, min, 0.22, 0.33, 0.01),
    'wbce': ('WBCE', 4, min, 0.023, 0.029, 0.001)
 })

CNN_NAMES = OrderedDict({
    '0000': 'ResNet',
    '0001': 'UNet',
    '0002': 'DCGAN'
})

MODALITY_NAMES = OrderedDict({
    'rgb': ('RGB', 0),
    'depth': ('Depth', 1),
    'isg': ('InsSeg', 2),
    'ssg': ('SemSeg', 3),
    'ofl': ('OptFl', 4),
})


class MetricResult:

    def __init__(self, metric, value):
        self.metric = metric
        self.value = value
        self.best_loss = False
        self.best_modality = False
        self.best_cnn = False
        self.name = METRIC_NAMES[metric][0]
        self.round_nr = METRIC_NAMES[metric][1]

    def get_latex_string(self):
        string = round(self.value, self.round_nr)
        if self.best_loss:
            string = f'\\textit{{{string}}}'
        if self.best_modality:
            string = f'\\textbf{{{string}}}'
        if self.best_cnn:
            string = f'\\underline{{{string}}}'
        return string


class ExperimentResult:

    def __init__(self, path):
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
            path_list = path.split('/')
        self.path = path
        self.loss = path_list[-6]
        self.modality = path_list[-5]
        self.modality_name = MODALITY_NAMES[self.modality][0]
        self.modality_idx = MODALITY_NAMES[self.modality][1]
        self.cnn = CNN_NAMES[path_list[-3]]
        self.metrics = {metric: MetricResult(metric, value) for metric, value in data.items()}

    def __str__(self):
        return f'{self.loss}-{self.modality_name}-{self.cnn}'

    def __repr__(self):
        return f'{self.loss}-{self.modality_name}-{self.cnn}'

    def __getitem__(self, item):
        if item not in self.metrics:
            return None
        return self.metrics[item]

    def is_loss(self, loss_name):
        if loss_name is None:
            return True
        return loss_name == self.loss

    def is_modality(self, modality_name):
        if modality_name is None:
            return True
        return modality_name == self.modality_name

    def is_cnn(self, cnn_name):
        if cnn_name is None:
            return True
        return cnn_name == self.cnn

    def get_row(self, metrics, latex=False):
        name = self.modality_name
        cells = [name]
        for metric in metrics:
            metric_result = self[metric]
            cell = ''
            if metric_result is not None:
                if latex:
                    cell = metric_result.get_latex_string()
                else:
                    cell = metric_result.value
            cells.append(cell)
        return cells


class ExperimentResults:

    def __init__(self):
        self.results = []

    def get_sub_results(self, modality=None, cnn=None, loss=None):
        sub_results = [result for result in self.results if result.is_modality(modality)]
        sub_results = [result for result in sub_results if result.is_cnn(cnn)]
        sub_results = [result for result in sub_results if result.is_loss(loss)]
        return sub_results

    def get_best(self, metric, modality=None, cnn=None, loss=None):
        sub_results = self.get_sub_results(modality, cnn, loss)
        values = [result[metric].value for result in sub_results]
        best_fn = METRIC_NAMES[metric][2]
        best_index = values.index(best_fn(values))
        return sub_results[best_index]

    def mark_best(self, str, metrics, modalities=None, cnns=None, losses=None):
        for metric in metrics:
            if str == 'modality':
                for cnn in cnns:
                    for loss in losses:
                        best_result = self.get_best(metric, cnn=cnn, loss=loss)
                        best_result[metric].best_modality = True
            elif str == 'cnn':
                for modality in modalities:
                    for loss in losses:
                        best_result = self.get_best(metric, modality=modality, loss=loss)
                        best_result[metric].best_cnn = True
            elif str == 'loss':
                for modality in modalities:
                    for cnn in cnns:
                        best_result = self.get_best(metric, modality=modality, cnn=cnn)
                        best_result[metric].best_loss = True

    def add(self, result):
        self.results.append(result)

    def sort(self):
        self.results = sorted(self.results, key=lambda r: r.modality_idx)


def get_results(exp_dir):
    """
    Gathers all .../eval/metrics.yml, generated during evaluation with the plot.py.
    Outputs a CSV containing all metrics to exp_dir/all_metrics.csv

    Args:
        exp_dir: Directory where /eval/metrics.yml should be searched for

    Returns:
        all_metrics: an OrderedDict with all metrics
    """
    results = ExperimentResults()
    paths = glob(os.path.join(exp_dir, '**/eval/'), recursive=True)

    for path in paths:
        metrics_path = os.path.join(path, 'metrics.yml')
        if os.path.exists(metrics_path):
            result = ExperimentResult(metrics_path)
            results.add(result)
        else:
            print(f'No metrics.yml found in {path}')
    results.sort()
    return results


def write_to_csv(path, name, metrics, results, latex):
    file_name = name
    if latex:
        file_name += '.latex'
    file_name += '.csv'
    with open(os.path.join(path, file_name), 'w') as file:
        writer = csv.writer(file)
        header = [name] + [METRIC_NAMES[metric][0] for metric in metrics]
        writer.writerow(header)
        for result in results:
            row = result.get_row(metrics, latex=latex)
            writer.writerow(row)


def evaluate_metrics(exp_dir, resnet_only=False):
    wbce = True if 'wbce' in exp_dir else False
    resnet_only = True if 'only' in exp_dir or resnet_only else False
    out_dir = os.path.join(exp_dir, 'results')
    os.makedirs(out_dir, exist_ok=True)
    metrics = [metric for metric in METRIC_NAMES.keys()]
    results = get_results(exp_dir)
    cnns = ['ResNet'] if wbce or resnet_only else CNN_NAMES.values()
    modalities = [m[0] for m in MODALITY_NAMES.values()]
    losses = ['wbce'] if wbce else ['mse']
    results.mark_best('modality', metrics, cnns=cnns, losses=losses)
    print(metrics, modalities, losses)
    results.mark_best('cnn', metrics, modalities=modalities, losses=losses)

    for cnn in cnns:
        cnn_results = results.get_sub_results(cnn=cnn)
        write_to_csv(out_dir, cnn, metrics, cnn_results, latex=False)
        write_to_csv(out_dir, cnn, metrics, cnn_results, latex=True)

    for metric in metrics:
        plot(out_dir, metric, modalities, results, wbce, resnet_only)


def plot(path, metric, modalities, results, wbce=False, resnet_only=False):
    data = []
    for modailty in modalities:
        modality_results = [result for result in results.get_sub_results(modality=modailty)]
        resnet = -1
        unet = -1
        dcgan = -1
        for result in modality_results:
            if result.cnn == 'ResNet':
                resnet = result[metric].value
            elif result.cnn == 'UNet':
                unet = result[metric].value
            elif result.cnn == 'DCGAN':
                dcgan = result[metric].value
        data.append([resnet] if wbce or resnet_only else [resnet, unet, dcgan])
    data = np.array(data)

    plt.rcParams['svg.fonttype'] = 'none'
    labels = ['Depth', 'InsSeg', 'SemSeg', 'OptFl']
    f = 0.3 if wbce or resnet_only else 0.8
    x = np.arange(len(labels)) * f  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    m_name, _, _, m_bot, m_top, m_int = METRIC_NAMES_WBCE[metric] if wbce else METRIC_NAMES[metric]
    print(m_name, np.min(data), np.max(data))
    resnet_color = '#B85450'
    unet_color = '#D6B656'
    dcgan_color = '#6C8EBF'

    if wbce or resnet_only:
        ax.axhline(data[0][0], color=resnet_color, linestyle='-')
        ax.bar(x, data[1:, 0], width, label='ResNet', zorder=2, color=resnet_color)
    else:
        ax.axhline(data[0][0], color=resnet_color, linestyle='-')
        ax.axhline(data[0][1], color=unet_color, linestyle='-')
        ax.axhline(data[0][2], color=dcgan_color, linestyle='-')

        ax.bar(x - width, data[1:, 0], width, label='ResNet', zorder=2, color=resnet_color)
        ax.bar(x, data[1:, 1], width, label='UNet', zorder=2, color=unet_color)
        ax.bar(x + width, data[1:, 2], width, label='DCGAN', zorder=2, color=dcgan_color)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(METRIC_NAMES[metric][0])

    y_lims = [m_bot, m_top]
    ax.set_ylim(y_lims)
    ax.set_yticks(np.arange(*y_lims, m_int))

    # ax.set_xlim(-1.5*width)
    ax.set_xticks(x, labels)
    # ax.legend()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    w = 5 if wbce or resnet_only else 8
    fig.set_size_inches(w, 3)
    fig.tight_layout()

    fig.savefig(os.path.join(path, f'{metric}.svg'), format='svg', dpi=1200)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description.')
    parser.add_argument('directory', help='Directory where /eval/metrics.yml should be searched for.')
    parser.add_argument('--only', type=bool, default=False, help='ResNet only')
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        print(f"Directory does not exit: {args.directory}")
        raise ValueError

    evaluate_metrics(args.directory, args.only)

    print('- Finished Metrics -')
