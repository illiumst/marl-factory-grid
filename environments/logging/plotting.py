import seaborn as sns
from matplotlib import pyplot as plt

PALETTE = 10 * (
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#e41a1c",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#888888",
    "#a6cee3",
    "#b2df8a",
    "#cab2d6",
    "#fb9a99",
    "#fdbf6f",
)


def plot(filepath, ext='png'):
    plt.tight_layout()
    figure = plt.gcf()
    figure.savefig(str(filepath), format=ext)
    plt.show()
    plt.clf()


def prepare_plot(filepath, results_df, ext='png'):
    results_df.Measurement = results_df.Measurement.str.replace('_', '-')
    hue_order = sorted(list(results_df.Measurement.unique()))
    try:
        sns.set(rc={'text.usetex': True}, style='whitegrid')
        sns.lineplot(data=results_df, x='Episode', y='Score', hue='Measurement',
                     ci=95, palette=PALETTE, hue_order=hue_order)
        plot(filepath, ext=ext)  # plot raises errors not lineplot!
    except (FileNotFoundError, RuntimeError):
        print('Struggling to plot Figure using LaTeX - going back to normal.')
        plt.close('all')
        sns.set(rc={'text.usetex': False}, style='whitegrid')
        sns.lineplot(data=results_df, x='Episode', y='Score', hue='Measurement',
                     ci=95, palette=PALETTE, hue_order=hue_order)
        plot(filepath, ext=ext)
