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


def prepare_tex(df, hue, style, hue_order):
    sns.set(rc={'text.usetex': True}, style='whitegrid')
    lineplot = sns.lineplot(data=df, x='Episode', y='Score', ci=95, palette=PALETTE,
                            hue_order=hue_order, hue=hue, style=style)
    # lineplot.set_title(f'{sorted(list(df["Measurement"].unique()))}')
    return lineplot


def prepare_plt(df, hue, style, hue_order):
    print('Struggling to plot Figure using LaTeX - going back to normal.')
    plt.close('all')
    sns.set(rc={'text.usetex': False}, style='whitegrid')
    lineplot = sns.lineplot(data=df, x='Episode', y='Score', hue=hue, style=style,
                            ci=95, palette=PALETTE, hue_order=hue_order)
    # lineplot.set_title(f'{sorted(list(df["Measurement"].unique()))}')
    return lineplot


def prepare_plot(filepath, results_df, ext='png', hue='Measurement', style=None, use_tex=False):
    df = results_df.copy()
    df[hue] = df[hue].str.replace('_', '-')
    hue_order = sorted(list(df[hue].unique()))
    if use_tex:
        try:
            _ = prepare_tex(df, hue, style, hue_order)
            plot(filepath, ext=ext)  # plot raises errors not lineplot!
        except (FileNotFoundError, RuntimeError):
            _ = prepare_plt(df, hue, style, hue_order)
            plot(filepath, ext=ext)
    else:
        _ = prepare_plt(df, hue, style, hue_order)
        plot(filepath, ext=ext)
