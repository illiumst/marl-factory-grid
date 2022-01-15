import seaborn as sns
import matplotlib as mpl
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
    ax = plt.gca()
    legends = [c for c in ax.get_children() if isinstance(c, mpl.legend.Legend)]

    if legends:
        figure.savefig(str(filepath), format=ext,  bbox_extra_artists=(*legends,), bbox_inches='tight')
    else:
        figure.savefig(str(filepath), format=ext)

    plt.show()
    plt.clf()


def prepare_tex(df, hue, style, hue_order):
    sns.set(rc={'text.usetex': True}, style='whitegrid')
    lineplot = sns.lineplot(data=df, x='Episode', y='Score', ci=95, palette=PALETTE,
                            hue_order=hue_order, hue=hue, style=style)
    lineplot.set_title(f'{sorted(list(df["Measurement"].unique()))}')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    return lineplot


def prepare_plt(df, hue, style, hue_order):
    print('Struggling to plot Figure using LaTeX - going back to normal.')
    plt.close('all')
    sns.set(rc={'text.usetex': False}, style='whitegrid')
    lineplot = sns.lineplot(data=df, x='Episode', y='Score', hue=hue, style=style,
                            ci=95, palette=PALETTE, hue_order=hue_order, )
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    # lineplot.set_title(f'{sorted(list(df["Measurement"].unique()))}')
    return lineplot


def prepare_center_double_column_legend(df, hue, style, hue_order):
    print('Struggling to plot Figure using LaTeX - going back to normal.')
    plt.close('all')
    sns.set(rc={'text.usetex': False}, style='whitegrid')
    fig = plt.figure(figsize=(10, 11))
    lineplot = sns.lineplot(data=df, x='Episode', y='Score', hue=hue, style=style,
                            ci=95, palette=PALETTE, hue_order=hue_order, legend=False)
    # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    lineplot.legend(hue_order, ncol=3, loc='lower center', title='Parameter Combinations', bbox_to_anchor=(0.5, -0.43))
    plt.tight_layout()
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
