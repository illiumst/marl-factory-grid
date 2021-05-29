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


def plot(filepath, ext='png', **kwargs):
    plt.rcParams.update(kwargs)

    plt.tight_layout()
    figure = plt.gcf()
    plt.show()
    figure.savefig(str(filepath), format=ext)


def prepare_plot(filepath, results_df, ext='png', tag=''):

    _ = sns.lineplot(data=results_df, x='Episode', y='Score', hue='Measurement', ci='sd')

    # %%
    sns.set_theme(palette=PALETTE, style='whitegrid')
    font_size = 16
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": False,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": font_size,
        "font.size": font_size,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": font_size - 2,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2
    }

    try:
        plot(filepath, ext=ext, tag=tag, **tex_fonts)
    except (FileNotFoundError, RuntimeError):
        tex_fonts['text.usetex'] = False
        plot(filepath, ext=ext, tag=tag, **tex_fonts)
    plt.show()
