"""Shared styling for gradient-conflict figures."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Arc
CONTEXT_COLOR = '#009E73'
GEN_COLORS = ['#8B1A1A', '#C43C2B', '#E87542', '#F2A65A']
GEN_MARKERS = ['s', '^', 'D', 'P']
NOMINAL = [1000, 750, 500, 250]
COLORS = ['#7f0000', '#b52b27', '#db6535', '#efa45f']
MARKERS = GEN_MARKERS
N_EXITS = 4
SEED = 28092026

def enlarge(fig):
    fig.canvas.draw()
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    for text in fig.findobj(Text):
        text.set_fontfamily('DejaVu Serif')
    for ax in fig.axes:
        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_fontsize(12)
        for title in (ax.title, ax._left_title, ax._right_title):
            title.set_fontsize(12.6)
        ax.tick_params(axis='both', labelsize=10.8)
        for tick in ax.get_xticklabels()+ax.get_yticklabels():
            tick.set_fontfamily('DejaVu Serif')
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(10.2)



def match_reference_axes(fig):
    positions = [ax.get_position().frozen() for ax in fig.axes]
    fig.set_size_inches(6.875, 3.17)
    for ax, bounds in zip(fig.axes, positions):
        ax.set_position([bounds.x0, .73 / 3.17, bounds.width, 2.2265 / 3.17])
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=12.6, pad=10)


def finish(fig, name):
    old_height = fig.get_figheight()
    new_height = old_height + 8 / 72
    positions = [ax.get_position().frozen() for ax in fig.axes]
    fig.set_size_inches(fig.get_figwidth(), new_height)
    for ax, bounds in zip(fig.axes, positions):
        ax.set_position([bounds.x0, bounds.y0 * old_height / new_height,
                         bounds.width, bounds.height * old_height / new_height])
    fig.canvas.draw()
    fixed_positions = [ax.get_position().frozen() for ax in fig.axes]
    for text in fig.findobj(Text):
        text.set_fontsize(text.get_fontsize() * 1.32)
    fig.canvas.draw()
    for ax, fixed in zip(fig.axes, fixed_positions):
        assert np.allclose(ax.get_position().bounds, fixed.bounds)
    text_height = fig.get_figheight()
    final_height = text_height + .46 + .19
    positions = [ax.get_position().frozen() for ax in fig.axes]
    fig.set_size_inches(fig.get_figwidth(), final_height)
    for ax, bounds in zip(fig.axes, positions):
        ax.set_position([bounds.x0 + (.10 / fig.get_figwidth() if name != 'distributions' else 0), (bounds.y0 * text_height + .46) / final_height,
                         bounds.width, bounds.height * text_height / final_height])
        if name == 'distributions':
            box = ax.get_position()
            shift_inches = .12
            ax.set_position([box.x0, box.y0 - shift_inches / final_height,
                             box.width, box.height])
            ax.set_title(ax.get_title(), fontsize=ax.title.get_fontsize(),
                         fontfamily='DejaVu Serif', pad=10 + shift_inches * 72)
        if name == 'paired':
            box = ax.get_position()
            extra = .55 / final_height
            ax.set_position([box.x0, box.y0 - extra, box.width, box.height + extra])
        if ax.get_ylabel():
            box = ax.get_position()
            ax.yaxis.set_label_coords(.025, box.y0 + box.height / 2, transform=fig.transFigure)
            ax.yaxis.label.set_verticalalignment('center')
    for ax in fig.axes:
        ax.tick_params(axis='both', labelsize=10.8 * 1.32 * .85)
    height = fig.get_figheight()
    final = 332 / 72
    positions = [ax.get_position().frozen() for ax in fig.axes]
    for legend in fig.legends:
        anchor = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        legend.set_bbox_to_anchor((anchor.x0, (anchor.y0*height + final-height)/final))
    fig.set_size_inches(fig.get_figwidth(), final)
    for ax, box in zip(fig.axes, positions):
        ax.set_position([box.x0, (box.y0*height + final-height)/final,
                         box.width, box.height*height/final])
    labels = {'distributions': '(a) Gradient distributions',
              'paired': '(c) Conflict across timesteps'}
    fig.text(.5, 12/332, labels[name], ha='center', fontsize=23.76, fontfamily='DejaVu Serif')
    return fig

def distributions(embeddings, count):
    exits = np.tile(np.arange(4), count//4)
    fig, axes = plt.subplots(1, 2, figsize=(6.875, 3.05))
    
    for ax, family in zip(axes, ("attention", "ffn")):
        coords = embeddings[family]
        ax.scatter(
            coords[:count, 0],
            coords[:count, 1],
            color=CONTEXT_COLOR,
            marker="o",
            s=7.5,
            alpha=0.74,
            edgecolors="white",
            linewidths=0.18,
            rasterized=True,
            zorder=2,
        )
        for exit_index, (color, marker) in enumerate(zip(GEN_COLORS, GEN_MARKERS)):
            points = coords[count:][exits == exit_index]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                color=color,
                marker=marker,
                s=9.5,
                alpha=0.78,
                edgecolors="white",
                linewidths=0.18,
                rasterized=True,
                zorder=3,
            )
    
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="both", which="major", labelsize=5.8, length=2.3,
                       width=0.55, pad=1.5, color="#333333")
        ax.grid(False)
        ax.margins(0.035)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_color("#333333")
    
    
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=CONTEXT_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.2,
            markersize=4.4,
            label=r"Context ($t_0$)",
        )
    ]
    legend_handles += [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.2,
            markersize=4.4,
            label=rf"Denoising ($t_{{{4 - index}}}$)",
        )
        for index, (color, marker) in enumerate(zip(GEN_COLORS, GEN_MARKERS))
    ]
    legend_handles = [legend_handles[0], *reversed(legend_handles[1:])]
    
    fig.subplots_adjust(left=0.065, right=0.992, top=0.93, bottom=0.20, wspace=0.18)
    
    enlarge(fig)
    handles = legend_handles
    fig.legend(handles=[handles[i] for i in (0,3,1,4,2)],
               loc='lower center', bbox_to_anchor=(.5,.002), ncol=3,
               frameon=False, fontsize=10.2, handletextpad=.25,
               columnspacing=.8, labelspacing=.35, borderaxespad=0)
    fig.subplots_adjust(left=.065, right=.992, top=.93, bottom=.20, wspace=.18)
    for ax, title in zip(fig.axes, ('Attention', 'FFN')):
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('', loc='left')
        ax.set_title(title, loc='center', fontsize=12.6, fontfamily='DejaVu Serif', fontweight='normal', pad=10)
    old_height = fig.get_figheight()
    extra_bottom = .12
    positions = [ax.get_position().frozen() for ax in fig.axes]
    fig.set_size_inches(fig.get_figwidth(), old_height + extra_bottom)
    for ax, bounds in zip(fig.axes, positions):
        ax.set_position([bounds.x0,
                         (bounds.y0 * old_height + extra_bottom) / (old_height + extra_bottom),
                         bounds.width, bounds.height * old_height / (old_height + extra_bottom)])
    return finish(fig, 'distributions')
    
def paired(values):
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 2, figsize=(6.875, 3.35), sharey=True)
    for ax, family in zip(axes, ("attention", "ffn")):
        ys = [values[family][:, exit_index] for exit_index in range(N_EXITS)]
        box = ax.boxplot(
            ys, positions=np.arange(N_EXITS), widths=0.5, patch_artist=True,
            showfliers=False, medianprops={"color": "#202830", "linewidth": 1.25},
            whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8},
        )
        for patch, color in zip(box["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.28)
        for index, (color, marker) in enumerate(zip(COLORS, MARKERS)):
            y = ys[index]
            x = index + rng.uniform(-0.16, 0.16, size=len(y))
            ax.scatter(x, y, s=10, color=color, marker=marker, alpha=0.62,
                       edgecolors="white", linewidths=0.22, zorder=3)
        ax.axhline(0, color="#0072B2", linestyle=(0, (5, 3)),
                   linewidth=1.45, zorder=1)
        ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1])
        ax.set_xticks(range(N_EXITS), NOMINAL)
        ax.set_xlabel("Denoising timestep")
        ax.set_title("Attention" if family == "attention" else "FFN")
        ax.set_ylim(-.55, .11)
        ax.grid(axis="y", color="#d9dde1", alpha=0.65, linewidth=0.55)
        for tick, gridline in zip(ax.get_yticks(), ax.get_ygridlines()):
            if np.isclose(tick, 0.0):
                gridline.set_visible(False)
        ax.tick_params(length=3, width=0.7)
    axes[0].set_ylabel('Paired cosine')
    fig.tight_layout(rect=[0, 0.01, 1, 0.99], w_pad=1.5)
    fig.set_size_inches(6.875, 2.6)
    enlarge(fig)
    fig.tight_layout(pad=.8, w_pad=1.5)
    match_reference_axes(fig)
    return finish(fig, 'paired')

def means(data):
    plt.rcParams.update({'font.family':'DejaVu Serif','pdf.fonttype':42,'svg.fonttype':'none'})
    W,H=6.875,3.931111
    fig=plt.figure(figsize=(W,H))
    blue,orange,ink='#0072B2','#D55E00','#243746'
    height=2.2265;width=height*1.83/1.66
    for center,f in zip((.28,.76),('attention','ffn')):
        ax=fig.add_axes([center-width/W/2,1.19/H,width/W,height/H])
        r=data['families'][f];c=np.array(r['context']);d=np.array(r['denoising'])
        ax.plot([0,0],[-.13,1.30],color='#A8A8A8',lw=.8)
        ax.axhline(0,color='#A8A8A8',lw=.8)
        for v,color in [(c,blue),(d,orange)]:
            ax.annotate('',xy=v,xytext=(0,0),arrowprops={'arrowstyle':'-|>','lw':2.5,'color':color,'mutation_scale':16})
        ax.add_patch(Arc((0,0),.65,.65,theta1=0,theta2=r['angle_degrees'],color=ink,lw=1.2))
        ax.text(.35,.29,f"{r['angle_degrees']:.1f}°",fontsize=12*1.32,color=ink)
        ax.set_title('Attention' if f=='attention' else 'FFN',fontsize=12.6*1.32,pad=10)
        ax.set_xlim(-.58,1.25);ax.set_ylim(-.13,1.53);ax.set_aspect('equal')
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
    fig.legend(handles=[Line2D([0],[0],color=blue,lw=2.5,label='Mean context gradient'),Line2D([0],[0],color=orange,lw=2.5,label='Mean denoising gradient')],loc='lower center',bbox_to_anchor=(.52,.018),ncol=1,frameon=False,fontsize=10.2*1.32,labelspacing=.35)
    height=fig.get_figheight(); final=332/72
    positions=[ax.get_position().frozen() for ax in fig.axes]
    for legend in fig.legends:
        legend.set_bbox_to_anchor((.52,(.018*height+final-height)/final))
    fig.set_size_inches(W,final)
    for ax,box in zip(fig.axes,positions):
        ax.set_position([box.x0,(box.y0*height+final-height)/final,box.width,box.height*height/final])
    fig.text(.5,12/332,'(b) Mean gradient directions',ha='center',fontsize=23.76,fontfamily='DejaVu Serif')
    return fig
