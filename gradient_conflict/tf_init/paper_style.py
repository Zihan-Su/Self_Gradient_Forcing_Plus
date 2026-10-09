import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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

def align_axes(fig):
    positions = [ax.get_position().frozen() for ax in fig.axes]
    fig.set_size_inches(6.875, 3.17)
    for ax, bounds in zip(fig.axes, positions):
        ax.set_position([bounds.x0, .73 / 3.17, bounds.width, 2.2265 / 3.17])
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=12.6, pad=10)

def restyle(fig,name):
    fig.canvas.draw()
    before=[ax.get_position().frozen() for ax in fig.axes]
    for t in fig.findobj(Text):
        t.set_fontsize(t.get_fontsize()*1.4)
    for ax in fig.axes:
        ax.tick_params(axis='both',labelsize=10.8*1.32*.85*1.15)
    if name.startswith('b_timestep'):
        for ax in fig.axes:
            ax.xaxis.label.set_fontsize(12*1.32)
            ax.yaxis.label.set_fontsize(12*1.32)
        ax=fig.axes[0];box=ax.get_position()
        ax.yaxis.set_label_coords(.023,box.y0+box.height/2,transform=fig.transFigure)
        ax.yaxis.label.set_verticalalignment('center')
    fig.canvas.draw()
    for ax,box in zip(fig.axes,before):
        assert np.allclose(ax.get_position().bounds,box.bounds),name

def format_panel(fig, name):
    fig.canvas.draw()
    height = fig.get_figheight()
    final_height = height + 8/72 + .46 + .19
    positions = [ax.get_position().frozen() for ax in fig.axes]
    for text in fig.findobj(Text):
        text.set_fontsize(text.get_fontsize() * 1.32)
    fig.set_size_inches(fig.get_figwidth(), final_height)
    tsne = name == 'a_tsne'
    for ax, box in zip(fig.axes, positions):
        x = box.x0 + (0 if tsne else .10 / fig.get_figwidth())
        y = (box.y0 * height + .46 - (.12 if tsne else .55)) / final_height
        h = (box.height * height + (0 if tsne else .55)) / final_height
        ax.set_position([x, y, box.width, h])
        if tsne:
            ax.set_title(ax.get_title(), fontsize=ax.title.get_fontsize(),
                         fontfamily='DejaVu Serif', pad=10 + .12 * 72)
    restyle(fig, name)

def distributions(embeddings, timesteps):
    ts=timesteps;n=len(embeddings['attention'])//2
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 7.0,
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.875, 3.05))

    for ax in axes:
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

    for ax,fam in zip(fig.axes,['attention','ffn']):
        xy=embeddings[fam];ax.scatter(*xy[:n].T,color='#009E73',s=2,alpha=.65,lw=0,rasterized=False,zorder=2)
        ax.scatter(*xy[n:].T,c=np.tile(ts,128),cmap='YlOrRd',vmin=0,vmax=1000,s=2,alpha=.65,lw=0,rasterized=False,zorder=3)
        ax.dataLim.set_points(np.array([xy.min(axis=0),xy.max(axis=0)]));ax.autoscale_view()
    enlarge(fig)
    fig.subplots_adjust(left=.085,right=.992,top=.93,bottom=.20,wspace=.18)
    for ax,title in zip(fig.axes,('Attention','FFN')):
        ax.set_title(title,loc='center',fontsize=12.6,fontfamily='DejaVu Serif',fontweight='normal',pad=10)
    old=fig.get_figheight();positions=[ax.get_position().frozen() for ax in fig.axes];fig.set_size_inches(fig.get_figwidth(),old+.12)
    for ax,b in zip(fig.axes,positions):ax.set_position([b.x0,(b.y0*old+.12)/(old+.12),b.width,b.height*old/(old+.12)])
    format_panel(fig,'a_tsne')

    fig.legend(handles=[Line2D([],[],ls='',marker='o',color='#009E73',markersize=9,label=r'Context ($t_0=0$)')],loc='center',bbox_to_anchor=(.255,.118),fontsize=17,frameon=False,handletextpad=.2)
    cax=fig.add_axes([.57,.102,.38,.025]);cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1000),cmap='YlOrRd'),cax=cax,orientation='horizontal')
    cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face');cb.set_ticks([0,500,1000]);cb.ax.tick_params(labelsize=12,pad=2,length=1.5);cb.outline.set_linewidth(.4)
    fig.text(.76,.157,r'Denoising ($t$)',ha='center',va='center',fontsize=17)
    return fig

def means(data):
    plt.rcParams.update({'font.family':'DejaVu Serif','pdf.fonttype':42,'svg.fonttype':'path'})
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

    restyle(fig,'b_mean_gradient')
    return fig

def paired(values, timesteps):
    ts=timesteps
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.875, 3.35), sharey=True)
    for ax, family in zip(axes, ("attention", "ffn")):
        ax.set_xlabel("Denoising timestep")
        ax.set_title("Attention" if family == "attention" else "FFN")
        ax.set_ylim(-.85, .30)
        ax.grid(axis="y", color="#d9dde1", alpha=0.65, linewidth=0.55)
        ax.tick_params(length=3, width=0.7)
    axes[0].set_ylabel('Paired cosine')
    for fam,ax in zip(('attention','ffn'),fig.axes):
        v=values[fam]
        ax.scatter(np.tile(ts,128),v.ravel(),s=2,c=np.tile(ts,128),cmap='YlOrRd',vmin=0,vmax=1000,alpha=.27,lw=0,rasterized=False,zorder=2)
        lo,med,hi=np.quantile(v,[.25,.5,.75],axis=0);ax.fill_between(ts,lo,hi,color='#D55E00',alpha=.23,lw=0);ax.plot(ts,med,color='#D55E00',lw=1.5)
        ax.axhline(0,color='#6BAED6',lw=1.45,ls=(0,(5,3)))
        ax.set_xlim(1000,0);ax.set_xticks([1000,500,0], [1000,500,0]);ax.set_yticks([-.8,-.4,0,.2]);ax.get_xticklabels()[-1].set_ha('right')
    fig.set_size_inches(6.875,2.6);enlarge(fig);fig.tight_layout(pad=.8,w_pad=1.5);align_axes(fig);format_panel(fig,'b_timestep_cosine')
    return fig
