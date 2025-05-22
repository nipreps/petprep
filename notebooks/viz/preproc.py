
import os
import glob
import warnings; warnings.simplefilter('ignore')
from pathlib import Path

import matplotlib as mpl
mpl.use('pgf')
from matplotlib import pyplot as plt
from matplotlib import gridspec

import numpy as np
import nibabel as nb
import seaborn as sn
from nilearn.image import concat_imgs, mean_img
from nilearn import plotting

sn.set_style("whitegrid", {
    'ytick.major.size': 5,
    'xtick.major.size': 5,
})
sn.set_context("notebook", font_scale=1)

pgf_with_custom_preamble = {
    'ytick.major.size': 0,
    'xtick.major.size': 0,
    'font.sans-serif': ['HelveticaLTStd-Light'],
    'font.family': 'sans-serif', # use serif/main font for text elements
    'text.usetex': False,    # use inline math for ticks
}
mpl.rcParams.update(pgf_with_custom_preamble)


pgf_with_custom_preamble = {
#     'font.sans-serif': ['Helvetica Light'],
#     'font.family': 'sans-serif', # use serif/main font for text elements
    'text.usetex': True,    # use inline math for ticks
    'pgf.rcfonts': False,   # don't setup fonts from rc parameters
    'pgf.texsystem': 'xelatex',
    'verbose.level': 'debug-annoying',
    "pgf.preamble": [
#         r'\renewcommand{\sfdefault}{phv}',
#         r'\usepackage[scaled=.92]{helvet}',
        r"""\usepackage{fontspec}
\setsansfont{HelveticaLTStd-Light}[
Extension=.otf,
BoldFont=HelveticaLTStd-Bold,
ItalicFont=HelveticaLTStd-LightObl,
BoldItalicFont=HelveticaLTStd-BoldObl,
]
\setmainfont{HelveticaLTStd-Light}[
Extension=.otf,
BoldFont=HelveticaLTStd-Bold,
ItalicFont=HelveticaLTStd-LightObl,
BoldItalicFont=HelveticaLTStd-BoldObl,
]
\setmonofont{Inconsolata-dz}
""",
        r'\renewcommand\familydefault{\sfdefault}',
#         r'\setsansfont[Extension=.otf]{Helvetica-LightOblique}',
#         r'\setmainfont[Extension=.ttf]{DejaVuSansCondensed}',
#         r'\setmainfont[Extension=.otf]{FiraSans-Light}',
#         r'\setsansfont[Extension=.otf]{FiraSans-Light}',
    ]
}
mpl.rcParams.update(pgf_with_custom_preamble)


plt.clf()
fig = plt.gcf()
_ = fig.set_size_inches(15, 2 * 3.1)
gs = gridspec.GridSpec(2, 2, width_ratios=[6, 1], height_ratios=[1, 1],  hspace=0.0, wspace=0.03)

thres = 20
vmin = 50
vmax = 200

ax1 = plt.subplot(gs[0, :-1])

disp = plotting.plot_anat(str(fprep_std), display_mode='z',
                          cut_coords=[-15, -5, 10, 20, 40], cmap='cividis', threshold=thres, vmin=vmin, vmax=vmax,
                          axes=ax1)
disp.add_contours(str(ATLAS_HOME / '1mm_tpm_csf.nii.gz'), colors=['k'], levels=[0.8])
disp.add_contours(str(ATLAS_HOME / '1mm_tpm_wm.nii.gz'), colors=['w'], levels=[0.8], linewidths=[1], alpha=0.7)
disp.add_contours(str(ATLAS_HOME / '1mm_brainmask.nii.gz'), colors=['k'], levels=[0.8], linewidths=[3], alpha=.7)

ax2 = plt.subplot(gs[1, :-1])
disp = plotting.plot_anat(str(feat_std), display_mode='z',
                          cut_coords=[-15, -5, 10, 20, 40], cmap='cividis', threshold=thres, vmin=vmin, vmax=vmax,
                          axes=ax2)
disp.add_contours(str(ATLAS_HOME / '1mm_tpm_csf.nii.gz'), colors=['k'], levels=[0.8])
disp.add_contours(str(ATLAS_HOME / '1mm_tpm_wm.nii.gz'), colors=['w'], levels=[0.8], linewidths=[1], alpha=0.7)
disp.add_contours(str(ATLAS_HOME / '1mm_brainmask.nii.gz'), colors=['k'], levels=[0.8], linewidths=[3], alpha=.7)

ax1.annotate(
    'fMRIPrep',
    xy=(0., .5), xycoords='axes fraction', xytext=(-24, .0),
    textcoords='offset points', va='center', color='k', size=24,
    rotation=90)

ax2.annotate(
    r'\texttt{feat}',
    xy=(0., .5), xycoords='axes fraction', xytext=(-24, .0),
    textcoords='offset points', va='center', color='k', size=24,
    rotation=90)


inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[1, 15],
                                              subplot_spec=gs[:, 1], wspace=0.01)

ax3 = fig.add_subplot(inner_grid[0])
gradient = np.hstack((np.zeros((50,)), np.linspace(0, 1, 120), np.ones((130,))))[::-1]
gradient = np.vstack((gradient, gradient))
ax3.imshow(gradient.T, aspect='auto', cmap=plt.get_cmap('cividis'))
ax3.xaxis.set_ticklabels([])
ax3.xaxis.set_ticks([])
ax3.yaxis.set_ticklabels([])
ax3.yaxis.set_ticks([])

ax4 = fig.add_subplot(inner_grid[1])
sn.distplot(nb.load(str(fprep_std)).get_data()[mask2mm], label='fMRIPrep',
            vertical=True, ax=ax4, kde=False, norm_hist=True)
sn.distplot(nb.load(str(feat_std)).get_data()[mask2mm], label=r'\texttt{feat}', vertical=True,
            color='darkorange', ax=ax4, kde=False, norm_hist=True)

# plt.gca().set_ylim((0, 300))
plt.legend(prop={'size': 20}, edgecolor='none')

ax4.xaxis.set_ticklabels([])
ax4.xaxis.set_ticks([])
ax4.yaxis.set_ticklabels([])
ax4.yaxis.set_ticks([])

plt.axis('off')
ax3.axis('off')
ax4.axis('off')

plt.savefig(str(out_folder / 'fmriprep-feat-std.pdf'),
            format='pdf', bbox_inches='tight', pad_inches=0.2, dpi=300)

################################################## Plotting close-ups
plt.clf()
for i, coords in enumerate([-5, 10, 20]):
    disp = plotting.plot_anat(str(fprep_std), display_mode='z', cut_coords=[coords], cmap='cividis', threshold=30, vmin=50, vmax=170)
    f = plt.gcf().set_size_inches(10, 20)
    disp.add_contours(str(ATLAS_HOME / '1mm_tpm_csf.nii.gz'), colors=['k'], levels=[0.8])
    disp.add_contours(str(ATLAS_HOME / '1mm_tpm_wm.nii.gz'), colors=['w'], levels=[0.8], alpha=0.7)
    disp.add_contours(str(ATLAS_HOME / '1mm_brainmask.nii.gz'), colors=['k'], levels=[0.8], linewidths=[3], alpha=.7)
    plt.savefig(str(out_folder / ('fmriprep-std-closeup%03d.svg' % i)), format='svg', bbox_inches='tight')

for i, coords in enumerate([-5, 10, 20]):
    disp = plotting.plot_anat(str(feat_std), display_mode='z', cut_coords=[coords], cmap='viridis', threshold=30, vmin=50, vmax=170)
    f = plt.gcf().set_size_inches(10, 20)
    disp.add_contours(str(ATLAS_HOME / '1mm_tpm_csf.nii.gz'), colors=['k'], levels=[0.8])
    disp.add_contours(str(ATLAS_HOME / '1mm_tpm_wm.nii.gz'), colors=['w'], levels=[0.8], alpha=0.7)
    disp.add_contours(str(ATLAS_HOME / '1mm_brainmask.nii.gz'), colors=['k'], levels=[0.8], linewidths=[3], alpha=.7)
    plt.savefig(str(out_folder / ('feat-std-closeup%03d.svg' % i)), format='svg', bbox_inches='tight')



###################################################### Plotting 1st level

plt.clf()

sn.set_context("notebook", font_scale=1.3)
sn.set_style("whitegrid", {
    'xtick.major.size': 5,
})

cut_coords = [0, 15, 30]

df = pd.read_csv(str(basedir / 'smoothness.csv'))

fig = plt.gcf()
_ = fig.set_size_inches(15, 2 * 3.1)
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1],  hspace=0.0, wspace=0.6)

ax3 = plt.subplot(gs[0, :-1])
ax4 = plt.subplot(gs[1, :-1])
ax1 = plt.subplot(gs[0, -1])
ax2 = plt.subplot(gs[1, -1])

disp = plotting.plot_stat_map(str(basedir / 'acm_fpre.nii.gz'),
                              bg_img=atlas, threshold=0.25, display_mode='z',
                              cut_coords=cut_coords, vmax=0.8, alpha=0.8,
                              axes=ax1, colorbar=False)
cbar = fig.colorbar(ax1, ticks=[0, 0.275, 0.725, 1.0])
cbar.ax.set_yticklabels([-0.8, -0.25, 0.25, 0.8])
# print(list(fig.axes[-1].get_yticks()))
# fig.axes[-1].set_yticks([0, 0.275, 0.5, 0.725, 1.0])
# print(list(fig.axes[-1].get_yticks()))

plotting.plot_stat_map(str(basedir / 'acm_feat.nii.gz'),
                       bg_img=atlas, threshold=0.25, display_mode='z',
                       cut_coords=cut_coords, vmax=0.8, alpha=0.8,
                       axes=ax2, colorbar=False)

ax1.annotate(
    'fMRIPrep',
    xy=(0., .5), xycoords='axes fraction', xytext=(-40, .0),
    textcoords='offset points', va='center', color='k', size=24,
    rotation=90)

ax2.annotate(
    r'\texttt{feat}',
    xy=(0., .5), xycoords='axes fraction', xytext=(-40, .0),
    textcoords='offset points', va='center', color='k', size=24,
    rotation=90)



fmriprep_smooth = df[df.pipeline.str.contains('fmriprep')][['fwhm_pre', 'fwhm_post']]
feat_smooth = df[df.pipeline.str.contains('feat')][['fwhm_pre', 'fwhm_post']]

cols = palettable.tableau.ColorBlind_10.hex_colors
sn.distplot(fmriprep_smooth.fwhm_pre, color=cols[0], ax=ax3,
            axlabel='Smoothing', label='fMRIPrep')
sn.distplot(feat_smooth.fwhm_pre, color=cols[1], ax=ax3,
            axlabel='Smoothing', label=r'\texttt{feat}')

sn.distplot(fmriprep_smooth.fwhm_post, color=cols[0], ax=ax4,
            axlabel='Smoothing', label='fMRIPrep')
sn.distplot(feat_smooth.fwhm_post, color=cols[1], ax=ax4,
            axlabel=r'\noindent\parbox{6.8cm}{\centering\textbf{Estimated smoothness} full~width~half~maximum~(mm)}',
            label='feat')

ax4.set_xlim([3, 8])
ax4.set_xticks([3, 4, 5, 6 , 7, 8])
ax4.set_xticklabels([3, 4, 5, 6 , 7, 8])
ax4.grid(False)

sn.despine(offset=0, trim=True)

ax4.set_ylim([-3, 8])
ax4.set_yticks([])
ax4.spines['left'].set_visible(False)

ax3.set_ylabel(r'\noindent\parbox{4.8cm}{\centering\textbf{Before smoothing} fraction of images}')
ax3.yaxis.set_label_coords(-0.1, 0.4)
ax4.set_ylabel(r'\noindent\parbox{4.8cm}{\centering\textbf{After smoothing} fraction of images}')
ax4.yaxis.set_label_coords(-0.1, 0.5)

# ax4.spines['bottom'].set_position(('outward', 20))
ax4.invert_yaxis()
ax4.xaxis.set_label_position('top')
ax4.spines['bottom'].set_visible(False)
ax4.spines['top'].set_visible(True)
ax4.spines['top'].set_position(('data', 5))
ax4.xaxis.set_ticks_position('top')
ax4.tick_params(axis='x', pad=-30)
ax4.xaxis.set_label_coords(0.6, 0.85)


ax3.set_xlim([3, 8])
ax3.set_xticks([])
ax3.set_xticklabels([])
ax3.set_ylim([0, 11])
ax3.set_yticks([])
ax3.set_yticklabels([])
ax3.grid(False)
ax3.set_xlabel('')
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

legend = ax3.legend(ncol=2, loc='upper right', bbox_to_anchor=(1.0,0.8))
legend.get_frame().set_facecolor('w')
legend.get_frame().set_edgecolor('gray')

ax3.set_title('A', fontdict={'fontsize': 24}, loc='left');
ax1.set_title('B', fontdict={'fontsize': 24}, loc='left');

plt.savefig(str(out_folder / 'fmriprep-feat-1stlevel-2.pdf'),
            format='pdf', bbox_inches='tight', pad_inches=0.2, dpi=300)


####################################################################### 2nd level

%matplotlib inline
import pandas as pd

import os
import glob
import warnings; warnings.simplefilter('ignore')
from pathlib import Path

import matplotlib as mpl
mpl.use('pgf')
from matplotlib import pyplot as plt
from matplotlib import gridspec

import numpy as np
import nibabel as nb
import seaborn as sn
from nilearn import plotting

import pandas as pd
import palettable

pgf_with_custom_preamble = {
    'text.usetex': True,    # use inline math for ticks
    'pgf.rcfonts': False,   # don't setup fonts from rc parameters
    'pgf.texsystem': 'xelatex',
    'verbose.level': 'debug-annoying',
    "pgf.preamble": [
        r"""\usepackage{fontspec}
\setsansfont{HelveticaLTStd-Light}[
Extension=.otf,
BoldFont=HelveticaLTStd-Bold,
ItalicFont=HelveticaLTStd-LightObl,
BoldItalicFont=HelveticaLTStd-BoldObl,
]
\setmainfont{HelveticaLTStd-Light}[
Extension=.otf,
BoldFont=HelveticaLTStd-Bold,
ItalicFont=HelveticaLTStd-LightObl,
BoldItalicFont=HelveticaLTStd-BoldObl,
]
\setmonofont{Inconsolata-dz}
\renewcommand\familydefault{\sfdefault}
""",]
}
mpl.rcParams.update(pgf_with_custom_preamble)

basedir = Path('/oak/stanford/groups/russpold/data/ds000030/1.0.3/derivatives/fmriprep_vs_feat_2.0-oe')

# jokesdf = pd.read_csv(basedir.parent.parent / 'fmriprep_vs_feat' / 'results.csv', index_col=0)
jokesdf = pd.read_csv(basedir.parent / 'fmriprep_vs_feat_2.0-jd' / 'results.csv', index_col=0)
jokesdf.columns = ['IFG', 'PCG', 'STN', 'correlation', 'bdice', 'bdicemasked', 'experiment', 'fdice',
       'fdicemasked', 'pipeline', 'preSMA', 'N']
jokesdf.pipeline = jokesdf.pipeline.str.replace('fslfeat_5.0.10', 'feat')
jokesdf.pipeline = jokesdf.pipeline.str.replace('fmriprep_1.0.8', 'fmriprep')
jokesdf.N = jokesdf.N.astype('uint8')

basedir = Path('/oak/stanford/groups/russpold/data/ds000030/1.0.3/derivatives/fmriprep_vs_feat_2.0-oe')
dataframe = pd.read_csv(basedir / 'l2' / 'group.csv', index_col=0)

cols = palettable.tableau.ColorBlind_10.hex_colors
sn.set_style("whitegrid")

plt.clf()
fig = plt.figure(figsize=(20,20))
plt.subplot(3,3,1)
sn.boxplot(x="N", y="bdice", hue='pipeline', hue_order=['fmriprep', 'fslfeat'],
           data=dataframe, palette=cols, linewidth=0.6)
# sn.pointplot(x="N", y="bdice", hue="pipeline", data=dataframe,
#              capsize=.2, palette=cols, size=0.5, aspect=.75, dodge=0.5)
plt.ylabel("Binary Dice (OE)")
plt.xlabel("Sample size $N$")

plt.subplot(3,3,2)
# sn.stripplot(x="N", y="fdice", hue="pipeline", data=dataframe)
sn.boxplot(x="N", y="fdice", hue="pipeline", hue_order=['fmriprep', 'fslfeat'],
           data=dataframe, palette=cols, linewidth=.6)
plt.ylabel("Fuzzy Dice (OE)")
plt.xlabel("Sample size $N$")
plt.subplot(3,3,3)
sn.boxplot(x="N", y="correlation", hue="pipeline", hue_order=['fmriprep', 'fslfeat'],
           data=dataframe, palette=cols, linewidth=.6)
# sn.stripplot(x="N", y="correlation", hue="pipeline", data=dataframe)
plt.ylabel("Correlation (OE)")
plt.xlabel("Sample size $N$")


basedir = Path('/oak/stanford/groups/russpold/data/ds000030/1.0.3/derivatives/fmriprep_vs_feat_2.0-jd')
jd1 = pd.read_csv(basedir / 'l2' / 'group.csv', index_col=0)
plt.subplot(3,3,4)
sn.boxplot(x="N", y="bdice", hue='pipeline', hue_order=['fmriprep', 'fslfeat'],
           data=jd1, palette=cols, linewidth=0.6)
# sn.pointplot(x="N", y="bdice", hue="pipeline", data=dataframe,
#              capsize=.2, palette=cols, size=0.5, aspect=.75, dodge=0.5)
plt.ylabel("Binary Dice (JD+OE)")
plt.xlabel("Sample size $N$")

plt.subplot(3,3,5)
# sn.stripplot(x="N", y="fdice", hue="pipeline", data=dataframe)
sn.boxplot(x="N", y="fdice", hue="pipeline", hue_order=['fmriprep', 'fslfeat'],
           data=jd1, palette=cols, linewidth=.6)
plt.ylabel("Fuzzy Dice (JD+OE)")
plt.xlabel("Sample size $N$")
plt.subplot(3,3,6)
sn.boxplot(x="N", y="correlation", hue="pipeline", hue_order=['fmriprep', 'fslfeat'],
           data=jd1, palette=cols, linewidth=.6)
# sn.stripplot(x="N", y="correlation", hue="pipeline", data=dataframe)
plt.ylabel("Correlation (JD+OE)")
plt.xlabel("Sample size $N$")



basedir = Path('/oak/stanford/groups/russpold/data/ds000030/1.0.3/derivatives/fmriprep_vs_feat_2.0-jd')
dataframe = pd.read_csv(basedir / 'l2-jd' / 'group.csv', index_col=0)
dataframe.columns = ['IFG', 'N', 'PCG', 'STN', 'correlation', 'bdice-old', 'bdice', 'fdice-old',
                     'fdice', 'pipeline', 'preSMA', 'repetition']
plt.subplot(3,3,7)
sn.boxplot(x="N", y="bdice", hue='pipeline', hue_order=['fmriprep', 'fslfeat'],
           data=dataframe, palette=cols, linewidth=0.6)
# sn.pointplot(x="N", y="bdice", hue="pipeline", data=dataframe,
#              capsize=.2, palette=cols, size=0.5, aspect=.75, dodge=0.5)
plt.ylabel("Binary Dice (JD+OE)")
plt.xlabel("Sample size $N$")

plt.subplot(3,3,8)
# sn.stripplot(x="N", y="fdice", hue="pipeline", data=dataframe)
sn.boxplot(x="N", y="fdice", hue="pipeline", hue_order=['fmriprep', 'fslfeat'],
           data=dataframe, palette=cols, linewidth=.6)
plt.ylabel("Fuzzy Dice (JD+OE)")
plt.xlabel("Sample size $N$")
plt.subplot(3,3,9)
sn.boxplot(x="N", y="correlation", hue="pipeline", hue_order=['fmriprep', 'fslfeat'],
           data=dataframe, palette=cols, linewidth=.6)
# sn.stripplot(x="N", y="correlation", hue="pipeline", data=dataframe)
plt.ylabel("Correlation (JD+OE)")
plt.xlabel("Sample size $N$")




# fprepdf = dataframe[dataframe.pipeline.str.startswith('fmriprep')]
# featdf = dataframe[dataframe.pipeline.str.contains('feat')]

# samplesizes = sorted(list(set(dataframe.N.values.ravel())))
# fprepstds = [fprepdf[fprepdf.N == s].fdice.mad() for s in samplesizes]
# featstds = [featdf[featdf.N == s].fdice.mad() for s in samplesizes]

# ax.scatter(x=samplesizes, y=fprepstds, color=cols[0])
# ax.scatter(x=samplesizes, y=featstds, color=cols[1])


plt.savefig(str(out_folder / 'fmriprep-feat-2stlevel-jd_vs_oe.pdf'),
            format='pdf', pad_inches=0.2, dpi=300)



cols = palettable.tableau.ColorBlind_10.hex_colors
sn.set_style("whitegrid")

plt.clf()
fig = plt.figure(figsize=(20,16))

plt.subplot(2,3,1)
# sn.stripplot(x="N", y="bdice", hue="pipeline", data=dataframe, jitter=0.3, alpha=0.5, cmap=cols, size=3)
sn.boxplot(x="N", y="bdice", hue="pipeline", data=jokesdf, palette=cols, linewidth=1)
plt.ylabel("Binary Dice (JD+OE)")
plt.xlabel("Sample size $N$")
plt.subplot(2,3,2)
# sn.stripplot(x="N", y="fdice", hue="pipeline", data=dataframe)
sn.boxplot(x="N", y="fdice", hue="pipeline", data=jokesdf, palette=cols, linewidth=1)
plt.ylabel("Fuzzy Dice (JD+OE)")
plt.xlabel("Sample size $N$")
plt.subplot(2,3,3)
sn.boxplot(x="N", y="correlation", hue="pipeline", data=jokesdf, palette=cols, linewidth=1)
# sn.stripplot(x="N", y="correlation", hue="pipeline", data=dataframe)
plt.ylabel("Correlation (JD+OE)")
plt.xlabel("Sample size $N$")

# dataframe = pd.read_csv(basedir.parent / 'fmriprep_vs_feat' / 'results.csv', index_col=0)
# dataframe.columns = ['IFG', 'PCG', 'STN', 'correlation', 'bdice', 'experiment', 'fdice',
#        'pipeline', 'preSMA', 'N']
# dataframe.pipeline = dataframe.pipeline.str.replace('fslfeat_5.0.9', 'feat')
# dataframe.pipeline = dataframe.pipeline.str.replace('fmriprep-1.0.3', 'fmriprep')
# dataframe.N = dataframe.N.astype('uint8')

plt.subplot(2,3,4)
# sn.stripplot(x="N", y="bdice", hue="pipeline", data=dataframe, jitter=0.3, alpha=0.5, cmap=cols, size=3)
sn.boxplot(x="N", y="bdice", hue='pipeline', data=dataframe, palette=cols, linewidth=1)
plt.ylabel("Binary Dice (OE)")
plt.xlabel("Sample size $N$")
plt.subplot(2,3,5)
# sn.stripplot(x="N", y="fdice", hue="pipeline", data=dataframe)
sn.boxplot(x="N", y="fdice", hue="pipeline", data=dataframe, palette=cols, linewidth=1)
plt.ylabel("Fuzzy Dice (OE)")
plt.xlabel("Sample size $N$")
plt.subplot(2,3,6)
sn.boxplot(x="N", y="correlation", hue="pipeline", data=dataframe, palette=cols, linewidth=1)
# sn.stripplot(x="N", y="correlation", hue="pipeline", data=dataframe)
plt.ylabel("Correlation (OE)")
plt.xlabel("Sample size $N$")
plt.savefig(str(out_folder / 'fmriprep-feat-2stlevel-jd_vs_oe.pdf'),
            format='pdf', pad_inches=0.2, dpi=300)


%matplotlib inline
cols = palettable.tableau.ColorBlind_10.hex_colors
sn.set_style("whitegrid")

plt.clf()
fig = plt.figure(figsize=(20,8))

plt.subplot(1,3,1)
# sn.stripplot(x="N", y="bdice", hue="pipeline", data=dataframe, jitter=0.3, alpha=0.5, cmap=cols, size=3)
sn.boxplot(x="N", y="bdice", hue="pipeline", data=dataframe, palette=cols, linewidth=1)
plt.ylabel("Binary Dice (OE)")
plt.xlabel("Sample size $N$")
plt.subplot(1,3,2)
# sn.stripplot(x="N", y="fdice", hue="pipeline", data=dataframe)
sn.boxplot(x="N", y="fdice", hue="pipeline", data=dataframe, palette=cols, linewidth=1)
plt.ylabel("Fuzzy Dice (OE)")
plt.xlabel("Sample size $N$")
plt.subplot(1,3,3)
sn.boxplot(x="N", y="correlation", hue="pipeline", data=dataframe, palette=cols, linewidth=1)
# sn.stripplot(x="N", y="correlation", hue="pipeline", data=dataframe)
plt.ylabel("Correlation (OE)")
plt.xlabel("Sample size $N$")



def read_rating(fname, rater=None):
    ds, sub = os.path.basename(os.path.splitext(fname)[0]).split('_')
    data = {'dataset': ds, 'subject': sub}
    if rater is not None:
        data['rater'] = rater

    with open(fname) as f:
        ratings = json.load(f)

    for reportlet in ratings['reports']:
        name = reportlet['name']

        if name == 'overall':
            data[name] = int(reportlet['rating'])
        elif '_T1w_' in name:
            data['t1_%s' % name.split('_T1w_')[-1]] = int(reportlet['rating'])
        elif '_bold_' in name:
            repname = 'bold_%s' % name.split('_bold_')[-1]
            data.setdefault(repname, []).append(int(reportlet['rating']))
        elif '_fieldmap_':
            repname = name.split('_fieldmap_')[-1]
            data.setdefault(repname, []).append(int(reportlet['rating']))
        else:
            print('Unsupported field name "%s"' % name)

    return data

def read_dataset(data_dir, fields=['overall', 't1_reconall', 't1_seg_brainmask', 't1_t1_2_mni',
                                   'bold_rois', 'bold_bbr', 'bold_syn_sdc']):
    dataset = [read_rating(f, rater='rater_1') for f in data_dir.glob('rater_1/*.json')]
    dataset += [read_rating(f, rater='rater_2') for f in data_dir.glob('rater_2/*.json')]
    dataset += [read_rating(f, rater='rater_3') for f in data_dir.glob('rater_3/*.json')]

    infields = list(set([a for g in dataset for a in g.keys()]))
    infields.remove('dataset')
    infields.remove('subject')
    infields.remove('rater')
    headers = ['dataset'] + infields

    failed = []
    unrated = []

    # Average
    dfs = []
    for i, d in enumerate(dataset):
        if 'bold_variant-hmcsdc_preproc' in d:
            d['bold_rois'] = list(d['bold_rois']) + list(d['bold_variant-hmcsdc_preproc'])
            del d['bold_variant-hmcsdc_preproc']
        for k, v in d.items():
            if k in ['dataset', 'subject', 'rater']:
                continue

            if isinstance(v, list):
                filtered = [vv for vv in v if int(vv) > 0]
                if filtered:
                    d[k] = float(np.average(filtered))
                else:
                    d[k] = np.nan
            else:
                v = float(v) if int(v) > 0 else np.nan

        dfs.append(pd.DataFrame(d, columns=headers, index=[i]))

    # Merge raters
    allraters = pd.concat(dfs).sort_values(by='dataset')
    allraters[infields] = allraters[infields].clip(0.0)

    averaged = []
    for ds in set(allraters.dataset.ravel().tolist()):
        d = {'dataset': ds.upper()}
        group = allraters[allraters.dataset.str.contains(ds)]
        groupavg = np.mean(group[headers[1:]].values, axis=0)
        d.update({k: v for k, v in zip(headers[1:], groupavg)})
        averaged.append(pd.DataFrame(d, columns=headers, index=pd.Index([d['dataset']])))
    dataframe = pd.concat(averaged).sort_values(by='dataset')
    dataframe.index.name = 'dataset'

    dataframe['bold_bbr'] = dataframe[['bold_bbr', 'bold_coreg']].mean(axis=1)
#     dataframe['bold_rois'] = dataframe[
#         ['bold_rois'] + ['bold_variant-hmcsdc_preproc'] if 'bold_variant-hmcsdc_preproc' in headers else []].mean(axis=1)
    # 'fmap_mask', 'bold_fmap_reg', 'bold_fmap_reg_vsm',
    return dataframe[fields].sort_values(by=fields, ascending=False)


# dataframe = pd.read_csv('fmriprep_qc.tsv', sep='\t')
# dataframe0 = dataframe[dataframe.version.str.contains('1.0.0')]
# dataframe = dataframe[dataframe.version.str.contains('1.0.7')]

dataframe = read_dataset(Path.home().joinpath('tmp/fmriprep-reports-2'))
dataframe0 = read_dataset(Path.home().joinpath('tmp/fmriprep-reports-1'))
dataframe0 = dataframe0.reindex(dataframe.index)
dataframe

dataframe0['qc'] = [1] * len(dataframe0)
dataframe['qc'] = [2] * len(dataframe)

# dataframe0.to_csv('fmriprep_1.0.0.tsv', sep='\t', index=None)
# dataframe.to_csv('fmriprep_1.0.7.tsv', sep='\t', index=None)

new = dataframe0.append(dataframe)
new['version'] = new.qc.map({1: '1.0.0', 2: '1.0.7'})
new['dataset'] = new.index.values
new[['dataset', 'version', 'overall', 't1_reconall', 't1_seg_brainmask', 't1_t1_2_mni', 'bold_rois', 'bold_bbr', 'bold_syn_sdc']].to_csv('fmriprep_qc.tsv', sep='\t', index=None)
