import matplotlib.pyplot as plt
import numpy as np 
import pickle
from utils import *
import warnings
warnings.filterwarnings("ignore")
import scipy.stats as st
import matplotlib.font_manager as font_manager
from scipy.optimize import curve_fit

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return y

tss = np.arange(0, 0.8,0.05).tolist()
tss.append(0.125)
tss = np.array(sorted(tss))

reps = 18

x0s_H1 = {}
x0s_H2 = {}

modules = [#('S', '1', 'OF', ''),
           #('R', '1', 'OF', 'day1'),
           ('R', '3', 'OF', 'day1'),
           ('Q', '2', 'OF', ''),
           ('R', '2', 'OF', 'day1'),
           ('R', '2', 'OF', 'day2'),
           ('Q', '1', 'OF', ''),
           ('R', '1', 'OF', 'day2'),
           #('R', '3', 'OF', 'day2')
           ]

colors = [ 'maroon', 'darkgreen', 'lightcoral', 'darkblue', 'lime', 'dodgerblue']
      
font = font_manager.FontProperties(family='Times New Roman', size = 90)

score_mod = {}
score_mod_std = {}

delta_tc1 = {}
delta_tc2 = {}


with open('Gamma_jit.pkl', 'rb') as f:
    Ds = pickle.load(f)
    
with open('spacings.pkl', 'rb') as f:
      g_spacing = pickle.load(f)
      
with open('grid_score_jit.pkl', 'rb') as f:
      score = pickle.load(f)
    


for rat_name, mod_name, sess_name, day_name in modules:
    x0s_H1[str(rat_name)+''+str(mod_name)+'_'+str(day_name)] = []
    x0s_H2[str(rat_name)+''+str(mod_name)+'_'+str(day_name)] = []
    score_mod[str(rat_name)+''+str(mod_name)+'_'+str(day_name)] = []
    score_mod_std[str(rat_name)+''+str(mod_name)+'_'+str(day_name)] = []
    
    delta_tc1[str(rat_name)+''+str(mod_name)+'_'+str(day_name)] = []
    delta_tc2[str(rat_name)+''+str(mod_name)+'_'+str(day_name)] = []
 
    
# =============================================================================
# tss_gs = np.arange(0, 0.8,0.15).tolist()
# for rat_name, mod_name, sess_name, day_name in modules:
#     for ts in tss_gs:
#         if ts == tss_gs[0]:
#             fig=plt.figure(figsize=(16,16))
#             axis=fig.add_subplot(111)
#             axis.hist(score[str(ts)+'_'+str(rat_name)+''+str(mod_name)+'_'+str(day_name)])
#             plt.title(str(rat_name)+' '+str(mod_name)+' '+str(day_name), fontsize=70)
#             for label in axis.xaxis.get_majorticklabels():
#                 label.set_fontsize(50)
#             for label in axis.yaxis.get_majorticklabels():
#                 label.set_fontsize(50)
#             axis.set_xlabel('GS',fontsize=70)
#             plt.show()
# 
#         score_mod[str(rat_name)+''+str(mod_name)+'_'+str(day_name)].append(np.median(score[str(ts)+'_'+str(rat_name)+''+str(mod_name)+'_'+str(day_name)]))
#         #score_mod_std[str(rat_name)+''+str(mod_name)+'_'+str(day_name)].append(np.std(score[str(ts)+'_'+str(rat_name)+''+str(mod_name)+'_'+str(day_name)]))
#         score_mod_std[str(rat_name)+''+str(mod_name)+'_'+str(day_name)].append(st.t.interval(alpha=0.683, df=len(score[str(ts)+'_'+str(rat_name)+''+str(mod_name)+'_'+str(day_name)])-1, loc=np.median(score[str(ts)+'_'+str(rat_name)+''+str(mod_name)+'_'+str(day_name)]), scale=st.sem(score[str(ts)+'_'+str(rat_name)+''+str(mod_name)+'_'+str(day_name)])))
# 
# =============================================================================
      
tss_plot = np.tile(tss, reps)
tss_plot_fit = np.tile(np.arange(0.,0.8,0.01), reps)


for i,(rat_name, mod_name, sess_name, day_name) in enumerate([#('S', '1', 'OF', ''),
           #('R', '1', 'OF', 'day1'),
           ('R', '3', 'OF', 'day1'),
           ('Q', '2', 'OF', ''),
           ('R', '2', 'OF', 'day1'),
           ('R', '2', 'OF', 'day2'),
           ('Q', '1', 'OF', ''),
           ('R', '1', 'OF', 'day2'),
           #('R', '3', 'OF', 'day2'),
                                                ]):
    tss_plot = np.tile(tss, reps)
    fig=plt.figure(figsize=(16,16))
    axis=fig.add_subplot(111)
    
    y = Ds[str(rat_name)+''+str(mod_name)+'_'+str(day_name)][:,:,0].flatten()
    y_mean = np.mean(Ds[str(rat_name)+''+str(mod_name)+'_'+str(day_name)][:,:,0], axis=0)
    #print('DH1=',y_mean[0])
    y_std = np.std(Ds[str(rat_name)+''+str(mod_name)+'_'+str(day_name)][:,:,0], axis=0)
    p0 = [np.max(y), np.median(tss_plot),1,np.min(y)] # this is an mandatory initial guess
    popt, pcov = curve_fit(sigmoid, tss_plot, y, p0)
    #print(str(rat_name)+''+str(mod_name)+'_'+str(day_name)+'_H1 --->'+str(int(np.round(popt[1],decimals=3)*1000))+' ms')
    delta_tc1[str(rat_name)+''+str(mod_name)+'_'+str(day_name)].append(int(np.round(popt[1],decimals=3)*1000))
    #plt.scatter(tss_plot,y,color = colors[i], marker = 'h', s=100)
    axis.errorbar(tss, y_mean, yerr = y_std, ecolor = colors[i], marker = 'h', markersize = 20, elinewidth = 5, markeredgecolor = colors[i], markerfacecolor = colors[i], capsize = 15, capthick = 5,ls = '', label = r'$'+str(rat_name)+'^{'+str(mod_name)+'}_{'+str(int(np.round(np.nanmean(g_spacing[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]),decimals =0)))+'}$')
    plt.plot(tss_plot_fit[:len(tss[:-1])*5], sigmoid(tss_plot_fit[:len(tss[:-1])*5], popt[0],popt[1],popt[2],popt[3]), lw = 6, ls = 'dashed', color = 'black')
    
    plt.scatter(popt[1], sigmoid(popt[1], popt[0],popt[1],popt[2],popt[3]), marker = '*', s =4000, color = 'darkblue')
    #customize labels 
    plt.locator_params(axis='both', nbins=5) 
    labels = [item.get_text() for item in axis.get_xticklabels()]
    #labels = [str(int(ts*1000)) for ts in tss]
    labels = ['','0', '200', '400', '600']
    axis.set_xticklabels(labels)
    
    x0s_H1[str(rat_name)+''+str(mod_name)+'_'+str(day_name)].append(int(np.round(popt[1],decimals=3)*1000))
    #axis.set_title('inflection point is at '+str(int(np.round(popt[1],decimals=3)*1000))+' ms',fontsize=50)
    #hfont = {'family':'Times New Roman'}
    font2 = {'family':'Times New Roman','color':'black','size':15}
    axis.set_ylabel(r'$\mathdefault{\Gamma_{1}}$',fontdict=font2, fontsize=130, rotation = 0, labelpad =80)
    axis.set_xlabel(r'$\Delta t\; [ms]$',fontsize=62)
    for label in axis.xaxis.get_majorticklabels():
        label.set_fontsize(50)
        #label.set_fontname('courier')
    #for label in ax2.xaxis.get_majorticklabels():
     #   label.set_fontsize(40)
        #label.set_fontname('verdana')
    for label in axis.yaxis.get_majorticklabels():
        label.set_fontsize(50)
        #label.set_fontname('courier')
    #for label in ax2.yaxis.get_majorticklabels():
     #   label.set_fontsize(40)
        #label.set_fontname('verdana')
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    #plt.xscale('log')
    plt.ylim(-0.125,0.95)
    plt.xlim(-0.02, 0.78)
    #plt.yscale('log')
    plt.text(0.05,0., s= str(np.round(popt[0]*popt[2]/4, decimals =2)), bbox=dict(facecolor='none', edgecolor=colors[i], boxstyle='round'),fontsize = 70)
    fig.tight_layout()
    plt.legend(loc = 'upper right',prop= font)
    
    #plt.savefig('sigm_fit_H1_'+str(rat_name)+''+str(mod_name)+'_'+str(day_name)+'.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    fig=plt.figure(figsize=(16,16))
    axis=fig.add_subplot(111)
    
    y = Ds[str(rat_name)+''+str(mod_name)+'_'+str(day_name)][:,:,1].flatten()
    y_mean = np.mean(Ds[str(rat_name)+''+str(mod_name)+'_'+str(day_name)][:,:,1], axis=0)
    print('DH2=',y_mean[0])
    if str(rat_name)+str(mod_name) =='R3':
        y_temp =y.tolist()
        temp = tss_plot.tolist()
        for z in range(5):
            y_temp.append(0.1)
            temp.append(0.8)
        for z in range(5):
            y_temp.append(0.1)
            temp.append(0.9)
        y = y_temp
        tss_plot =  temp
        tss_plot =  temp
    
    y_std = np.std(Ds[str(rat_name)+''+str(mod_name)+'_'+str(day_name)][:,:,1], axis=0)
    p0 = [max(y), np.median(tss_plot),   50, min(y)] # this is an mandatory initial guess
    popt, pcov = curve_fit(sigmoid, tss_plot, y, p0)
    print(str(rat_name)+''+str(mod_name)+'_'+str(day_name)+'_H2 --->'+str(int(np.round(popt[1],decimals=3)*1000))+' ms')
    delta_tc2[str(rat_name)+''+str(mod_name)+'_'+str(day_name)].append(int(np.round(popt[1],decimals=3)*1000))
    axis.plot(tss_plot_fit[:len(tss[:-1])*5], sigmoid(tss_plot_fit[:len(tss[:-1])*5], popt[0],popt[1],popt[2],popt[3]), lw = 6, ls = 'dashed', color = 'black')
    #axis.plot(tss_plot, sigmoid(tss_plot, popt[0],popt[1],popt[2],popt[3]), lw = 6, ls = 'dashed', color = 'black')
    axis.errorbar(tss, y_mean, yerr = y_std, ecolor = colors[i], marker = 'h', markersize = 20, elinewidth = 5, markeredgecolor = colors[i], markerfacecolor = colors[i], capsize = 15, capthick = 5,ls = '', label = r'$'+str(rat_name)+'^{'+str(mod_name)+'}_{'+str(int(np.round(np.nanmean(g_spacing[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]),decimals =0)))+'}$')
    
    
    #axis.scatter(popt[1], sigmoid(popt[1], popt[0],popt[1],popt[2],popt[3]), marker = '*', s =4000, color = 'darkblue')
    #customize labels 
    labels = [item.get_text() for item in axis.get_xticklabels()]
    #labels = [str(int(ts*1000)) for ts in tss]
    labels = ['','0', '200', '400', '600']
    axis.set_xticklabels(labels)
    plt.locator_params(axis='both', nbins=5) 
    x0s_H2[str(rat_name)+''+str(mod_name)+'_'+str(day_name)].append(int(np.round(popt[1],decimals=3)*1000))
    #axis.set_title('inflection point is at '+str(int(np.round(popt[1],decimals=3)*1000))+' ms',fontsize=50)
    font2 = {'family':'Times New Roman','color':'black','size':15}
    axis.set_ylabel(r'$\mathdefault{\Gamma_{2}}$', fontdict=font2, fontsize = 130, rotation =0, labelpad = 80)
    axis.set_xlabel(r'$\Delta t\; [ms]$',fontsize=62)
    for label in axis.xaxis.get_majorticklabels():
        label.set_fontsize(50)
        #label.set_fontname('courier')
    #for label in ax2.xaxis.get_majorticklabels():
     #   label.set_fontsize(40)
        #label.set_fontname('verdana')
    for label in axis.yaxis.get_majorticklabels():
        label.set_fontsize(50)
        #label.set_fontname('courier')
    #for label in ax2.yaxis.get_majorticklabels():
     #   label.set_fontsize(40)
        #label.set_fontname('verdana')
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    plt.ylim(-0.125,0.95)
    plt.xlim(-0.02, 0.76)
    fig.tight_layout()
    
    plt.legend(loc = 'upper right',prop= font)
    #plt.savefig('sigm_fit_H2_'+str(rat_name)+''+str(mod_name)+'_'+str(day_name)+'.png', dpi=300, bbox_inches='tight')
    plt.show()
    
with open('delta_tc1.pkl', 'wb') as f:
      pickle.dump(delta_tc1, f)
with open('delta_tc2.pkl', 'wb') as f:
      pickle.dump(delta_tc2, f)