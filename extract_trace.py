import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits

from astropy.modeling import models, fitting

import pyds9

def bad_pixels():
    pos = 1
    
    gr = 'R'

    #for pos in range(1,6):
    os.system('dfits Data/*slope.fits | fitsort NAXIS1 NAXIS2 | grep -e "-%s-%d-" -e "-%d-%s-" |grep -v "\-H\-" | awk \'{print $1}\' > file.list' %(gr, pos, pos, gr))
    
    files = open('file.list').readlines()
    data = []
    for file in files:
        print(file)
        im = pyfits.open(file[:-1])
        sh = im[0].data.shape
        data.append(im[0].data.flatten())
    
    data = np.vstack(data)
    high = np.percentile(data, 20, axis=0)
    lo = data.min(axis=0)
    
    rms = 1.48*astropy.stats.median_absolute_deviation(high)
    bad = (high > 10*rms) | (high < -5*rms)
    
    pyfits.writeto('badpix_pos%d_%s.fits' %(pos, gr), data=bad.reshape(sh)*1, clobber=True)

def get_centroids():
    filters = ['F090W', 'F115W','F140M', 'F158M', 'F150W', 'F200W']
    grisms = ['C','R']
    positions = range(1,6)
    
    centroids = {}
    
    xc = yc = 0
    for gr in grisms:
        for pos in positions:
            bp = pyfits.open('badpix_pos%d_%s.fits' %(pos, gr))[0].data > 0
            if (gr == 'C'):
                bp = bp.T
            
            for filter in filters:
                file = glob.glob('Data/NIST*-%s-%d-%s*slope.fits' %(filter[:-1], pos, gr))[0]
                d = pyfits.open(file)
                
                if gr == 'C':
                    direct = d[0].data.T
                else:
                    direct = d[0].data
                #
                ds9.frame(1)
                ds9.view(direct*(~bp))
                ds9.set('pan to %f %f' %(xc+1, yc+1))
                x = raw_input('Center on direct image: \n')
                
                x0, y0 = np.cast[float](ds9.get('pan image').split())-1
                yp, xp = np.indices(direct.shape)

                Rmax = 5
                R = np.sqrt((x0-xp)**2+(y0-yp)**2)
                ok = (R <= Rmax) & (~bp)

                norm = np.sum(direct[ok])
                xc = np.sum((xp*direct)[ok])/norm
                yc = np.sum((yp*direct)[ok])/norm
                ds9.set('pan to %f %f' %(xc+1, yc+1))
                
                label = '%s-%d-%s' %(filter, pos, gr)
                print(label, xc, yc)
                
                centroids[label] = [xc, yc]
     
    trace_fits = collections.OrderedDict()
    for gr in grisms:
        for pos in positions:
            fig = plt.figure(figsize=[10,5])
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            
            import matplotlib
            cmap = matplotlib.cm.get_cmap('coolwarm')
            colors = {'F090W':0.05,  'F115W':0.3, 'F140M':0.8,  'F158M':0.4,  'F150W':0.7,  'F200W':0.95}
            
            for filter in filters:
                label = '%s-%d-%s' %(filter, pos, gr)
                data = extract(filter=filter, gr=gr, pos=pos)
                x, fit, xc, yc = data
                trace_fits[label] = data
                
                color = cmap(colors[filter])
                ax1.scatter(x-xc, np.abs(fit[:,2]), color=color, marker='.', alpha=0.8)
                ax2.scatter(x-xc, fit[:,1]-yc, color=color, label=label, marker='.', alpha=0.8)
            
            for ax in [ax1, ax2]:
                ax.set_xlim(-1800, 1500)
                ax.set_xlabel(r'$\delta x$')
                
            ax2.set_ylim(-15,18)
            ax1.set_ylim(0,2.5)
            ax2.legend(loc='lower right', fontsize=9)

            #ax1.set_ylabel('Lorentz FWHM')
            ax1.set_ylabel('MOFFAT FWHM')
            ax2.set_ylabel(r'$\delta y$')

            ax1.grid()
            ax2.grid()
            
            fig.tight_layout(pad=0.1)
            fig.savefig('moff_trace_%d_%s.pdf' %(pos, gr))
    
    fwcpos = {}
    xref = {}
    yref = {}
    
    for gr in grisms:
        for pos in positions:
            for filter in filters:
                label = '%s-%d-%s' %(filter, pos, gr)
                x, fit, xc, yc = trace_fits[label]
                
                if gr == 'R':
                    root = 'gr150r_%s_%d' %(filter.lower(), pos)
                else:
                    root = 'gr150c_%s_%d' %(filter.lower(), pos)
                
                file = '%s_orders.dat' %(root)
                
                g = pyfits.open(glob.glob('Data/NIST*-%s-%s-%d*slope.fits' %(filter[:-1], gr, pos))[0])
                d = pyfits.open(glob.glob('Data/NIST*-%s-%d-%s*slope.fits' %(filter[:-1], pos, gr))[0])
                
                head = g[0].header
                xg, yg = head['YACTPOS'], head['XACTPOS']
                head = d[0].header
                xd, yd = head['YACTPOS'], head['XACTPOS']

                fwcpos[file] = g[0].header['FWCPOS']
                if gr == 'R':
                    xref[file] = xg-xd
                    yref[file] = yg-yd
                else:
                    xref[file] = yg-yd
                    yref[file] = xg-xd
                                        
                fp = open(file,'w')
                fp.write('# %f 0.0 0.0\n' %(fwcpos[file]))
                np.savetxt(fp, np.array([x-xc, fit[:,1]-yc]).T)
                fp.close()
    
    gr, pos, filter = 'R', 1, 'F090W'
    center = np.load('centroids.npy')[0]
    
    label = '%s-%d-%s' %(filter, pos, gr)
    x, fit, xc, yc = trace_fits[label]
    x0, y0 = center[label]
    
    if gr == 'R':
        root = 'gr150r_%s_%d' %(filter.lower(), pos)
    else:
        root = 'gr150c_%s_%d' %(filter.lower(), pos)
    
    file = '%s_orders.dat' %(root)
    
    g = pyfits.open(glob.glob('Data/NIST*-%s-%s-%d*slope.fits' %(filter[:-1], gr, pos))[0])
    if gr == 'C':
        #direct = d[0].data.T
        grism = g[0].data.T
    else:
        #direct = d[0].data
        grism = g[0].data
    
    sw = np.loadtxt('wfss-trace/InitialTracePolynomials/gr150r_%s_all_orders.dat' %(filter.lower()))
    sw = np.loadtxt('wfss-trace/InitialTracePolynomials/gr150c_%s_all_orders.dat' %(filter.lower()))
    
    fig = plt.figure(figsize=[14,3])
    ax = fig.add_subplot(111)
    ax.imshow(unicorn.candels.clipLog(grism, scale=[-10,10000]), cmap='viridis_r', aspect='auto', origin='lower')
    ax.scatter(x, fit[:,1]+y0-yc, marker='+', color='r')
    ax.scatter(sw[:,0], sw[:,1]-1, marker='s', color='orange')
    
    ax.set_ylim(y0-10, y0+10)
    ax.set_xlim(x0-100, x0+300)
    fig.tight_layout()
    
    ### Compare centroids to source positions in header
    grism_offset = OrderedDict()
    grism_actpos = OrderedDict()
    
    for gr in grisms:
        for pos in [1,2,3,4,5]:
            for filter in filters:
                label = '%s-%d-%s' %(filter, pos, gr)
                label2 = 'gr150%s_%s_%d_orders.dat' %(gr.lower(), filter.lower(), pos)
                label2 = label
                
                x, fit, xc, yc = trace_fits[label]
                
                if gr == 'R':
                    root = 'gr150r_%s_%d' %(filter.lower(), pos)
                else:
                    root = 'gr150c_%s_%d' %(filter.lower(), pos)
                
                file = '%s_orders.dat' %(root)
                
                g = pyfits.open(glob.glob('Data/NIST*-%s-%s-%d*slope.fits' %(filter[:-1], gr, pos))[0])
                d = pyfits.open(glob.glob('Data/NIST*-%s-%d-%s*slope.fits' %(filter[:-1], pos, gr))[0])
                
                head = g[0].header
                xg, yg = head['YACTPOS'], head['XACTPOS']
                head = d[0].header
                xd, yd = head['YACTPOS'], head['XACTPOS']

                dx = xg-xd
                dy = yg-yd

                ci = center[label]
                if gr == 'C':
                    ci = center[label][::-1]
                    grism_offset[label2] = [dy, dx]
                    grism_actpos[label2] = [yg, xg]
                else:
                    grism_offset[label2] = [dx, dy]
                    grism_actpos[label2] = [xg, yg]
                    
                print('%s %.1f %.1f  %.1f %.1f  / %.1f %.1f  %.1f %.1f' %(label, ci[0], ci[1], ci[0]+dx, ci[1]+dy, xd-1, yd-1, xg-1, yg-1))
    
    np.save('grism_offset.npy', [grism_offset])
    np.save('grism_actpos.npy', [grism_actpos])
    
def show_traces(trace_scale=50, show_dms=True):
    
    if False:
        # dump to yaml
        import yaml
        def float_representer(dumper, value):
            text = '{0:.2f}'.format(value)
            return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)
        
        yaml.add_representer(float, float_representer)
        d_trace = {}
        
        for k in rot_trace:
            d_trace[k] = {'x':rot_trace[k][0].astype(np.float32).tolist(), 
                          'y':rot_trace[k][1].astype(np.float32).tolist()}
        
        with open('rot_trace.yml','w') as fp:
            yaml.dump(d_trace, fp)
        
        with open('grism_actpos.yml','w') as fp:
            yaml.dump(dict(grism_actpos), fp)
        
    cp = [(0.0, 0.4470588235294118, 0.6980392156862745),
          (0.0, 0.6196078431372549, 0.45098039215686275),
          (0.8352941176470589, 0.3686274509803922, 0.0),
          (0.8, 0.4745098039215686, 0.6549019607843137),
          (0.0, 0.4470588235294118, 0.6980392156862745),
          (0.0, 0.6196078431372549, 0.45098039215686275)]
    
    #for i in range(5):
    #    cp[i] = plt.cm.Spectral(i/3.)
    cp[0] = 'navy'
    cp[1] = 'steelblue'
    cp[2] = 'coral'
    cp[3] = 'darkred'
        
    #rot_trace = np.load('rot_trace.npy', encoding='latin1', allow_pickle=True)[0]
    
    #grism_actpos = np.load('grism_actpos.npy', encoding='latin1', allow_pickle=True)[0]
    import yaml
    with open('rot_trace.yml') as fp:
        rot_trace = yaml.load(fp, Loader=yaml.SafeLoader)
    with open('grism_actpos.yml') as fp:
        grism_actpos = yaml.load(fp, Loader=yaml.SafeLoader)
        
    # rename keys
    for k in list(grism_actpos.keys()):
        gi = k[5].upper()
        fi = k.split('_')[1].upper()
        ni = k.split('_')[2]
        newkey = '{0}-{1}-{2}'.format(fi, ni, gi)
        print(k, newkey)
        grism_actpos[newkey] = grism_actpos[k]
        
    gr = 'C'
    #fig = plt.figure(figsize=[12,6])
    fs = [6.5,3.46]
    fs = [6.5, 3.3]
    fs = [7, 3.1*7/6.5]
    #fig = plt.figure(figsize=fs)
    fig, axes = plt.subplots(1,2,figsize=fs, sharey=True, 
                            gridspec_kw={'width_ratios':[1,2300./3000]})
    
    marker = '.'
    
    for ig, gr in enumerate(['R', 'C']):
        #ax = fig.add_subplot(121+ig)
        #plt.axis('equal')
        ax = axes[ig]
        
        for pos in [1,2,3,4,5]:
            for i, filter in enumerate(['F090W', 'F115W', 'F150W', 'F200W']):
                if filter.endswith('M'):
                    continue
            
                label = '%s-%d-%s' %(filter, pos, gr)
                color = cp[i]

                tr = rot_trace[label]                                
                # from yaml file
                trace = [np.array(tr['x']), np.array(tr['y'])]
                
                y0 = np.interp(0, trace[0], trace[1])
                
                #trace_scale = 50
                if gr == 'C':
                    xy = grism_actpos[label]
                    
                    scatter = xy[1]-trace[1]*trace_scale, xy[0]+trace[0]
                    label = np.array([xy[1], xy[1]-y0*trace_scale]), np.array([xy[0], xy[0]])
                else:
                    xy = grism_actpos[label][::-1]
                    scatter = xy[1]+trace[0], xy[0]+trace[1]*trace_scale
                    label = np.array([xy[1], xy[1]]), np.array([xy[0], xy[0]+y0*trace_scale])
                
                # Correct to DMS coordinates
                if show_dms:
                    scatter = CV3_to_DMS(scatter)
                    label = CV3_to_DMS(label)
                    xy = CV3_to_DMS(xy)
                    
                    if True & (gr == 'C'):
                        # Flip curvature by hand to fix bug in DMS coords
                        scatter = np.array(scatter)
                        #print(scatter.shape)
                        dy = scatter[1] - xy[0]
                        scatter[1] = xy[0] - dy
                        
                        dy = label[1] - xy[0]
                        label = np.array(label)
                        label[1] = xy[0] - dy
                        
                ax.scatter(scatter[0], scatter[1], color=color, 
                           alpha=0.5, marker='.', s=12)
                           
                if '09' in filter:
                    ax.plot(label[0], label[1], color='k', alpha=0.4)
                    
                # if gr == 'C':
                #     ax.scatter(xy[1]-trace[1]*100, xy[0]+trace[0], color=color, alpha=0.5, marker='s', s=16)
                #     if '09' in filter:
                #         ax.plot([xy[1], xy[1]-y0*100], [xy[0], xy[0]], color='k', alpha=0.4)
                # else:
                #     xy = grism_actpos[label][::-1]
                #     ax.scatter(xy[1]+trace[0], xy[0]+trace[1]*50, color=color, alpha=0.5, marker='s', s=16)
                #     if '09' in filter:
                #         ax.plot([xy[1], xy[1]], [xy[0], xy[0]+y0*50], color='k', alpha=0.4)
            
                ax.scatter(xy[1], xy[0], color='k', marker='s', s=15, 
                           alpha=0.8, zorder=100)
                           
                if gr == 'R':
                    ax.text(xy[1]+30 - 70*(pos in [5,2]), 
                            xy[0]+60 + 20*(pos in [5,2]), 
                    '%d' %(pos), ha='left', va='bottom', fontsize=8)
                    
                    ax.text(830, 1200, r'$\lambda^{+1}$', 
                            rotation=0, fontsize=8, ha='center', 
                            color='0.4')
                    ax.text(810, 1100, r'$\rightarrow$', 
                            rotation=260, fontsize=8, ha='center', 
                            color='0.4')
                else:
                    ax.text(xy[1], xy[0]+60, '%d' %(pos), ha='center', va='bottom', fontsize=8)
                    ax.text(1040, 680, r'$\leftarrow \lambda^{+1}$', 
                            rotation=-5, fontsize=8, ha='left', color='0.4')
    
        ax.plot([0,2040,2040,0,0], [0,0,2040,2040,0], color='k', alpha=0.5, zorder=-100)
        ax.set_xlabel(r'$x$')
        if gr == 'R':
            ax.set_ylabel(r'$y$')
        else:
            ax.set_yticklabels([])
            
        #ax.set_title('GR150'+gr)        
        if gr == 'C':
            ax.set_xlim(-800, 2300)
            if show_dms:
                ax.set_xlim(-600, 2500) # DMS
                ax.set_xlim(-200, 2200) # DMS
            
        else:
            ax.set_xlim(-530, 2570)
            if show_dms:
                ax.set_xlim(-300, 2800) # DMS
            
        ax.set_ylim(-800,2300)
        if show_dms:
            ax.set_ylim(-450,2450) # DMS
        
        ax.set_aspect(1)
        ax.set_yticks(range(0,2050,512))
        ax.set_yticklabels([])
        ax.set_xticks(range(0,2050,512))
        ax.set_xticklabels([])
        
        if gr == 'R':
            ax.text(40, 1950, 'GR150'+gr, ha='left', va='top', fontsize=10)
        else:
            ax.text(40, 1700, 'GR150'+gr, ha='left', va='top', fontsize=10)
        
        if show_dms:
            ax.set_ylim(-450, 2150)
            
        if 0:
            ax.grid()
        else:
            ax.axis('off')
            ax.vlines([512,1024,1536], 0, 2048, linestyle=':', color='k', 
                      alpha=0.5, zorder=-1, linewidth=1)
            
            ax.hlines([512,1024,1536], 0, 2048, linestyle=':', color='k', 
                      alpha=0.5, zorder=-1, linewidth=1)
            
    
    axes[1].text(-40, 60, r'$y \rightarrow$', ha='right', va='bottom', 
                 rotation=90, fontsize=8)
    axes[1].text(-30, -40, r'$x \rightarrow$', ha='left', va='top', 
                              rotation=0, fontsize=8)
    
    for i, filter in enumerate(['F090W', 'F115W', 'F150W', 'F200W']):
        axes[0].text(0 + 380*(i % 4), -100-0*(i > 1), filter, ha='left', va='top', color=cp[i], size=9)
    
    fig.axes[0].text(0, -240, r'Cross dispersion $\times$ {0:.0f}'.format(trace_scale), ha='left', va='top', color='k', size=9)
    
    fig.tight_layout(pad=0.0, w_pad=0.0)
    #fig.savefig('full_trace_DMS_v2.pdf')
    return fig
    
def CV3_to_DMS(xy):
    """
    Translate CV3 to DMS:
    
    Rotate 90 degrees CW and flip X
    """
    newx = 2048 - xy[1]
    newy = 2048 - xy[0]

    return newx, newy
    
def extract(filter='F200W', gr='R', pos=2, bin=6):
    import scipy.ndimage as nd
    import glob
    import astropy.stats
    from scipy import polyval, polyfit
    
    from nis_trace import dms_tools
    
    #filter='F200W'; gr='R'; pos=2
    
    d = pyfits.open(glob.glob('Data/NIST*-%s-%d-%s*slope.fits' %(filter[:-1], pos, gr))[0])
    g = pyfits.open(glob.glob('Data/NIST*-%s-%s-%d*slope.fits' %(filter[:-1], gr, pos))[0])
    
    dms_tools.make_slope_flt(d.filename(), output='Data/trace-d_%s-%d-%s_flt.fits' %(filter, pos, gr), make_dms=True)
    dms_tools.make_slope_flt(g.filename(), output='Data/trace-g_%s-%s-%d_flt.fits' %(filter, gr, pos), make_dms=True)
    
    if gr == 'C':
        direct = d[0].data.T
        grism = g[0].data.T
    else:
        direct = d[0].data
        grism = g[0].data
        
    f = nd.median_filter(grism, size=3)
    #bp = ((grism/f > 20) & (grism > 20)) | (grism < -15) | ((grism/f < -20) & (grism > 20))
    bp = pyfits.open('badpix_pos%d_%s.fits' %(pos, gr))[0].data > 0
    if (gr == 'C'):
        bp = bp.T
        
    grism -= np.median(grism[~bp])
    
    ds9 = pyds9.DS9()
    
    # ds9.frame(1)
    # ds9.view(direct*(~bp))
    # x = raw_input('Center on direct image: \n')
    # 
    # x0, y0 = np.cast[float](ds9.get('pan image').split())-1
    # yp, xp = np.indices(direct.shape)
    # 
    # Rmax = 5
    # R = np.sqrt((x0-xp)**2+(y0-yp)**2)
    # ok = (R <= Rmax) & (~bp)
    # 
    # norm = np.sum(direct[ok])
    # xc = np.sum((xp*direct)[ok])/norm
    # yc = np.sum((yp*direct)[ok])/norm
    # ds9.set('pan to %f %f' %(xc, yc))
    center = np.load('centroids.npy')[0]
    label = '%s-%d-%s' %(filter, pos, gr)
    xc, yc = center[label]
    
    yi = int(yc)
    N = 25
    grism = grism[yi-N:yi+N,:]
    bp = bp[yi-N:yi+N,:]
    direct = direct[yi-N:yi+N,:]
    yc += N-yi
    
    ### Fill bad pixels
    med = nd.median_filter(grism*(~bp), size=3)
    grism[bp] = med[bp]
        
    ds9.frame(1)
    ds9.view(direct*(~bp))
    ds9.set('pan to %f %f' %(xc, yc))
    
    ds9.frame(2)
    ds9.view(grism*(~bp))
    ds9.set('pan to %f %f' %(xc+1, yc+1))
    
    rms = 1.48*astropy.stats.median_absolute_deviation(grism[~bp])
    
    ### Binning
    old_grism = grism*1
    kern = np.ones((1,bin), dtype=float)
    grism_bin = nd.convolve(grism, kern/bin)
    grism_bin = grism_bin[:,bin/2::bin]
    sh = grism_bin.shape

    med = np.median(grism_bin, axis=0)
    ytot = grism_bin.sum(axis=0) - med*sh[0]
    ymax = nd.maximum_filter(ytot, size=3)

    has_flux = (ytot > 2*rms*np.sqrt(sh[0])) & (ytot > 0.3*ymax)
    
    xarr = np.arange(old_grism.shape[1])[bin/2::bin]
    
    xarr = xarr[has_flux]
    grism = grism_bin[:,has_flux]
    sh = grism.shape
    
    ## Fit trace
    #prof = models.Lorentz1D(x_0=yc, fwhm=1., amplitude=200)
    prof = models.Moffat1D(x_0=yc, amplitude=200, alpha=0.5)

    #fitter = fitting.LevMarLSQFitter()
    fitter = fitting.SimplexLSQFitter()

    NP = prof.parameters.shape[0]
    fit_parameters = np.zeros((sh[1], NP))
    chi2 = np.zeros(sh[1])
        
    exit = chi2*0
    
    for i in range(sh[1]):
        ycol = grism[:,i]
        xcol = np.arange(ycol.shape[0])
        ok = (ycol > -3*rms) & (np.abs(xcol-yc) < 20)
        print(i, sh[1], ok.sum())
        if ok.sum() < 10:
            continue
        
        if gr == 'R':
            if pos in [3]:
                #prof.x_0 = yc + -2.5*(i-xc)/1000.+1
                c = [  4.47759354e-07,  -1.98989218e-03,   9.17132031e-01]
            elif pos in [4]:
                c = [  6.04226631e-07,  -2.56780327e-03,   5.79842663e-01]
            elif pos in [2]:
                #prof.x_0 = yc + 10.*(i-xc)/1000.+1
                c = [ -2.02789267e-06,   9.83248931e-03,   1.83525285e+00]
            elif pos in [5]:
                #prof.x_0 = yc + 6.*(i-xc)/500.+0
                c = [ -2.04956089e-06,   1.22338672e-02,   6.77623583e-01]
            else:
                #prof.x_0 = yc + 5*(i-xc)/1000.+1
                c = [ -6.88355931e-07,   4.57909177e-03,   9.59810484e-01]
            
            prof.x_0 = polyval(c, xarr[i]-xc)+yc

        else:
            if pos in [3]:
                c = [  4.46900735e-06,   4.87356233e-04,   2.14943287e+00]
            elif pos in [4]:
                c = [  1.60368931e-06,  -1.26773260e-03,   1.72656021e+00]
            elif pos in [2]:
                c = [  4.49324583e-06,  -4.90339750e-03,   2.25293293e+00]
            elif pos in [5]:
                c = [  1.69482778e-06,  -3.25596441e-03,   2.47758602e+00]
            else:
                c = [  3.37917572e-06,  -3.39615933e-03,   2.68204278e+00]
            
            prof.x_0 = polyval(c, xarr[i]-xc)+yc
                
        prof.amplitude = ycol[ok].max()
        yprof = fitter(prof, xcol[ok], ycol[ok], maxiter=1000)
        fit_parameters[i,:] = yprof.parameters
        yfit = yprof(xcol)
        var = (rms*g[0].header['EXPTIME'])**2 + yfit*g[0].header['EXPTIME']
        
        chi2[i] = np.sum((ycol-yfit)**2/var)
        exit[i] = fitter.fit_info['exit_mode']
    
    bad = (exit != 0) | (fit_parameters[:,2] <= 0) | (fit_parameters[:,2] > 1.9) #(np.abs(np.abs(fit_parameters[:,2])-1) > 0.9)
    return xarr[~bad], fit_parameters[~bad,:], xc, yc
    
    bad = (np.diff(chi2) > 5) | (fit_parameters[1:,0] < 3*rms) | (fit_parameters[1:,1] <= 0) | (np.abs(np.abs(fit_parameters[1:,2])-1) > 1) | np.abs(np.diff(fit_parameters[:,1]) > 0.5)
    
    xpix = np.arange(2048)[1:]
    # plt.plot(np.arange(2048)[1:], fit_parameters[1:,1], marker='.', alpha=0.5, label=filter)
    # 
    # plt.plot(np.arange(2048)[1:][~bad], fit_parameters[1:,1][~bad], marker='.', alpha=0.5, label=filter)
    
    #plt.plot(np.arange(2048)[1:], fit_parameters[1:,1], marker='.', alpha=0.5, label=filter)
    
    clip = np.abs(np.diff(xpix[~bad]) > 5) | (np.abs(np.diff(fit_parameters[1:,1][~bad])) > 0.4) | (np.abs(np.diff(fit_parameters[1:,2][~bad])) > 0.2)
    #plt.plot(np.arange(2048)[1:][~bad][1:][~clip] - xc, fit_parameters[1:,1][~bad][1:][~clip] - yc, marker='.', alpha=0.5, label=filter)
    
    return np.arange(2048)[1:][~bad][1:][~clip], fit_parameters[1:,:][~bad,:][1:,:][~clip,:], xc, yc
    
    c = polyfit(np.arange(2048)[1:][~bad][1:][~clip] - xc, fit_parameters[1:,1][~bad][1:][~clip] - yc, 2)
    plt.plot(np.arange(2048)-xc, polyval(c, np.arange(2048)-xc))
    print(c)
    