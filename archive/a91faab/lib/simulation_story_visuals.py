"""Figures for the draft-5 beamforming and contact studies.

Plots consume saved scientific outputs.  No simulation, posterior sampling,
normalizer evaluation, parameter fitting, or validation selection occurs here.
Real contact normals live in RP2; a complex-array sphere is always an explicitly
labelled density section rather than a marginal of its higher-dimensional law.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from .plotting import plt, save
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.colors import LogNorm

BLUE = '#0072B2'
ORANGE = '#D55E00'
GREEN = '#009E73'
PURPLE = '#A55AA3'
GREY = '#6C757D'
BLACK = '#202124'
PALE = '#E8EEF1'

METHOD_LABELS = {
    'mean': r'$\overline P_t^+$',
    'mode': r'$P_{\widehat{\mathcal X}_t}$',
    'isotropic': r'$\overline P(\kappa P_{\widehat{\mathcal X}_t})$',
    'iso': r'$\overline P(\kappa P_{\widehat{\mathcal X}_t})$',
    'axial': r'$\overline P(\kappa P_{\widehat{\mathcal X}_t})$',
    'scalar': 'Regularized PCA',
    'regularized': r'$\widetilde M_t$',
    'loaded': 'Loaded covariance',
    'scm': 'Loaded covariance',
    'lw': 'Ledoit--Wolf',
    'ledoit_wolf': 'Ledoit--Wolf',
    'grouse': 'GROUSE',
    'pca': 'PCA',
    'residual_pca': 'PCA of residuals',
    'ridge': 'Ridge regression',
    'ekf': 'Extended Kalman filter',
    'gaussian': 'Extended Kalman filter',
    'laplace': r'$\mathcal N(0,H_t^{-1})$',
    'mean_identity': r'$\overline P_t^+,\ G_t=I$',
    'mode_identity': r'$P_{\widehat{\mathcal X}_t},\ G_t=I$',
    'mode_same_A': r'$P_{\widehat{\mathcal X}_t}$',
    'oracle': 'Known covariance',
}
METHOD_COLORS = dict(mean=BLUE, mode=ORANGE, isotropic=PURPLE,iso=PURPLE,axial=PURPLE,
    scalar='#A6761D',loaded=GREEN,scm=GREEN,lw='#006B54', ledoit_wolf='#006B54',
    grouse='#7C573F', pca='#BD6B42', residual_pca='#7C573F', ridge=GREEN,
    regularized='#A6761D',ekf=GREEN,gaussian=GREEN,laplace='#8C6BB1',mean_identity=PURPLE,
    mode_identity='#BD6B42',mode_same_A=ORANGE,oracle=GREY)


def _finish(fig, path=None):
    if not plt.rcParams['text.usetex'] or fig.dpi != 200 or plt.rcParams['savefig.dpi'] != 200:
        raise AssertionError('The simulation figures require LaTeX and 200 dpi')
    if path is not None:
        save(fig, Path(path))
    return fig


def _clean(ax, axis='y'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis=axis, alpha=.15, linewidth=.6)


def _label(method):
    if str(method) not in METHOD_LABELS:raise ValueError(f'Define paper notation for method {method}')
    return METHOD_LABELS[str(method)]


def _color(method):
    return METHOD_COLORS.get(str(method), GREY)


def _select(frame, **selectors):
    result=frame
    for key,value in selectors.items():
        if key not in result:
            continue
        if isinstance(value,(float,int)):
            result=result[np.isclose(result[key].astype(float),value)]
        else:
            result=result[result[key].astype(str).eq(str(value))]
    if result.empty:
        raise ValueError(f'No saved rows for {selectors}')
    return result


def load_studies(root, profile='paper'):
    """Load all four components through the current redesign validation gate."""
    from .simulation_validation import require_gate,verify_component
    root=Path(root).resolve()
    if profile not in ('quick','paper'):
        raise ValueError('Expected quick or paper')
    base=root/'results/simulation_redesign'/profile
    data=dict(root=root,base=base,profile=profile,gate=require_gate(root))
    for component in ('geometry','beamforming','contact','scaling'):
        folder=base/component
        item=dict(folder=folder,metadata=verify_component(root,profile,component))
        for path in sorted(folder.glob('*.csv')):
            item[path.stem]=pd.read_csv(path)
        for path in sorted(folder.glob('*.json')):
            item[path.stem]=json.loads(path.read_text())
        for path in sorted(folder.glob('*.npz')):
            with np.load(path,allow_pickle=False) as record:
                if path.stem in item:item[path.stem+'_table']=item[path.stem]
                item[path.stem]={key:record[key] for key in record.files}
        data[component]=item
    return data


def paper_geometry(data, path=None, *, task='action'):
    """The real directional example: same mode and all distance quantiles."""
    from .distribution_geometry_visuals import (sphere_mesh,real_bingham_log_density,
        shared_density_norm,draw_density_sphere)
    record=data['matrices'] if 'matrices' in data else data
    A=np.asarray(record['A']);logZ=np.asarray(record['log_normalizer'])
    mean=np.asarray(record.get('mean_normal',record.get('mean')))
    mode=np.asarray(record['mode']);radius=np.asarray(record['radius'])
    hpd=np.asarray(record['hpd_loss']);mesh=sphere_mesh()
    logs=[real_bingham_log_density(a,z,mesh.nodes) for a,z in zip(A,logZ)]
    norm=shared_density_norm(logs,decades=4)
    fig=plt.figure(figsize=(15.8,5.2),layout='constrained')
    axes=[fig.add_subplot(1,3,i+1,projection='3d' if i<2 else None) for i in range(3)]
    surfaces=[]
    for i in range(2):
        surfaces.append(draw_density_sphere(axes[i],mesh,logs[i],norm=norm,
            mode=mode[i],A=A[i],hpd_loss=float(hpd[i]),ball_radius=float(radius[i]),
            title=rf'$q_{{A^{{({i+1})}}}}$',axis_labels=(r'$n_1$',r'$n_2$',r'$n_3$')))
    if task=='action':
        desired=np.asarray(record['soft_desired'])[1:]
        commands=np.asarray(record['soft_action_B'])[:,1:]
        axial=np.asarray(record['soft_action_axial'])[0,1:]
        values=[(desired,BLACK,'--',r'$v_t^{\rm des}$'),
            (axial,GREY,':',r'$q_{\kappa P_{\widehat{\mathcal X}}}$'),
            (commands[0],BLUE,'-',r'$v_t^{\rm B},\ q_{A^{(1)}}$'),
            (commands[1],ORANGE,'-',r'$v_t^{\rm B},\ q_{A^{(2)}}$')]
        for value,color,ls,label in values:
            axes[2].plot([0,value[0]],[0,value[1]],color=color,ls=ls,lw=1.8,label=label)
            axes[2].scatter(*value,color=color,s=35,zorder=5)
        axes[2].set(xlabel=r'$e_2^\top v$',ylabel=r'$e_3^\top v$',
            title=r'Same request, different $v_t^{\rm B}$',xlim=(0,.82),ylim=(0,.82))
        axes[2].set_aspect('equal',adjustable='box')
        axes[2].legend(frameon=False,fontsize=9,loc='upper left');_clean(axes[2])
    elif task=='risk':
        queries=np.eye(3)[1:]
        risks=np.array([[v@(m-m@m)@v for v in queries] for m in mean])
        x=np.arange(2)
        for i,color in enumerate((BLUE,ORANGE)):
            axes[2].bar(x+(i-.5)*.31,risks[i],width=.29,color=color,
                        label=rf'$q_{{A^{{({i+1})}}}}$')
        axes[2].set(xticks=x,xticklabels=[r'$e_2$',r'$e_3$'],ylabel=r'$R_t(v)$',
            title='The response risk depends on direction',ylim=(0,max(risks.ravel())*1.22))
        axes[2].legend(frameon=False);_clean(axes[2])
    else:raise ValueError('Expected action or risk')
    # draw_density_sphere returns its color-mapped artist; it may wrap it in a dict.
    mappable=surfaces[0].get('surface') if isinstance(surfaces[0],dict) else surfaces[0]
    fig.colorbar(mappable,ax=axes[:2],shrink=.64,pad=.01,label=r'$q_A([n])$')
    return _finish(fig,path)


def beam_setup(story,path=None):
    """Physical geometry, passive observations, and the covariance rank."""
    U=np.asarray(story['U']);n=U.shape[-1]
    s=float(np.asarray(story.get('s',.2)));tau2=float(np.asarray(story.get('tau2',1.)))
    desired=float(np.asarray(story.get('desired_angle_deg',0.)))
    interferer=float(np.asarray(story.get('angle_deg',10.)))
    fig,axes=plt.subplots(1,3,figsize=(15.8,4.6),layout='constrained')
    ax=axes[0];xs=np.arange(n)
    ax.plot(xs,np.zeros(n),color=GREY,lw=1.5)
    ax.scatter(xs,np.zeros(n),s=46,facecolors='white',edgecolors=BLACK,lw=1.6,zorder=3)
    centre=(n-1)/2
    for angle,color,label,offset in [(desired,GREEN,r'$a$',-.7),(interferer,ORANGE,r'$h$',.7)]:
        theta=np.deg2rad(angle);start=np.array([centre-2.3*np.sin(theta),2.3*np.cos(theta)])
        stop=np.array([centre-.45*np.sin(theta),.45*np.cos(theta)])
        ax.add_patch(FancyArrowPatch(start,stop,arrowstyle='-|>',mutation_scale=17,lw=2,color=color))
        ax.text(start[0]+offset,start[1]+.12,rf'{label}: ${angle:g}^\circ$',color=color,ha='center')
    ax.plot([centre,centre],[.1,2.7],ls=':',color=GREY,lw=1)
    ax.text(centre,-.53,r'$\lambda/2$ sensor spacing',ha='center',color=GREY)
    ax.set(xlim=(-.8,n-.2),ylim=(-.8,3.2),title='Desired signature and interferer')
    ax.axis('off')
    image=axes[1].imshow(abs(U[:min(len(U),16)]).T,origin='lower',aspect='auto',cmap='Greys',interpolation='nearest')
    axes[1].set(xlabel='Snapshot',ylabel='Sensor',title=r'Observed $|u_{tj}|$')
    fig.colorbar(image,ax=axes[1],shrink=.72,pad=.02,label='Magnitude')
    eigenvalues=np.r_[s+tau2,np.repeat(s,n-1)]
    axes[2].bar(np.arange(1,n+1),eigenvalues,color=[ORANGE]+[GREY]*(n-1))
    axes[2].axhline(s,color=GREY,ls=':',lw=1.2)
    axes[2].text(1.4,s+tau2,r'$s+\tau^2$',va='center')
    axes[2].set(xlabel='Eigenvalue index',ylabel='Variance',title=r'$sI+\tau^2P_{\mathcal X}$',
                xticks=[1,n],ylim=(0,(s+tau2)*1.2))
    _clean(axes[2])
    return _finish(fig,path)


def complex_posterior_section(story,path=None):
    """Real-coefficient section of a complex posterior plus its full spectrum."""
    from .distribution_geometry_visuals import (sphere_mesh,restriction_log_density,
        shared_density_norm,draw_density_sphere)
    A=np.asarray(story['A']);M=np.asarray(story['Pbar'])
    if A.ndim!=2:raise ValueError('Select one saved reporting time first')
    values,E=np.linalg.eigh(A);order=np.argsort(values)[::-1];values=values[order];E=E[:,order]
    U3=E[:,[0,1,len(E)-1]];reduced=(U3.conj().T@A@U3).real
    mesh=sphere_mesh();logZ=float(np.asarray(story['log_normalizer']))
    logs=restriction_log_density(A,U3,logZ,mesh.nodes)-(values[0]-logZ)
    norm=shared_density_norm([logs],decades=4)
    fig=plt.figure(figsize=(14.8,5.1),layout='constrained')
    ax=fig.add_subplot(1,3,1,projection='3d')
    surface=draw_density_sphere(ax,mesh,logs,norm=norm,mode=np.array([1.,0,0]),
        A=reduced,hpd_loss=story.get('hpd_loss'),ball_radius=story.get('radius'),
        title=r'$h=U_3x,\quad x\in S^2$',axis_labels=(r'$x_1$',r'$x_2$',r'$x_3$'))
    mappable=surface.get('surface') if isinstance(surface,dict) else surface
    fig.colorbar(mappable,ax=ax,shrink=.6,pad=.01,label=r'$q_A([U_3x])/q_A(\widehat{\mathcal X})$')
    ax=fig.add_subplot(1,3,2)
    gaps=values[0]-values[1:]
    ax.bar(np.arange(2,len(values)+1),gaps,color=GREY)
    ax.set(xlabel='Eigenvector index',ylabel=r'$\lambda_1(A)-\lambda_j(A)$',title='All complementary directions')
    _clean(ax)
    ax=fig.add_subplot(1,3,3)
    spectrum=np.diag(E.conj().T@M@E).real;n=len(M)
    rho=(spectrum[0]-1/n)/(1-1/n);axial=np.r_[spectrum[0],np.repeat((1-rho)/n,n-1)]
    x=np.arange(1,n+1)
    ax.plot(x,spectrum,'o-',color=BLUE,label=r'$\overline P_t^+$')
    ax.plot(x,axial,'s--',color=PURPLE,label=r'$\overline P(\kappa P_{\widehat{\mathcal X}_t})$')
    if np.any(spectrum<=0) or np.any(axial<=0):
        raise ValueError('The logarithmic mean spectrum requires positive saved eigenvalues')
    ax.set_yscale('log')
    ax.set(xlabel='Eigenvector index',ylabel='Mean eigenvalue',title='Directional uncertainty in the mean')
    ax.legend(frameon=False,fontsize=9);_clean(ax)
    return _finish(fig,path)


def paired_benchmark(metrics, *, endpoint, reference, selectors=None, path=None, title=None):
    """Paired trajectory scatter and gain distribution for one declared target."""
    frame=_select(metrics,**(selectors or {}))
    keys=[x for x in ['N','n','law','condition','angle_deg','s','t','trial'] if x in frame]
    left=frame[frame.method.eq(reference)][keys+[endpoint]].rename(columns={endpoint:'reference'})
    right=frame[frame.method.eq('mean')][keys+[endpoint]].rename(columns={endpoint:'mean'})
    paired=left.merge(right,on=keys,validate='one_to_one')
    finite=np.isfinite(paired.reference)&np.isfinite(paired['mean'])
    valid=paired[finite];unresolved=len(paired)-len(valid)
    if valid.empty:raise ValueError('No finite saved paired losses')
    gain=valid.reference-valid['mean']
    fig,axes=plt.subplots(1,2,figsize=(12.7,4.8),layout='constrained')
    lo=min(valid.reference.min(),valid['mean'].min());hi=max(valid.reference.max(),valid['mean'].max())
    pad=max(1e-10,(hi-lo)*.05)
    axes[0].scatter(valid.reference,valid['mean'],s=24,alpha=.5,color=BLUE,edgecolors='none')
    axes[0].plot([lo-pad,hi+pad],[lo-pad,hi+pad],ls='--',color=GREY,lw=1.2)
    axes[0].set(xlabel=_label(reference)+' loss',ylabel=r'$\overline P_t^+$ loss',
        xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),title='Each point is an independent trajectory')
    bins=np.histogram_bin_edges(gain,bins='fd')
    axes[1].hist(gain,bins=bins,color=BLUE,alpha=.75,edgecolor='white',linewidth=.5)
    axes[1].axvline(0,color=GREY,ls='--',lw=1.2)
    axes[1].set(xlabel=_label(reference)+r' loss $-\,\overline P_t^+$ loss',
        ylabel='Trajectories',title='Paired differences')
    axes[1].text(.97,.94,f'{len(valid)} pairs; {unresolved} unresolved',ha='right',va='top',
        transform=axes[1].transAxes,color=GREY,fontsize=9)
    for ax in axes:_clean(ax)
    if title:fig.suptitle(title)
    return _finish(fig,path)


def scaling_figure(data,path=None):
    """Measured primitive scaling, native moment costs, and exact state storage."""
    frame=data['summary'];memory=data['memory']
    if 'case' in frame:frame=frame[frame['case'].astype(str).isin(['span_80','fixed_span80','fixed_span_80'])]
    # Final runners use either a median column or a raw-repetition table.
    timing=next((x for x in ['wall_median','median_seconds','seconds_median','median','seconds'] if x in frame),None)
    if timing is None:raise ValueError('No saved timing summary column')
    fig,axes=plt.subplots(1,3,figsize=(15.8,4.8),layout='constrained')
    style={'conditional_real':(ORANGE,'s','Real'),'marginal_complex':(BLUE,'o','Complex')}
    for model,g in frame.groupby('model',sort=False):
        color,marker,name=style.get(model,(GREY,'o',str(model)))
        correction=g[g.operation.eq('correction')].sort_values('N')
        if not correction.empty:
            axes[0].loglog(correction.N,correction[timing]*1e6,marker=marker,color=color,label=name)
        for operation,ls in [('mode','--'),('mean','-')]:
            block=g[g.operation.eq(operation)].sort_values('N')
            if block.empty:continue
            label=(r'$P_{\widehat{\mathcal X}}$' if operation=='mode' else r'$\overline P(A)$')+f', {name.lower()}'
            axes[1].loglog(block.N,block[timing]*1e3,marker=marker,color=color,ls=ls,label=label)
            if 'resolved' in block:
                attempted=block.get('attempts',block.get('attempted',1))
                failed=block[block.resolved.astype(float)<attempted]
                axes[1].scatter(failed.N,failed[timing]*1e3,marker='x',s=60,color=BLACK,zorder=5)
    axes[0].set(xlabel=r'$N$',ylabel=r'Correction ($\mu$s)',title='One complete-vector correction')
    axes[1].set(xlabel=r'$N$',ylabel='Wall time (ms)',title='Eigenvectors and native moments')
    for model,g in memory.groupby('model',sort=False):
        color,marker,name=style.get(model,(GREY,'o',str(model)))
        field=next((x for x in ['parameter_bytes','A_bytes','dense_bytes'] if x in g),None)
        if field is None:raise ValueError('No exact stored-parameter byte count')
        q=g.drop_duplicates('N').sort_values('N')
        axes[2].loglog(q.N,q[field]/1024,marker=marker,color=color,label=name)
    axes[2].set(xlabel=r'$N$',ylabel='Stored parameter (KiB)',title=r'Dense $A$: $O(N^2)$ storage')
    for ax in axes:
        _clean(ax,axis='both');ax.legend(frameon=False,fontsize=8)
    return _finish(fig,path)


def beam_story(data, t=16, N=8):
    """Select the preassigned ULA10, s=.2 trajectory, never a favorable trial."""
    record=data['stories'];base=f'n{N}__';prefix=f'n{N}_t{t}__'
    out={key[len(base):]:value for key,value in record.items() if key.startswith(base)}
    out.update({key[len(prefix):]:value for key,value in record.items() if key.startswith(prefix)})
    if 'A' not in out:raise ValueError('No saved preassigned beam report')
    out.update(t=t,N=N,s=.2,tau2=1.,desired_angle_deg=0.,angle_deg=10.)
    out['U']=out['U'][:t]
    out['radius']=out.get('ball_radius')
    out['hpd_loss']=out.get('hpd_threshold')
    return out


def contact_static_story(data,kind='unbalanced',sigma=.01,t=8):
    record=data['story'];ik=list(record['static_kinds']).index(kind)
    iz=int(np.argmin(abs(np.asarray(record['static_sigmas'])-sigma)))
    times=np.asarray(record['static_times']);index=int(np.flatnonzero(times==t)[0])
    prefix=f'static_{ik}_{iz}_';out={}
    for name,value in record.items():
        if not name.startswith(prefix):continue
        key=name[len(prefix):]
        out[key]=value[:t] if key in ('U','V') else value[index]
    out.update(kind=kind,sigma=sigma,t=t,normal=record['static_true_normal'],queries=record['queries'])
    return out


def contact_dynamic_story(data,condition='sphere'):
    record=data['story'];ic=list(record['dynamic_conditions']).index(condition);prefix=f'dynamic_{ic}_'
    out={name[len(prefix):]:value for name,value in record.items() if name.startswith(prefix)}
    out.update(condition=condition,report_times=np.asarray(record['dynamic_report_times']),queries=record['queries'])
    return out


def contact_mechanics(story,path=None):
    """Physical contact and one measured force/velocity pair."""
    p=np.asarray(story['positions']);normal=np.asarray(story['world_normal'])[0]
    v=np.asarray(story['world_V'])[0];observed=np.asarray(story['world_U'])[0]
    clean=v-normal*np.dot(normal,v)
    tangent=clean/np.linalg.norm(clean) if np.linalg.norm(clean)>1e-12 else np.eye(3)[np.argmin(abs(normal))]
    tangent-=normal*np.dot(tangent,normal);tangent/=np.linalg.norm(tangent)
    fig=plt.figure(figsize=(12.9,5.2),layout='constrained')
    ax=fig.add_subplot(1,2,1,projection='3d')
    radius=float(np.linalg.norm(p[0]));theta,phi=np.meshgrid(np.linspace(0,np.pi,35),np.linspace(0,2*np.pi,61))
    ax.plot_surface(radius*np.sin(theta)*np.cos(phi)*1000,radius*np.sin(theta)*np.sin(phi)*1000,
        radius*np.cos(theta)*1000,color=PALE,alpha=.25,linewidth=0,shade=False)
    ax.plot(*(p*1000).T,color=BLACK,lw=2);ax.scatter(*(p[0]*1000),color=BLACK,s=25)
    edge=max(radius*1000,1.)
    ax.quiver(*(p[0]*1000),*(normal*edge*.35),color=ORANGE,linewidth=2,arrow_length_ratio=.2)
    across=np.cross(normal,tangent)
    pu,pv=np.meshgrid(np.linspace(-.22*edge,.22*edge,2),np.linspace(-.22*edge,.22*edge,2))
    patch=p[0]*1000+pu[...,None]*tangent+pv[...,None]*across
    ax.plot_surface(patch[...,0],patch[...,1],patch[...,2],color=BLUE,alpha=.23,shade=False)
    label=p[0]*1000+normal*edge*.44
    ax.text(*label,r'$[n_t]$',color=ORANGE,fontsize=10)
    label=p[0]*1000+tangent*edge*.3
    ax.text(*label,r'$\mathcal X_t$',color=BLUE,fontsize=10)
    ax.set(xlabel=r'$p_1$ (mm)',ylabel=r'$p_2$ (mm)',zlabel=r'$p_3$ (mm)',title='Maintained contact')
    ax.set_box_aspect((1,1,1));ax.view_init(elev=23,azim=-52)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.set_major_locator(plt.MaxNLocator(3));axis.pane.fill=False
    ax=fig.add_subplot(1,2,2)
    def coordinates(x):return 1000*np.array([tangent@x,normal@x])
    for value,color,label in [(v,BLUE,r'$v_t$'),(clean,ORANGE,r'$\bar u_t=P_{\mathcal X_t}v_t$')]:
        point=coordinates(value);ax.annotate('',xy=point,xytext=(0,0),arrowprops=dict(arrowstyle='-|>',lw=2.2,color=color))
        ax.plot([],[],color=color,label=label)
    point=coordinates(observed);ax.scatter(*point,s=45,facecolors='white',edgecolors=PURPLE,lw=1.6,label=r'$u_t$')
    a,b=coordinates(v),coordinates(clean);ax.plot([a[0],b[0]],[a[1],b[1]],color=GREY,ls=':',lw=1.5)
    bound=max(np.max(abs(np.r_[a,b,point])),1.)*1.25
    ax.set(xlim=(-bound*.25,bound),ylim=(-bound,bound),xlabel='Tangent component (mm/s)',
        ylabel='Normal component (mm/s)',title='Requested, allowed, measured')
    ax.axhline(0,color=GREY,lw=.8,alpha=.5);ax.set_aspect('equal',adjustable='box');ax.legend(frameon=False,fontsize=10)
    _clean(ax)
    return _finish(fig,path)


def contact_posterior(data,path=None,t=8):
    """Actual broad conditional posteriors, with expected Fisher directions."""
    from .distribution_geometry_visuals import (bingham_sphere_mesh,real_bingham_log_density,
        shared_density_norm,draw_density_sphere)
    stories=[contact_static_story(data,kind,.01,t) for kind in ('balanced','unbalanced')]
    meshes=[bingham_sphere_mesh(q['normal_parameter']) for q in stories]
    logs=[real_bingham_log_density(q['normal_parameter'],q['normal_logz'],mesh.nodes) for q,mesh in zip(stories,meshes)]
    norm=shared_density_norm(logs,decades=4)
    fig=plt.figure(figsize=(15.8,5.2),layout='constrained')
    axes=[fig.add_subplot(1,3,i+1,projection='3d' if i<2 else None) for i in range(3)]
    angle=np.deg2rad(-43.);R=np.array([[np.cos(angle),0,-np.sin(angle)],[0,1,0],[np.sin(angle),0,np.cos(angle)]])
    for i,(q,mesh,values) in enumerate(zip(stories,meshes,logs)):
        mapping=draw_density_sphere(axes[i],mesh,values,norm=norm,rotation=R,
            mode=q['normal_mode'],truth=q['normal'],A=q['normal_parameter'],hpd_loss=q['hpd_threshold'],
            ball_radius=q['ball_radius'],title='Cyclic axes' if i==0 else r'Requests near $e_1$',
            axis_labels=(r'$n_1$',r'$n_2$',r'$n_3$'))
    fig.colorbar(mapping,ax=axes[:2],shrink=.64,pad=.01,label=r'$q_{-A_t^+}([n])$',extend='min')
    x=np.arange(2)
    for i,(q,color) in enumerate(zip(stories,(BLUE,ORANGE))):
        axes[2].bar(x+(i-.5)*.3,q['fisher_truth_eigenvalues'],width=.28,color=color,
            label='Cyclic axes' if i==0 else r'Near $e_1$')
    axes[2].set(xticks=x,xticklabels=[r'$\lambda_{\min}$',r'$\lambda_{\max}$'],
        ylabel=r'Eigenvalue of $F_t$',title='Equal energy, different information')
    axes[2].legend(frameon=False);_clean(axes[2])
    return _finish(fig,path)


def contact_benchmarks(data,path=None):
    """Response and soft-action comparisons, paired by independent trajectory."""
    frame=_select(data['static_metrics'],kind='unbalanced',t=8)
    fig,axes=plt.subplots(1,3,figsize=(15.8,5.1),layout='constrained')
    targets=[('ridge','response_mse',1e6,'Ridge response loss',r'$\overline P_t^+v$ response loss'),
             ('isotropic','response_mse',1e6,r'$\overline P(\kappa P_{\widehat{\mathcal X}_t})v$ loss',r'$\overline P_t^+v$ loss'),
             ('mode','cost_10',1e6,r'$v_t^{\rm B}(P_{\widehat{\mathcal X}_t})$ cost',r'$v_t^{\rm B}$ cost')]
    mean=frame[frame.method.eq('mean')].set_index('trial')
    for ax,(reference,endpoint,scale,xlabel,ylabel) in zip(axes,targets):
        other=frame[frame.method.eq(reference)].set_index('trial');ids=mean.index.intersection(other.index)
        x=scale*other.loc[ids,endpoint].to_numpy();y=scale*mean.loc[ids,endpoint].to_numpy()
        finite=np.isfinite(x)&np.isfinite(y);x=x[finite];y=y[finite]
        low=max(0,min(x.min(),y.min())*.95);high=max(x.max(),y.max())*1.06
        ax.scatter(x,y,s=25,color=BLUE,alpha=.48,edgecolors='none')
        ax.plot([low,high],[low,high],color=GREY,ls='--',lw=1.2)
        units=r'$[(\mathrm{mm}/\mathrm{s})^2]$'
        ax.set(xlabel=xlabel+'\n'+units,ylabel=ylabel+'\n'+units,xlim=(low,high),ylim=(low,high))
        ax.text(.04,.96,f'{len(x)} independent pairs',transform=ax.transAxes,va='top',color=GREY,fontsize=9)
        _clean(ax)
    axes[0].set_title('A strong unconstrained reference')
    axes[1].set_title(r'Same mode and $r_{t,\mathrm{rms}}$')
    axes[2].set_title(r'The decision loss, $\lambda=10$')
    return _finish(fig,path)


def contact_tracking(data,path=None):
    """True and estimated normal geometry, plus current-state response densities."""
    from .distribution_geometry_visuals import sphere_mesh,hpd_curves
    story=contact_dynamic_story(data);p=np.asarray(story['positions']);R=np.asarray(story['R'])
    truth=np.asarray(story['world_normal']);estimated=np.einsum('tij,tj->ti',R,story['mode_normal'])
    # Sign alignment is display/scoring only, not an estimator input.
    estimated*=np.where(np.sum(estimated*truth,axis=1)<0,-1.,1.)[:,None]
    fig=plt.figure(figsize=(15.6,5.2),layout='constrained')
    ax=fig.add_subplot(1,3,1,projection='3d');mesh=sphere_mesh(33,65);X=mesh.nodes
    ax.plot_wireframe(X[...,0],X[...,1],X[...,2],rcount=9,ccount=17,color=GREY,alpha=.12,lw=.5)
    ax.plot(*truth.T,color=BLACK,lw=2,label=r'$[n_t]$')
    ax.plot(*estimated.T,color=ORANGE,lw=1.4,label=r'$[\widehat n_t]$')
    times=np.asarray(story['report_times']);last=len(truth)-1
    for t in times:
        j=int(t)-1;idx=int(np.flatnonzero(times==t)[0]);C=-story['A'][j]
        worldC=R[j]@C@R[j].T
        for curve in hpd_curves(worldC,float(story['report_hpd_threshold'][idx])):
            centre=estimated[j]
            if np.nanmean(curve@centre)<0:continue
            ax.plot(*curve.T,color=PURPLE,alpha=.55,lw=1.2)
        ax.scatter(*estimated[j],color=ORANGE,s=18)
    ax.set(title='Normal lines along the contact',xlabel=r'$n_1$',ylabel=r'$n_2$')
    # Keep the third coordinate label inside this panel, away from the next ylabel.
    ax.text2D(.85,.52,r'$n_3$',transform=ax.transAxes,color=GREY,fontsize=10)
    ax.set_box_aspect((1,1,1));ax.view_init(elev=23,azim=-52);ax.legend(frameon=False,fontsize=9,loc='upper left')
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.set_major_locator(plt.MaxNLocator(3));axis.pane.fill=False
    ax=fig.add_subplot(1,3,2)
    metrics=_select(data['dynamic_metrics'],condition='sphere')
    for method,color,marker in [('mean',BLUE,'o'),('mean_identity',PURPLE,'s'),('gaussian',GREEN,'^')]:
        g=metrics[metrics.method.eq(method)].groupby('t').normal_chordal_error.mean()
        label={'mean':r'$q_{A_t^+}$','mean_identity':r'$q_{A_t^+},\ G_t=I$'}.get(method,_label(method))
        ax.plot(g.index,g.values,marker=marker,color=color,label=label)
    ax.set(xlabel='Observation',ylabel=r'Mean $d_c(\widehat{\mathcal X}_t,\mathcal X_t)$',title='Measured frame transport matters')
    ax.legend(frameon=False,fontsize=8);_clean(ax)
    ax=fig.add_subplot(1,3,3)
    query=np.asarray(story['queries'])[0];samples=np.asarray(story['report_normal_samples'])
    values=[]
    for nodes in samples:
        response=query[None,:]-nodes*(nodes@query)[:,None]
        values.append(response[:,2]*1000)
    violin=ax.violinplot(values,positions=np.arange(len(times)),widths=.75,showextrema=False)
    for body in violin['bodies']:body.set_facecolor(BLUE);body.set_edgecolor(BLUE);body.set_alpha(.28)
    mean=np.asarray(story['report_response_mean'])[:,0,2]*1000
    true=np.asarray(story['report_response_truth'])[:,0,2]*1000
    lo=np.asarray(story['report_lower95'])[:,0,2]*1000;hi=np.asarray(story['report_upper95'])[:,0,2]*1000
    lo50=np.asarray(story['report_lower50'])[:,0,2]*1000;hi50=np.asarray(story['report_upper50'])[:,0,2]*1000
    x=np.arange(len(times));ax.vlines(x,lo,hi,color=BLUE,lw=1.2);ax.vlines(x,lo50,hi50,color=BLUE,lw=4)
    ax.scatter(x,mean,color=BLUE,s=22,label=r'$\overline P_t^+v^\star$')
    ax.scatter(x,true,color=BLACK,marker='x',s=38,label='True clean response')
    ax.set(xticks=x,xticklabels=[str(t) for t in times],xlabel='Observation',ylabel='Velocity component (mm/s)',
        title=r'Current response to $v^\star=v_0e_1$')
    ax.legend(frameon=False,fontsize=8);_clean(ax)
    return _finish(fig,path)


def higher_contact(data,scaling,path=None):
    """Avoid the artificial 1/N gain from averaging isotropic query loss."""
    frame=data['higher_dimension'];fig,axes=plt.subplots(1,2,figsize=(12.8,4.8),layout='constrained')
    dimensions=sorted(frame.n.unique());methods=['mean','mode','residual_pca'];offset=np.linspace(-.24,.24,len(methods))
    for method,delta in zip(methods,offset):
        block=frame[frame.method.eq(method)]
        values=[block[block.n.eq(n)].normalized_response_mse.dropna().to_numpy() for n in dimensions]
        box=axes[0].boxplot(values,positions=np.arange(len(dimensions))+delta,widths=.2,patch_artist=True,
            showfliers=True,flierprops=dict(marker='o',markersize=3,markerfacecolor=_color(method),markeredgecolor='none',alpha=.5),
            medianprops=dict(color='white',lw=1.3),whiskerprops=dict(color=_color(method)),
            capprops=dict(color=_color(method)),boxprops=dict(facecolor=_color(method),edgecolor=_color(method)))
        axes[0].plot([],[],color=_color(method),lw=5,label=_label(method))
    axes[0].set(xticks=np.arange(len(dimensions)),xticklabels=[str(n) for n in dimensions],xlabel=r'$N$',
        ylabel=r'$\|D-P_{\mathcal X}\|_F^2$',title=r'One constraint, $t=4N$ observations')
    axes[0].legend(frameon=False,fontsize=9);_clean(axes[0])
    times=scaling['summary'];times=times[(times.model=='conditional_real')&(times['case']=='data_T64')]
    for method,operation,color in [('mode','endpoint_mode',ORANGE),('mean','endpoint_mean',BLUE)]:
        q=times[times.operation.eq(operation)].sort_values('N')
        axes[1].loglog(q.N,1000*q.wall_median,'o-',color=color,label=_label(method))
    axes[1].set(xlabel=r'$N$',ylabel='Wall time (ms)',title='64 corrections and one output')
    axes[1].legend(frameon=False);_clean(axes[1],axis='both')
    return _finish(fig,path)


def beam_directional(data,path=None):
    """Native complex equal-spectrum beliefs with different task decisions."""
    from .distribution_geometry_visuals import sphere_mesh,restriction_log_density,shared_density_norm,draw_density_sphere
    f=data['fixture'];Q=np.asarray(f['Q'])[:,:3];mesh=sphere_mesh()
    matrices=[f['A1'],f['A2']];logs=[]
    for i,A in enumerate(matrices,1):
        z=float(f['logZ'+str(i)])
        logs.append(restriction_log_density(A,Q,z,mesh.nodes)-(np.linalg.eigvalsh(A)[-1]-z))
    norm=shared_density_norm(logs,decades=4)
    fig=plt.figure(figsize=(15.7,5.1),layout='constrained')
    axes=[fig.add_subplot(1,3,i+1,projection='3d' if i<2 else None) for i in range(3)]
    radius=float(data['fixture_table'].radius95.iloc[0])
    for i,(A,values) in enumerate(zip(matrices,logs)):
        mapping=draw_density_sphere(axes[i],mesh,values,norm=norm,mode=np.array([1.,0,0]),
            A=(Q.conj().T@A@Q).real,ball_radius=radius,title=rf'$q_{{A^{{({i+1})}}}}$',
            axis_labels=(r'$x_1$',r'$x_2$',r'$x_3$'))
    fig.colorbar(mapping,ax=axes[:2],shrink=.64,pad=.01,extend='min',label=r'$q_A([U_3x])/q_A(\widehat{\mathcal X})$')
    table=data['fixture_table'];x=np.arange(2)
    for i,color in [(1,BLUE),(2,ORANGE)]:
        g=table[table.distribution.eq(i)].set_index('method')
        axes[2].bar(x+(i-1.5)*.32,[g.loc[m,'excess'] for m in ('mode','iso')],width=.3,color=color,
            label=rf'$q_{{A^{{({i})}}}}$')
    axes[2].set(xticks=x,xticklabels=[r'$P_{\widehat{\mathcal X}}$',r'$\overline P(\kappa P_{\widehat{\mathcal X}})$'],
        ylabel='Conditional expected-power excess',title='Equal radius does not determine the decision')
    axes[2].legend(frameon=False);_clean(axes[2])
    return _finish(fig,path)


def beam_benchmarks(data,path=None):
    frame=_select(data['metrics'],s=.2,t=16)
    fig,axes=plt.subplots(1,3,figsize=(15.8,4.6),layout='constrained')
    for ax,N in zip(axes[:2],[8,16]):
        g=frame[frame.law.eq('haar') & frame.n.eq(N)];a=g[g.method.eq('lw')].set_index('trial');b=g[g.method.eq('mean')].set_index('trial')
        ids=a.index.intersection(b.index);x=a.loc[ids,'true_power'];y=b.loc[ids,'true_power']
        lo=min(x.min(),y.min())*.95;hi=max(x.max(),y.max())*1.04
        ax.scatter(x,y,s=23,alpha=.5,color=BLUE,edgecolors='none');ax.plot([lo,hi],[lo,hi],ls='--',color=GREY)
        ax.set(xlabel='Ledoit--Wolf output power',ylabel=r'$\overline P_t^+$ output power',xlim=(lo,hi),ylim=(lo,hi),
            title=rf'Haar spatial signatures, $N={N}$')
        _clean(ax)
    g=frame[frame.law.eq('ula10') & frame.n.eq(8)];base=g[g.method.eq('mean')].set_index('trial')
    for method in ['mode','iso','scm','grouse']:
        other=g[g.method.eq(method)].set_index('trial');ids=base.index.intersection(other.index)
        gains=np.sort((other.loc[ids,'true_power']-base.loc[ids,'true_power']).to_numpy())
        axes[2].plot(gains,np.arange(1,len(gains)+1)/len(gains),color=_color(method),label=_label(method),lw=1.7)
    axes[2].axvline(0,color=GREY,ls='--',lw=1)
    axes[2].set(xlabel=r'Reference power $-\,\overline P_t^+$ power',ylabel='Fraction of trajectories',
        title='Paired differences retain gains and losses',ylim=(0,1))
    axes[2].legend(frameon=False,fontsize=8);_clean(axes[2])
    return _finish(fig,path)


def beam_responses(data,path=None):
    story=beam_story(data,16,8);angles=np.asarray(data['stories']['scan_degrees']);V=story['scan_vectors']
    fig,axes=plt.subplots(1,3,figsize=(15.8,4.7),layout='constrained')
    for method in ['mean','mode','lw']:
        w=story['weight_'+method];response=abs(V.conj()@w)**2
        axes[0].plot(angles,10*np.log10(np.maximum(response,1e-5)),color=_color(method),label=_label(method))
    axes[0].axvline(0,color=GREEN,ls=':',lw=1.2);axes[0].axvline(10,color=ORANGE,ls=':',lw=1.2)
    axes[0].set(xlabel=r'Arrival angle ($^\circ$)',ylabel=r'$|w^Ha(\theta)|^2$ (dB)',
        title='The actual distortionless beampattern',ylim=(-45,5));axes[0].legend(frameon=False,fontsize=9);_clean(axes[0])
    scan=_select(data['scans'],n=8,law='ula10',s=.2,t=16,trial=0).sort_values('angle_deg')
    x=scan.angle_deg.to_numpy()
    axes[1].fill_between(x,scan.q025.to_numpy(),scan.q975.to_numpy(),color=BLUE,alpha=.17,label=r'Central 95\%')
    axes[1].fill_between(x,scan.q25.to_numpy(),scan.q75.to_numpy(),color=BLUE,alpha=.3,label=r'Central 50\%')
    axes[1].plot(x,scan['mean'],color=BLUE,label=r'$\overline P_t^+$')
    axes[1].plot(x,scan.truth,color=BLACK,label='True covariance')
    axes[1].set(xlabel=r'Arrival angle ($^\circ$)',ylabel='Covariance scanning power',title='Covariance response')
    axes[1].legend(frameon=False,fontsize=8);_clean(axes[1])
    pair=_select(data['paired'],n=8,s=.2,reference='lw',endpoint='true_power')
    matrix=np.full((3,3),np.nan);uncertain=np.ones((3,3),bool)
    for i,t in enumerate([4,16,64]):
        for j,angle in enumerate([3,10,30]):
            r=pair[(pair.t==t)&pair.law.eq(f'ula{angle}')].iloc[0]
            baseline=_select(data['metrics'],n=8,s=.2,t=t,law=f'ula{angle}',method='lw').true_power.mean()
            matrix[i,j]=100*r.gain/baseline;uncertain[i,j]=not(r.lower>0 or r.upper<0)
    limit=max(float(np.nanmax(abs(matrix))),1.)
    image=axes[2].imshow(matrix,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
    for (i,j),value in np.ndenumerate(matrix):
        axes[2].text(j,i,f'{value:+.1f}',ha='center',va='center',color='white' if abs(value)>limit*.6 else BLACK,fontsize=10)
        if uncertain[i,j]:axes[2].plot(j+.33,i-.28,'o',ms=3,mfc='none',mec=GREY)
    axes[2].set(xticks=np.arange(3),xticklabels=[r'$3^\circ$',r'$10^\circ$',r'$30^\circ$'],
        yticks=np.arange(3),yticklabels=['4','16','64'],xlabel='Desired/interferer separation',ylabel='Snapshots',
        title='Power saved relative to Ledoit--Wolf')
    fig.colorbar(image,ax=axes[2],shrink=.72,pad=.02,label=r'Power saved (\%)')
    return _finish(fig,path)
