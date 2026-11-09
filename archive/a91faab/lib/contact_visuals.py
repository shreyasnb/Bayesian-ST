"""Physical paths, paired decisions and uncertainty for the contact pilot.

Figures use the shared TeX configuration. Surface geometry is plant/scoring
information, never an input to the estimators displayed here.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from .plotting import plt, save
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


CONDITIONS=['plane','sphere_large','sphere_small','weak','anisotropic']
LABELS={'plane':'Plane','sphere_large':r'$R=0.20$ m',
        'sphere_small':r'$R=0.05$ m','weak':'Weak probes',
        'anisotropic':'Anisotropic resistance'}
COLORS={'plane':'#0072B2','sphere_large':'#009E73','sphere_small':'#D55E00',
        'weak':'#CC79A7','anisotropic':'#E69F00'}
METHODS=['Posterior mean','Mode','Tuned mode','Scalar shrinkage','Ridge map','Velocity PCA']
METHOD_COLORS={'Posterior mean':'#0072B2','Mode':'#D55E00','Tuned mode':'#777777',
               'Scalar shrinkage':'#009E73','Ridge map':'#CC79A7','Velocity PCA':'#E69F00'}


def _finish(fig,path):
    assert plt.rcParams['text.usetex'] and fig.dpi==200
    if path is not None:save(fig,Path(path))
    return fig


def _config(frames):
    config=frames.get('config',{})
    return config.get('config',config)


def _v0(frames):return float(_config(frames).get('v0',.02))


def _order(values,stress=True):
    values=set(values)
    return [c for c in CONDITIONS if c in values and (stress or c!='anisotropic')]


def _column(frame,names,required=True):
    for name in names:
        if name in frame:return name
    if required:raise KeyError('Missing one of '+', '.join(names))
    return None


def _vector(frame,alternatives):
    for names in alternatives:
        if all(name in frame for name in names):return frame[list(names)].to_numpy(float)
    return None


def _student(values):
    values=np.asarray(values,float)
    if len(values)<2:return 0.
    return float(student_t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))


def _parameter(value):
    if value and abs(value)<.001:
        power=int(np.floor(np.log10(abs(value))));coefficient=value/10**power
        return rf'10^{{{power}}}' if np.isclose(coefficient,1.) else rf'{coefficient:g}\times10^{{{power}}}'
    return f'{value:g}'


def physical_setup(frames,path=None):
    """Three physical surfaces with one preassigned path and tangent planes."""
    data=frames['paths'];conditions=[c for c in ['plane','sphere_large','sphere_small'] if c in set(data.condition)]
    fig=plt.figure(figsize=(15,5.7),dpi=200,layout='constrained')
    axes=[fig.add_subplot(1,len(conditions),i+1,projection='3d') for i in range(len(conditions))]
    time_name=_column(data,['time','physical_time','tau'])
    color_norm=Normalize(float(data[time_name].min()),float(data[time_name].max()))
    cmap=plt.get_cmap('viridis')
    for ax,condition in zip(axes,conditions):
        g=data[data.condition==condition].sort_values('t')
        if 'trial' in g:g=g[g.trial==g.trial.min()]
        positions=g[['x','y','z']].to_numpy(float)
        normals=g[['nx','ny','nz']].to_numpy(float)
        radius=float(np.median(np.linalg.norm(positions,axis=1))) if condition!='plane' else None
        extent=(radius if radius else max(float(np.ptp(positions,axis=0).max())*.7,.04))
        if radius:
            longitude=np.linspace(0,2*np.pi,33);latitude=np.linspace(0,np.pi,18)
            sx=radius*np.outer(np.cos(longitude),np.sin(latitude))
            sy=radius*np.outer(np.sin(longitude),np.sin(latitude))
            sz=radius*np.outer(np.ones_like(longitude),np.cos(latitude))
            ax.plot_surface(sx,sy,sz,color='#D9DFE4',alpha=.14,linewidth=0,shade=False)
            ax.plot_wireframe(sx,sy,sz,color='#9EA9B0',alpha=.22,lw=.35,rstride=3,cstride=3)
        else:
            normal=normals[0];basis=np.linalg.svd(normal[None,:])[2][1:].T
            centre=positions.mean(axis=0)
            q=np.linspace(-extent,extent,9);aa,bb=np.meshgrid(q,q)
            plane=centre[:,None,None]+basis[:,0,None,None]*aa+basis[:,1,None,None]*bb
            ax.plot_surface(*plane,color='#D9DFE4',alpha=.3,shade=False,linewidth=0)
        segments=np.stack([positions[:-1],positions[1:]],axis=1)
        line=Line3DCollection(segments,cmap=cmap,norm=color_norm,lw=2.2)
        line.set_array(g[time_name].to_numpy()[:-1]);ax.add_collection3d(line)
        ax.scatter(*positions[0],s=30,facecolor='white',edgecolor='#222222',zorder=5)
        ax.scatter(*positions[-1],s=30,color='#222222',zorder=5)
        patch_size=extent*.17
        for index in [0,len(positions)//2,len(positions)-1]:
            tangent=np.linalg.svd(normals[index][None,:])[2][1:].T
            offsets=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
            vertices=positions[index]+patch_size*offsets@tangent.T
            ax.add_collection3d(Poly3DCollection([vertices],facecolor='#56B4E9',edgecolor='#0072B2',alpha=.22,lw=.7))
        free=_vector(g,[('vx','vy','vz'),('free_vx','free_vy','free_vz'),('v0','v1','v2')])
        actual=_vector(g,[('dx','dy','dz'),('velocity_x','velocity_y','velocity_z'),('dx0','dx1','dx2')])
        index=len(positions)//2
        if free is not None and actual is not None:
            for vector,color in [(free[index],'#0072B2'),(actual[index],'#D55E00')]:
                arrow=extent*.5*vector/_v0(frames)
                ax.quiver(*positions[index],*arrow,color=color,lw=1.8,arrow_length_ratio=.22)
        centre=np.zeros(3) if radius else positions.mean(axis=0)
        limit=extent*1.17
        for setter,c in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],centre):setter(c-limit,c+limit)
        ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=35 if condition=='plane' else -57)
        ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=LABELS[condition])
        ax.tick_params(labelsize=8,pad=0)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
            axis.pane.fill=False;axis.set_major_locator(MaxNLocator(4))
            axis._axinfo['grid'].update(color=(.65,.65,.65,.3),linewidth=.5)
    fig.colorbar(ScalarMappable(norm=color_norm,cmap=cmap),ax=axes,orientation='horizontal',
                 fraction=.04,pad=.04,shrink=.7,label='Time (s)')
    fig.legend(handles=[Line2D([],[],color='#0072B2',lw=2,label=r'Free velocity $F/d$'),
                        Line2D([],[],color='#D55E00',lw=2,label=r'Constrained velocity $\dot p$'),
                        Line2D([],[],color='#56B4E9',lw=6,alpha=.5,label='Local tangent plane')],
               loc='outside upper center',ncol=3,fontsize=10)
    fig.supxlabel('One preassigned trajectory per surface. Tangent planes come from geometry; arrows are scaled for display.',fontsize=10)
    return _finish(fig,path)


def paired_losses(frames,path=None,include_stress=False):
    """Separate nearby projector decisions from the much larger ridge gains."""
    data=frames['paired'];conditions=_order(data.condition,stress=include_stress)
    fig=plt.figure(figsize=(max(15.,3.5*len(conditions)),7.7),dpi=200,layout='constrained')
    grid=fig.add_gridspec(2,len(conditions),height_ratios=[1.,1.15])
    axes=[fig.add_subplot(grid[0,index]) for index in range(len(conditions))]
    ridge=fig.add_subplot(grid[1,:])
    selected_failures=frames.get('metadata',{}).get('checks',{}).get('selected_validation_numerical_failures',0)
    for ax,condition in zip(axes,conditions):
        g=data[data.condition==condition]
        refs=[m for m in ['Mode','Tuned mode','Scalar shrinkage'] if m in set(g.reference)]
        point_rows=g[g.reference.isin(refs)]
        magnitude=float(np.abs(point_rows[['low','high']].to_numpy()).max())/_v0(frames)**2
        power=int(np.floor(np.log10(magnitude))) if magnitude else 0
        if abs(power)<2:power=0
        scale=10.**power
        ax.axvline(0,color='#555555',ls='--',lw=1)
        for index,method in enumerate(refs):
            row=g[g.reference==method].iloc[0]
            failed=bool(getattr(row,'numerical_failures',0)>0 or selected_failures>0)
            value=float(row.gain)/(_v0(frames)**2*scale);lo=float(row.low)/(_v0(frames)**2*scale);hi=float(row.high)/(_v0(frames)**2*scale)
            ax.errorbar(value,index,xerr=[[value-lo],[hi-value]],fmt='o',capsize=3,
                        color=METHOD_COLORS[method],markerfacecolor='white' if failed else METHOD_COLORS[method],ms=6)
        title=LABELS.get(condition,condition)
        if condition=='sphere_small':title+=' (primary)'
        unit=r'MSE gain / $v_0^2$'+(rf' ($\times10^{{{power}}}$)' if power else '')
        ax.set(title=title,yticks=range(len(refs)),yticklabels=refs,xlabel=unit)
        ax.invert_yaxis();ax.grid(axis='x',alpha=.12)
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    ridge.axvline(0,color='#555555',ls='--',lw=1)
    for index,condition in enumerate(conditions):
        row=data[(data.condition==condition)&(data.reference=='Ridge map')].iloc[0]
        failed=bool(getattr(row,'numerical_failures',0)>0 or selected_failures>0)
        value=float(row.gain)/_v0(frames)**2;lo=float(row.low)/_v0(frames)**2;hi=float(row.high)/_v0(frames)**2
        ridge.errorbar(value,index,xerr=[[value-lo],[hi-value]],fmt='o',capsize=3,
                       color=COLORS[condition],markerfacecolor='white' if failed else COLORS[condition],ms=6)
    ridge.set(title='Ridge linear map',yticks=range(len(conditions)),
              yticklabels=[LABELS[c] for c in conditions],xlabel=r'Response MSE reduction / $v_0^2$')
    ridge.invert_yaxis();ridge.grid(axis='x',alpha=.12)
    fig.supxlabel(r'Reference loss minus posterior-mean loss; positive favors the posterior mean. Paired 95\% Student intervals across independent trajectories. '
                  'Queries are averaged within each trajectory; axes use different scales. Open markers flag unresolved integration.',fontsize=9)
    return _finish(fig,path)


def coverage_width(frames,path=None,time=None,include_stress=False):
    """State regions and scalar-response intervals keep separate width units."""
    data=frames['coverage'];time=int(data.t.max()) if time is None else time
    data=data[data.t==time]
    available=set(data.region)
    specifications=[(['State ball','ball','ball_covered'],'Subspace ball','Mean chordal radius'),
                    (['State HPD','hpd','hpd_covered'],'State HPD region','Mean HPD loss threshold'),
                    (['Clean response','component','clean','component_clean','clean_response','component_coverage_mean'],'Clean response',r'Mean scalar interval width / $v_0$'),
                    (['noisy','component_noisy','noisy_response'],'Noisy observation',r'Mean scalar interval width / $v_0$')]
    selected=[(next(name for name in aliases if name in available),title,label)
              for aliases,title,label in specifications if any(name in available for name in aliases)]
    if len(selected)<3:raise ValueError('Expected state ball, HPD and component-response coverage summaries')
    if len(selected)==3:shape=(1,3);size=(15,5)
    else:shape=(2,2);size=(12,8)
    fig,axs=plt.subplots(*shape,figsize=size,dpi=200,layout='constrained',squeeze=False)
    for ax,(region,title,label) in zip(axs.flat,selected):
        ax.axhline(.95,color='#555555',ls='--',lw=1)
        subset=data[data.region==region]
        for condition in _order(subset.condition,stress=include_stress):
            row=subset[subset.condition==condition].iloc[0]
            ok=(getattr(row,'numerical_failures',0)==0 and getattr(row,'numerical_ok',True))
            ax.errorbar(row.mean_width,row.coverage,yerr=[[row.coverage-row.low],[row.high-row.coverage]],
                        fmt='o',ms=7,capsize=3,color=COLORS[condition],
                        markerfacecolor=COLORS[condition] if ok else 'white')
        ax.set(title=title,xlabel=label,ylabel='Coverage',ylim=(-.025,1.025));ax.grid(alpha=.12)
    fig.legend(handles=[Line2D([],[],color=COLORS[c],marker='o',ls='',label=LABELS[c])
                        for c in _order(data.condition,stress=include_stress)],
               loc='outside upper center',ncol=3,fontsize=10)
    count=int(data.trials.max())
    fig.supxlabel(f'Observation {time}; {count} independent trajectories. State bars: '+r'95\% Wilson intervals; component bars: Student intervals across trajectory means.'+'\n'
                  'Component queries are averaged within each trajectory. Region sizes have different units. Open markers flag unresolved integration.',fontsize=9)
    return _finish(fig,path)


def response_intervals(frames,path=None,condition='weak',trial=0):
    """Pointwise response intervals for three preassigned hypothetical forces."""
    data=frames['query_predictions'];g=data[(data.condition==condition)&(data.trial==trial)]
    components=sorted(g.component.unique());queries=sorted(g['query'].unique())[:3]
    fig,axs=plt.subplots(len(components),len(queries),figsize=(13,8),dpi=200,layout='constrained',sharex=True,squeeze=False)
    for row,component in enumerate(components):
        for col,query in enumerate(queries):
            ax=axs[row,col];part=g[(g.component==component)&(g['query']==query)].sort_values('t')
            for x in part.itertuples():
                good=bool(getattr(x,'numerical_ok',True))
                ax.plot([x.t,x.t],np.array([x.lower,x.upper])/_v0(frames),color='#0072B2' if good else '#AA3377',lw=4,alpha=.35)
            ax.plot(part.t,part['mean']/_v0(frames),'o-',ms=4,color='#0072B2',label='Posterior mean')
            ax.plot(part.t,part.truth/_v0(frames),'*-',ms=7,color='#222222',lw=1.2,label='True response')
            if 'numerical_ok' in part:
                unresolved=~part.numerical_ok.astype(bool)
                ax.scatter(part.loc[unresolved,'t'],part.loc[unresolved,'mean']/_v0(frames),
                           s=38,facecolors='white',edgecolors='#AA3377',zorder=6)
            if 'mode' in part:ax.plot(part.t,part['mode']/_v0(frames),'--',color='#D55E00',lw=1.1,label='Mode')
            ax.grid(axis='y',alpha=.12)
            if row==0:ax.set_title(f'Query {int(query)+1}')
            if col==0:ax.set_ylabel(rf'Component {int(component)+1} / $v_0$')
            if row==len(components)-1:ax.set_xlabel('Observation')
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=3)
    fig.supxlabel(LABELS[condition]+r': preassigned trajectory zero; bars are pointwise 95\% clean-response intervals. '
                  'Open points flag unresolved integration. Queries do not move the tool.',fontsize=9)
    return _finish(fig,path)


def risk_calibration(frames,path=None):
    """Maintained posterior risk versus realized nine-query response loss."""
    data=frames['posterior'];data=data[data.t==data.t.max()]
    risk_name=_column(data,['risk_mean','posterior_risk','mean_risk'])
    error_name=_column(data,['empirical_response_error','response_loss','loss'])
    fig,axs=plt.subplots(1,2,figsize=(13,5.4),dpi=200,layout='constrained')
    upper=0.
    for index,condition in enumerate(_order(data.condition)):
        g=data[data.condition==condition];risk=g[risk_name]/_v0(frames)**2;error=g[error_name]/_v0(frames)**2
        axs[0].scatter(risk,error,s=22,alpha=.5,color=COLORS[condition],label=LABELS[condition],edgecolors='none')
        upper=max(upper,float(risk.max()),float(error.max()))
        difference=(error-risk).to_numpy();mean=float(difference.mean());half=_student(difference)
        axs[1].errorbar(mean,index,xerr=half,fmt='o',capsize=4,color=COLORS[condition])
    axs[0].plot([0,upper],[0,upper],'--',color='#666666',lw=1)
    axs[0].set(xlabel=r'Posterior clean-response risk / $v_0^2$',ylabel=r'Realized response MSE / $v_0^2$',title='Risk and error are different quantities')
    axs[0].legend(fontsize=9);axs[0].grid(alpha=.12)
    conditions=_order(data.condition);axs[1].axvline(0,ls='--',color='#666666',lw=1)
    axs[1].set(yticks=range(len(conditions)),yticklabels=[LABELS[c] for c in conditions],
               xlabel=r'Mean (error minus risk) / $v_0^2$',title='Independent-trajectory mean discrepancy')
    axs[1].invert_yaxis();axs[1].grid(axis='x',alpha=.12)
    fig.supxlabel(r'Queries are averaged within each trajectory. Bars are pointwise 95\% Student intervals; model mismatch can invalidate posterior-risk calibration.',fontsize=10)
    return _finish(fig,path)


def benchmark_losses(frames,path=None):
    """All endpoint methods on a log scale, including force-blind velocity PCA."""
    data=frames['metrics'];data=data[data.t==data.t.max()]
    conditions=_order(data.condition)
    fig,axs=plt.subplots(1,len(conditions),figsize=(3.5*len(conditions),5.4),dpi=200,layout='constrained',squeeze=False)
    for ax,condition in zip(axs.flat,conditions):
        g=data[data.condition==condition]
        methods=[m for m in METHODS if m in set(g.method)]
        for index,method in enumerate(methods):
            values=g[g.method==method].loss.to_numpy()/_v0(frames)**2
            ax.scatter(np.full(len(values),index),values,s=11,color=METHOD_COLORS[method],alpha=.35,edgecolors='none')
            ax.scatter(index,values.mean(),s=40,marker='D',color=METHOD_COLORS[method],edgecolor='white',zorder=4)
        ax.set(yscale='log',xticks=range(len(methods)),xticklabels=methods,title=LABELS[condition],ylabel=r'Response MSE / $v_0^2$')
        ax.tick_params(axis='x',rotation=65,labelsize=8);ax.grid(axis='y',alpha=.12)
    fig.supxlabel('Terminal loss per independent trajectory; diamonds show means. Velocity PCA discards the applied-force inputs.',fontsize=10)
    return _finish(fig,path)


def runtime(frames,path=None):
    """Actual reporting frequency is retained in cumulative method costs."""
    raw=frames['metrics'];rows=[]
    for _,g in raw.groupby(['condition','trial','method'],sort=False):
        row=g.sort_values('t').iloc[-1].copy()
        for column in ['integration_seconds','decision_seconds']:
            if column in g:row[column]=g[column].sum()
        rows.append(row)
    data=pd.DataFrame(rows)
    fig,axs=plt.subplots(1,2,figsize=(13,5),dpi=200,layout='constrained')
    methods=[m for m in METHODS if m in set(data.method)]
    cost_columns=[c for c in ['update_seconds','integration_seconds','decision_seconds'] if c in data]
    labels={'update_seconds':'Correction','integration_seconds':'Posterior integration','decision_seconds':'Decision'}
    colors=['#777777','#0072B2','#009E73'];left=np.zeros(len(methods))
    for column,color in zip(cost_columns,colors):
        values=np.array([data[data.method==m][column].mean() for m in methods])
        axs[0].barh(range(len(methods)),values,left=left,label=labels[column],color=color,alpha=.8);left+=values
    axs[0].set(yticks=range(len(methods)),yticklabels=methods,xlabel='Mean cumulative time (s)',title='Measured computation')
    axs[0].invert_yaxis();axs[0].legend(fontsize=9);axs[0].grid(axis='x',alpha=.12)
    for method in methods:
        g=data[data.method==method]
        total=sum(g[c] for c in cost_columns)
        axs[1].scatter(total,g.loss/_v0(frames)**2,s=20,alpha=.5,color=METHOD_COLORS[method],label=method,edgecolors='none')
    axs[1].set(xscale='log',yscale='log',xlabel='Cumulative time (s)',ylabel=r'Response MSE / $v_0^2$',title='Cost and task loss')
    axs[1].grid(alpha=.12)
    fig.supxlabel('Update time is cumulative; integration and decisions are summed over the three reporting times. '
                  'Cloud implementation timings do not establish a general speed advantage.',fontsize=9)
    return _finish(fig,path)


def mechanical_diagnostics(frames,path=None):
    """Travel, motion, retained excitation and true local observation information."""
    data=frames['trajectories'];posterior=frames['posterior'];paths=frames['paths']
    posterior=posterior[posterior.t==posterior.t.max()]
    conditions=_order(data.condition)
    fig,axs=plt.subplots(2,2,figsize=(13,8),dpi=200,layout='constrained')
    for index,condition in enumerate(conditions):
        g=data[data.condition==condition]
        for ax,column,label in [(axs[0,0],'surface_travel','Surface travel (m)'),
                                (axs[0,1],'normal_rotation','Accumulated normal rotation (rad)')]:
            value=g[column].to_numpy(float);mean=float(value.mean());half=_student(value)
            ax.scatter(value,np.full(len(value),index),s=12,alpha=.25,color=COLORS[condition],edgecolors='none')
            ax.errorbar(mean,index,xerr=half,fmt='D',ms=5,capsize=3,color=COLORS[condition])
            ax.set(xlabel=label,yticks=range(len(conditions)),yticklabels=[LABELS[c] for c in conditions])
        g=posterior[posterior.condition==condition]
        value=g.minimum_retained_excitation.to_numpy(float)/_v0(frames)**2
        axs[1,0].scatter(value,np.full(len(value),index),s=14,alpha=.4,color=COLORS[condition],edgecolors='none')
    axs[0,0].set_title('Physical travel');axs[0,1].set_title('Geometry-driven motion')
    axs[1,0].set(yticks=range(len(conditions)),yticklabels=[LABELS[c] for c in conditions],
                 xscale='log',xlabel=r'Minimum retained input-Gram eigenvalue / $v_0^2$',title='Input excitation')
    for ax in [axs[0,0],axs[0,1],axs[1,0]]:ax.invert_yaxis();ax.grid(axis='x',alpha=.12)
    ax=axs[1,1];sigma=float(_config(frames).get('sigma',.001))
    for condition in ['sphere_small','weak']:
        g=paths[paths.condition==condition].sort_values('t')
        normal=g[['nx','ny','nz']].to_numpy();v=g[['vx','vy','vz']].to_numpy()
        # For n in S^2 and a tangent perturbation, Gaussian observation Fisher
        # is [(n^T v)^2 I_2 + (Q^T v)(Q^T v)^T]/s. The smaller eigenvalue is
        # (n^T v)^2/s. This plant/scoring diagnostic is not supplied to a filter.
        local=np.sum(normal*v,axis=1)**2/sigma**2
        ax.plot(g.time,local,color=COLORS[condition],label=LABELS[condition])
    ax.set(xlabel='Time (s)',ylabel=r'$\lambda_{\min}(F_t)$',title='Local information at the true tangent')
    ax.legend(fontsize=9);ax.grid(alpha=.12)
    count=int(data.trial.nunique())
    fig.supxlabel(rf'Top: {count} independent trajectories per condition, mean and pointwise 95\% intervals. Bottom right: one preassigned path. '
                  'Small input-Gram eigenvalues alone do not imply weak tangent information.',fontsize=10)
    return _finish(fig,path)


def validation_curves(frames,path=None):
    """Show the frozen curved-mixture tuning without treating conditions as trials."""
    data=frames['tuning'];data=data[data.scope=='curved']
    fig,axs=plt.subplots(1,3,figsize=(15,4.7),dpi=200,layout='constrained')
    ax=axs[0]
    for method in ['Posterior mean','Tuned mode','Velocity PCA']:
        g=data[data.method==method].sort_values('beta')
        if not len(g):continue
        y=g.validation_loss/_v0(frames)**2
        ax.plot(g.beta,y,'o-',ms=4,color=METHOD_COLORS[method],label=method)
        good=g.selected.astype(bool)
        ax.scatter(g.loc[good,'beta'],y[good],marker='*',s=110,color=METHOD_COLORS[method],edgecolor='white',zorder=5)
    ax.set(title='Memory',xlabel=r'$\beta$',ylabel=r'Validation response MSE / $v_0^2$');ax.legend(fontsize=9)
    for ax,method,column,symbol in [(axs[1],'Scalar shrinkage','rho',r'\rho'),
                                    (axs[2],'Ridge map','regularization',r'\lambda')]:
        g=data[data.method==method]
        optimized=pd.DataFrame()
        if method=='Scalar shrinkage' and 'optimization' in g:
            analytic=g.optimization.fillna('').str.contains('Analytic',case=False)
            optimized=g[analytic].sort_values('beta');g=g[~analytic]
        parameters=sorted(g[column].dropna().unique());palette=plt.get_cmap('viridis')
        for index,value in enumerate(parameters):
            part=g[np.isclose(g[column],value)].sort_values('beta')
            color=palette(index/max(len(parameters)-1,1));y=part.validation_loss/_v0(frames)**2
            ax.plot(part.beta,y,'o-',ms=3,color=color,label=rf'${symbol}={_parameter(value)}$')
            selected=part.selected.astype(bool)
            ax.scatter(part.loc[selected,'beta'],y[selected],marker='*',s=110,color=color,edgecolor='white',zorder=5)
        if len(optimized):
            y=optimized.validation_loss/_v0(frames)**2
            ax.plot(optimized.beta,y,'k--',lw=1.7,label=r'Optimized $\rho$')
            selected=optimized.selected.astype(bool)
            ax.scatter(optimized.loc[selected,'beta'],y[selected],marker='*',s=110,color='#222222',edgecolor='white',zorder=5)
        ax.set(title=method,xlabel=r'$\beta$',ylabel=r'Validation response MSE / $v_0^2$')
        ax.legend(fontsize=8,ncol=2)
    for ax in axs:
        ax.set_yscale('log');ax.set_xticks(sorted(data.beta.unique()))
        ax.grid(axis='y',alpha=.12)
    fig.supxlabel('One policy per method is selected on the curved validation mixture; stars mark frozen choices. '
                  'The static plane control is tuned separately with no forgetting.',fontsize=10)
    return _finish(fig,path)
