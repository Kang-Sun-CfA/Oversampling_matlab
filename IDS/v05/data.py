import numpy as np
import pandas as pd
import logging
import torch
from torch.utils.data import Dataset
import os,sys,glob
import re
import matplotlib.pyplot as plt
from netCDF4 import Dataset as Ds_nc

def years_since(dt,base_year=2020):
    '''
    convert timestamps to years since base_year (float64), the raw time t that 
    TemporalEncoder in model.py encodes
    dt:
        pd.DatetimeIndex, e.g., mid points of aggregation intervals
    returns:
        np.ndarray of years since base_year
    '''
    dt = pd.DatetimeIndex(dt)
    nday_in_year = 365+dt.is_leap_year.astype(int)
    doy = dt.day_of_year-1+dt.hour/24+dt.minute/1440+dt.second/86400
    return np.asarray(dt.year-base_year+doy/nday_in_year,dtype=np.float64)


def jitter_time(time,span,p_jitter=1.,generator=None):
    '''
    uniformly perturb raw time within its span, before encoding
    time:
        [2] or [B,2], (years since base_year, local solar hour)
    span:
        same shape as time, (aggregation interval in years, configured 
        diurnal jitter span in hours). use 0 to disable jitter for a component
    p_jitter:
        probability to jitter each component of each sample
    returns:
        jittered copy of time; local hour wrapped to [0,24)
    '''
    time = time.clone()
    u = torch.rand(time.shape,generator=generator)-0.5 # [-0.5,0.5)
    do = (torch.rand(time.shape,generator=generator) < p_jitter).float()
    time = time+u*span*do
    time[...,1] = torch.remainder(time[...,1],24.)
    return time


class Stream:
    '''
    one instrument product (e.g., tropomi_no2) on one grid
    terminology:
        unit: one row of the calendar, i.e., one l3 at one local time of one 
            date, e.g., 2020-01-01 13:30 for LEO, or 2020-01-01 09:30, 10:30, ...
            for GEO. times are local, so dates/intervals start at local midnight
        spec: one (interval,offset) pair, i.e., one phase shift
        interval: an aggregation window [start,end), clipped to the stream range
        frame: one aggregated 2-D time step, i.e., (interval, local hour)
    tables:
        calendar_df: index = local times of units; columns ilh (index into 
            local_hours), path, exists, val_fold, and {role}_spec{ispec}, the row 
            in that role's interval table each unit contributes to (-1 if none)
        interval_dfs['train'],interval_dfs['val']: one row per interval, specs stacked
            in config order; columns ispec, interval, offset, start, end, mid, 
            t, t_span, n_day, coverage, ifirst, ilast (calendar positions)
    val_fold:
        assigned by date; all local hours of a date share it
        k in 0..nfold-1: validation date of fold k, training date of the other folds
        -1: training date in case there is only one fold
    '''
    def __init__(
        self,name='tropomi_no2',start='2020-01-01',end='2025-12-31',
        local_hours=[13.5],path_pattern=None,base_year=2020,
        split_kw=None,train_kw=None,val_kw=None,infer_kw=None,psf_idx=0
    ):
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.local_hours = list(local_hours)
        self.base_year = base_year
        # which psf of the model blurs this stream; streams that only split 
        # space or time of the same data share one
        self.psf_idx = int(psf_idx)
        split_kw = {} if split_kw is None else dict(split_kw)
        self.start = pd.Timestamp(start or split_kw['start']).normalize()
        self.end = pd.Timestamp(end or split_kw['end']).normalize()
        
        # calendar of units and file paths
        times,ilh = Stream.get_calendar_times(self.start,self.end,self.local_hours)
        calendar_df = pd.DataFrame(index=times,data={'ilh':ilh})
        if path_pattern is not None:
            calendar_df['path'] = [t.strftime(path_pattern) for t in calendar_df.index]
            calendar_df['exists'] = [os.path.exists(p) for p in calendar_df['path']]
            if not calendar_df['exists'].any():
                self.logger.warning(f'{name}: no l3 file exists!')
        else:
            calendar_df['exists'] = True # dummy calendar, e.g., for tests
        
        # train/val split by date, on dates that may be shared by all streams
        self.nfold = split_kw.pop('nfold',1)
        split_start = split_kw.pop('start',None) or self.start
        split_end = split_kw.pop('end',None) or self.end
        val_fold = Stream.split_calendar(
            start=split_start,end=split_end,nfold=self.nfold,**split_kw
        )
        dates = calendar_df.index.normalize()
        if not dates.isin(val_fold.index).all():
            raise ValueError('split dates must cover the stream calendar!')
        calendar_df['val_fold'] = val_fold.reindex(dates).values
        self.do_val = (calendar_df['val_fold'] >= 0).any()
        
        # aggregation intervals and the unit -> interval mapping. a role is built 
        # only if its kw is given, e.g., only infer_kw after training
        self.interval_dfs = {}
        role_kws = {'train':train_kw,'val':val_kw,'infer':infer_kw}
        for role,kw in role_kws.items():
            if kw is None:continue
            if not self.do_val and role == 'val':continue
            kw = dict(kw)
            specs = kw.pop('specs',[('1M','0D')])
            min_coverage = kw.pop('min_coverage',0.)
            tables = []
            for ispec,(interval,offset) in enumerate(specs):
                df = Stream.get_intervals(self.start,self.end,interval,offset)
                df = df.loc[df['coverage'] >= min_coverage]
                df.insert(0,'ispec',ispec)
                df.insert(1,'interval',interval)
                df.insert(2,'offset',offset)
                tables.append(df)
            table = pd.concat(tables,ignore_index=True)
            for ispec in range(len(specs)):
                rows = np.where(table['ispec'] == ispec)[0]
                calendar_df[f'{role}_spec{ispec}'] = Stream.map_times(
                    calendar_df.index,table['start'].values[rows],
                    table['end'].values[rows],rows
                )
            table['t'] = years_since(table['mid'],base_year)
            nday_in_year = 365+pd.DatetimeIndex(table['mid']).is_leap_year.astype(int)
            table['t_span'] = table['n_day']/nday_in_year
            table['ifirst'],table['ilast'] = Stream.get_bounds(
                calendar_df[[f'{role}_spec{i}' for i in range(len(specs))]].values,len(table))
            self.interval_dfs[role] = table
        self.calendar_df = calendar_df
    
    @staticmethod
    def get_calendar_times(start,end,local_hours):
        '''
        local times of all units, date-major: d0 h0, d0 h1, ..., d1 h0, ...
        returns:
            pd.DatetimeIndex of times, np.ndarray of local hour indices
        '''
        dates = pd.date_range(
            pd.Timestamp(start).normalize(),pd.Timestamp(end).normalize(),freq='D')
        nlh = len(local_hours)
        hours = pd.to_timedelta(
            np.tile(np.asarray(local_hours,dtype=float),len(dates)),unit='h')
        times = pd.DatetimeIndex(dates.repeat(nlh))+hours
        return times,np.tile(np.arange(nlh),len(dates))
    
    @staticmethod
    def parse_interval(s):
        '''
        parse an interval/offset string explicitly
        'nD' or 'nd' -> ('D',n), n days, e.g., '30D', '91d', '-15d', '0D'
        'nM' -> ('M',n), n calendar months, e.g., '1M', '3M'
        lowercase 'm' is rejected, as pandas reads it as minutes
        '''
        m = re.fullmatch(r'\s*([+-]?\d+)\s*([dDM])\s*',str(s))
        if m is None:
            raise ValueError(
                f'cannot parse {s!r}; use nD (days) or nM (calendar months), '
                'e.g., 30D or 1M; lowercase m is minutes in pandas!')
        return m.group(2).upper(),int(m.group(1))
    
    @staticmethod
    def shift(time_,unit,n):
        '''time + n days or n calendar months'''
        if unit == 'D':
            return time_+pd.Timedelta(days=n)
        return time_+pd.DateOffset(months=n)
    
    @staticmethod
    def get_intervals(start,end,interval='1M',offset='0D'):
        '''
        intervals [start,end) of one spec covering dates start..end (inclusive).
        the same for days and months: edge(k) = shift(start + k*interval, offset),
        from the last edge at or before start to the first edge after end. 
        edge intervals are clipped to the stream range so their mid/t_span 
        describe the covered dates only. months are anchored at start, so 1M 
        means calendar months when start is the 1st of a month
        returns:
            pd.DataFrame with start, end, mid, n_day, coverage (n_day/full length)
        '''
        start = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end).normalize()+pd.Timedelta(days=1) # exclusive
        unit,n = Stream.parse_interval(interval)
        ounit,on = Stream.parse_interval(offset)
        if n <= 0:
            raise ValueError(f'interval {interval!r} must be positive!')
        def edge(k):
            return Stream.shift(Stream.shift(start,unit,n*k),ounit,on)
        # last edge at or before start
        k = 0
        while edge(k) > start:
            k -= 1
        while edge(k+1) <= start:
            k += 1
        edges = [edge(k)]
        while edges[-1] < end:
            k += 1
            edges.append(edge(k))
        edges = pd.DatetimeIndex(edges)
        full_start,full_end = edges[:-1],edges[1:]
        clip_start = full_start.where(full_start > start,start)
        clip_end = full_end.where(full_end < end,end)
        df = pd.DataFrame(dict(start=clip_start,end=clip_end))
        df['mid'] = df['start']+(df['end']-df['start'])/2
        df['n_day'] = (df['end']-df['start']).dt.days
        df['coverage'] = df['n_day']/(full_end-full_start).days
        return df.loc[df['n_day'] > 0].reset_index(drop=True)
    
    @staticmethod
    def map_times(times,starts,ends,rows):
        '''
        for each time, the row of the interval containing it, -1 if none
        starts,ends:
            sorted, non-overlapping interval bounds of one spec
        rows:
            row numbers of these intervals in the role's interval table
        '''
        times = np.asarray(times,dtype='datetime64[ns]')
        starts = np.asarray(starts,dtype='datetime64[ns]')
        ends = np.asarray(ends,dtype='datetime64[ns]')
        i = np.searchsorted(ends,times,side='right')
        inside = (i < len(starts))
        inside[inside] &= times[inside] >= starts[i[inside]]
        out = np.full(len(times),-1,dtype=int)
        out[inside] = np.asarray(rows)[i[inside]]
        return out
    
    @staticmethod
    def split_calendar(
        start,end,nfold=1,val_interval='1M',val_fraction=0.,seed=0
    ):
        '''
        split dates into folds within each val_interval block, as in v04:
        block i uses rng(seed+i); nfold > 1 permutes dates, array_splits them, and 
        assigns the splits to folds in a random order; nfold == 1 draws 
        int(ndate*val_fraction) validation dates. all local hours of a date share 
        its fold, so streams with different local hours (LEO, GEO) share folds
        returns:
            pd.Series of val_fold indexed by date, -1 for train-only dates
        '''
        dates = pd.date_range(
            pd.Timestamp(start).normalize(),pd.Timestamp(end).normalize(),freq='D')
        val_fold = np.full(len(dates),-1,dtype=int)
        if nfold == 1 and val_fraction == 0:
            return pd.Series(val_fold,index=dates,name='val_fold')
        blocks = Stream.get_intervals(start,end,val_interval,'0D')
        block = Stream.map_times(
            dates,blocks['start'].values,blocks['end'].values,np.arange(len(blocks)))
        for iblock in range(len(blocks)):
            idx = np.where(block == iblock)[0]
            rng = np.random.default_rng(seed+iblock)
            if nfold > 1:
                splits = np.array_split(rng.permutation(idx),nfold)
                for ifold,random_fold in enumerate(rng.permutation(range(nfold))):
                    val_fold[splits[random_fold]] = ifold
            else:
                val_idx = rng.choice(idx,size=int(len(idx)*val_fraction),replace=False)
                val_fold[val_idx] = 0
        return pd.Series(val_fold,index=dates,name='val_fold')
    
    @staticmethod
    def get_bounds(unit_rows,nrow):
        '''
        first/last calendar position of each interval row, so the loader knows 
        when an interval can be closed (its accumulator released)
        unit_rows:
            [nunit,nspec], output of map_times for all specs of a role
        '''
        first,last = np.full(nrow,-1,dtype=int),np.full(nrow,-1,dtype=int)
        for ispec in range(unit_rows.shape[1]):
            iunit = np.where(unit_rows[:,ispec] >= 0)[0]
            rows = unit_rows[iunit,ispec]
            # units are sorted, so the first/last occurrence of each row bounds it
            urows,ifirst = np.unique(rows,return_index=True)
            _,ilast = np.unique(rows[::-1],return_index=True)
            first[urows] = iunit[ifirst]
            last[urows] = iunit[len(rows)-1-ilast]
        return first,last
    
    def estimate_memory(self,nfield,ny,nx,nbyte=4,nbyte_accumulator=8):
        '''
        rough memory in GB, e.g., before load
        frames: float32 aggregated frames of all roles, [nfold,nframe,nfield+2,ny,nx];
            infer has one fold
        accumulator: float64 partial sums of open intervals. each unit is in one 
            interval per spec, so at most one interval per spec is open, holding 
            sums per val_fold group (nfold+1) and local hour: w*v per field, w, n
        '''
        nlh = len(self.local_hours)
        nfold = max(self.nfold,1)
        frames,accumulator = 0,0
        for role,table in self.interval_dfs.items():
            nfold_role = 1 if role == 'infer' else nfold
            frames += nfold_role*len(table)*nlh*(nfield+2)*ny*nx*nbyte
            nspec = table['ispec'].nunique()
            accumulator += nspec*(nfold+1)*nlh*(nfield+2)*ny*nx*nbyte_accumulator
        return dict(frames=frames/1e9,accumulator=accumulator/1e9)
    
    def get_frame_df(self,role='train'):
        '''
        frame table, interval rows crossed with local hours; t and local_hour 
        are the raw time for TemporalEncoder, t_span for its jitter
        '''
        table = self.interval_dfs[role]
        frame_df = table.loc[
            np.repeat(table.index.values,len(self.local_hours)),
            ['ispec','interval','offset','start','end','mid','t','t_span']
        ].rename_axis('irow').reset_index()
        frame_df['local_hour'] = np.tile(self.local_hours,len(table))
        return frame_df
    
    @staticmethod
    def load_popy_l3(
        path,field_names,west=None,east=None,south=None,north=None,grid_size=None
    ):
        '''
        popy-like l3 loader
        '''
        with Ds_nc(path,'r') as nc:
            xgrid = nc['xgrid'][:].data
            ygrid = nc['ygrid'][:].data
            if west is not None:
                xmask = (xgrid >= west) & (xgrid <= east)
                ymask = (ygrid >= south) & (ygrid <= north)
            else:
                xmask = np.ones(len(xgrid),dtype=bool)
                ymask = np.ones(len(ygrid),dtype=bool)
            xidx = np.arange(len(xmask),dtype=int)[xmask]
            yidx = np.arange(len(ymask),dtype=int)[ymask]
            xgrid,ygrid = xgrid[xidx],ygrid[yidx]
            w = nc['total_sample_weight'][yidx,xidx].filled(0.)
            n = nc['num_samples'][yidx,xidx].filled(0.)
            # more efficient to directly load trimed frames than load full
            # then trim. let it break if field names not in file
            v = np.stack(
                [
                    nc[f][yidx,xidx].filled(np.nan) for f in field_names
                ],axis=0
            )
            if grid_size is not None and not np.isclose(grid_size,nc.grid_size):
                do_block_reduce = True
            else:
                do_block_reduce = False
        # use popy machinery if need to
        if do_block_reduce:
            from popy import Level3_Data
            l3 = Level3_Data()
            for key,value in zip(
                ['xgrid','ygrid','total_sample_weight','num_samples'],
                [xgrid,ygrid,w,n]
            ):
                l3.add(key,value)
            for key,value in zip(field_names,v):
                l3.add(key,value)
            l3.check()
            l3 = l3.block_reduce(grid_size)
            xgrid,ygrid = l3['xgrid'],l3['ygrid']
            w,n = l3['total_sample_weight'],l3['num_samples']
            v = np.stack([l3[k] for k in field_names],axis=0)
        
        nan_mask = np.isnan(v).any(axis=0)
        v[np.broadcast_to(nan_mask[None,:,:],v.shape)] = 0
        w[nan_mask] = 0
        n[nan_mask] = 0
        return dict(
            xgrid=xgrid,ygrid=ygrid,v=v.astype(np.float64),
            w=w.astype(np.float64),n=n.astype(np.float64)
        )
    
    def load(
        self,field_names,west=None,east=None,south=None,north=None,grid_size=None,
        skip_errors=True,log_every=1000,log_acc=False):
        '''
        stream calendar units through interval accumulators into aggregated frames
        field_names:
            fields to aggregate as total_sample_weight-weighted means, e.g., 
            column_amount_DD, its _xy/_rs, column_amount, surface_altitude_DD
        creates:
            self.frames[role], float32 [nfold,nframe,nchannel,ny,nx], frames 
            ordered as get_frames(role); channels are self.channel_names
        weighted mean of each field uses pixels where all fields are valid
        '''
        self.field_names = list(field_names)
        self.channel_names = self.field_names+['total_sample_weight','num_samples']
        nlh,nfold = len(self.local_hours),max(self.nfold,1)
        # "group" covers all possible values in val_fold, available for every 
        # calendar_df entry, by shifting its value by +1. dates labeled as 0,1,2,3,4
        # will be in groups 1,2,3,4,5, group 0 is still there but empty. here nfold=5
        # and ngroup=6. dates labeled as -1,0 will be in groups 0,1. here nfold=1 and
        # ngroup=2.
        ngroup = nfold+1 
        # spec_columns are like 'train':['train_spec0','train_spec1'],'val':['val_spec0']
        spec_columns = {
            role:[c for c in self.calendar_df.columns if c.startswith(f'{role}_spec')]
            for role in self.interval_dfs.keys()
        }
        self.frames,accumulators = {},{}
        n_loaded,n_failed = 0,0
        for iunit,(unit_time,row) in enumerate(self.calendar_df.iterrows()):
            # load one unit
            data = None
            if row['exists']:
                try:
                    data = Stream.load_popy_l3(
                        row.path,field_names,west,east,south,north,grid_size
                    )
                except Exception as e:
                    if not skip_errors:
                        raise
                    self.logger.warning(f'{self.name}: failed to load {unit_time}: {e}')
                    n_failed += 1
            if data is not None:
                if not self.frames:
                    # frames separated to train and val, dimension:
                    # nfold,nframe=nilh*ninterval,nchannel,ny,nx, note ninterval includes
                    # intervals of all specs
                    self._init_frames(data,nfold,nlh)
                n_loaded += 1
                # group and local hour indices of the unit (row of calendar_df)
                group,ilh = int(row['val_fold'])+1,int(row['ilh'])
                if log_acc:new_acc_count = 0
                # looping over {train,val}_spec{0,...} for the unit
                for role in self.interval_dfs.keys():
                    for col in spec_columns[role]:
                        # which row in interval_dfs[role] should the unit add to
                        irow = int(row[col])
                        if irow < 0:
                            continue
                        key = (role,irow)
                        # create an accumulator for the row in interval_dfs[role]
                        if key not in accumulators:
                            if log_acc:
                                if new_acc_count == 0:
                                    acc_str = 'at {}'.format(
                                        unit_time.strftime('%Y%m%dT%H%M'))
                                acc_str += ', opening {}:{}-{}'.format(
                                    role[0],
                                    self.interval_dfs[role].iloc[irow].start.strftime('%Y%m%d'),
                                    self.interval_dfs[role].iloc[irow].end.strftime('%Y%m%d')
                                )
                                new_acc_count += 1
                            accumulators[key] = self._new_accumulator(data,ngroup,nlh)
                        # add the unit to the interval, e.g. row in interval_df
                        self._add(accumulators[key],data,group,ilh)
                if log_acc and new_acc_count > 0:
                    acc_str += f', {len(accumulators)} intervals open'
                    self.logger.warning(acc_str)
            # close the intervals ending at this unit, also when it has no data
            for role,table in self.interval_dfs.items():
                for irow in np.where(table['ilast'].values == iunit)[0]:
                    acc = accumulators.pop((role,irow),None)
                    if acc is not None:
                        if log_acc:
                            acc_str = 'at {}, '.format(
                                    unit_time.strftime('%Y%m%dT%H%M'))
                            acc_str += 'closing {}:{}-{}'.format(
                                role[0],
                                self.interval_dfs[role].iloc[irow].start.strftime('%Y%m%d'),
                                self.interval_dfs[role].iloc[irow].end.strftime('%Y%m%d')
                            )
                            acc_str += f', {len(accumulators)} intervals open'
                            self.logger.warning(acc_str)
                        self._close(role,irow,acc,nfold,nlh)
            if log_every and (iunit+1) % log_every == 0:
                print_str = [
                    '{}, {}-{}'.format(
                        k[0][0],
                        self.interval_dfs[k[0]].iloc[k[1]].start.strftime('%Y%m%d'),
                        self.interval_dfs[k[0]].iloc[k[1]].end.strftime('%Y%m%d')
                    ) 
                    for k in accumulators.keys()
                ]
                self.logger.warning(
                    'at {}, '.format(unit_time.strftime('%Y%m%dT%H%M'))+\
                    f'accumulating: {print_str}'
                )
        if accumulators:
            self.logger.warning(f'{len(accumulators)} intervals were never closed!')
        self.logger.info(f'{self.name}: loaded {n_loaded} units, {n_failed} failed')
    
    def _init_frames(self,data,nfold,nlh):
        '''allocate output frames once the grid is known from the first unit'''
        self.xgrid,self.ygrid = data['xgrid'],data['ygrid']
        ny,nx = data['w'].shape
        nfield = len(self.field_names)
        for role,table in self.interval_dfs.items():
            nfold_role = 1 if role == 'infer' else nfold # infer uses all dates
            f = np.full((nfold_role,len(table)*nlh,nfield+2,ny,nx),np.nan,dtype=np.float32)
            f[:,:,nfield:] = 0. # no weight/samples until closed with data
            self.frames[role] = f
    
    @staticmethod
    def _new_accumulator(data,ngroup,nlh):
        '''partial sums per val_fold group and local hour, float64'''
        nfield,ny,nx = data['v'].shape
        return dict(
            wv=np.zeros((ngroup,nlh,nfield,ny,nx)), # sum of w*v
            w=np.zeros((ngroup,nlh,ny,nx)), # sum of w
            n=np.zeros((ngroup,nlh,ny,nx)) # sum of n
        )
    
    @staticmethod
    def _add(acc,data,group,ilh):
        '''add one unit to its val_fold group and local hour'''
        w = data['w'][None] # [1,ny,nx]
        acc['wv'][group,ilh] += w*data['v']
        acc['w'][group,ilh] += data['w']
        acc['n'][group,ilh] += data['n']
    
    def _close(self,role,irow,acc,nfold,nlh):
        '''
        write one interval's frames for all folds and release its accumulator.
        train fold k sums groups other than k+1 (incl. -1 units); val fold k is 
        group k+1; infer has one fold summing all groups, i.e., all dates. 
        explicit sums, no total-minus-partial cancellation
        '''
        nfield = len(self.field_names)
        ngroup = nfold+1
        nfold_role = 1 if role == 'infer' else nfold
        for k in range(nfold_role):
            # group index is fold index + 1
            if role == 'train':
                groups = [g for g in range(ngroup) if g != k+1]
            elif role == 'val':
                groups = [k+1]
            else:
                groups = list(range(ngroup))
            # sum out group dimension: ngroup,nlh,nfield,ny,nx->nlh,nfield,ny,nx
            # where only train groups and val group are summed
            wv = acc['wv'][groups].sum(axis=0)
            w = acc['w'][groups].sum(axis=0)
            w = np.broadcast_to(w[:,None,:,:],wv.shape)
            with np.errstate(invalid='ignore',divide='ignore'):
                mean = np.where(w > 0,wv/w,np.nan)
            sl = slice(irow*nlh,(irow+1)*nlh)
            # push to the frame slot: fold k, all nlh for the row in interval_df
            self.frames[role][k,sl,:nfield] = mean
            self.frames[role][k,sl,nfield] = acc['w'][groups].sum(axis=0)
            self.frames[role][k,sl,nfield+1] = acc['n'][groups].sum(axis=0)


class CropSampler:
    '''
    a crop is identifed as (iframe,i,j), its top-left 
    pixel (i,j) in a frame of shape [frame_y,frame_x]
    mode:
        'fixed': regular grid from (0,0) with stride, plus a last crop at the 
            frame edge so every pixel is covered; for validation
        'random': per frame, a random start in [0,stride) and a random subset of
            int(n*crop_fraction) crops; for training, redrawn each epoch
    the random generator of each frame is seeded by (seed,epoch,iframe), so the 
    crops of an epoch are the same after restarting from a checkpoint
    '''
    def __init__(
        self,frame_y,frame_x,crop_y=64,crop_x=64,stride_y=8,stride_x=8,
        mode='random',seed=0
    ):
        if crop_y > frame_y or crop_x > frame_x:
            raise ValueError(
                f'crop {crop_y}x{crop_x} is larger than frame {frame_y}x{frame_x}!')
        if mode not in ['fixed','random']:
            raise ValueError(f'mode must be fixed or random, not {mode}!')
        self.frame_y,self.frame_x = frame_y,frame_x
        self.crop_y,self.crop_x = crop_y,crop_x
        self.stride_y,self.stride_x = stride_y,stride_x
        self.mode = mode
        self.seed = seed
    
    @staticmethod
    def get_starts(frame_len,crop_len,stride,start=0,cover_edge=False):
        '''crop start indices along one axis; cover_edge appends frame_len-crop_len'''
        starts = np.arange(start,frame_len-crop_len+1,stride,dtype=int)
        if cover_edge and starts[-1] != frame_len-crop_len:
            starts = np.append(starts,frame_len-crop_len)
        return starts
    
    def get_frame_crops(self,start_y=0,start_x=0,cover_edge=False):
        '''all (i,j) of one frame, [n,2]'''
        ii = self.get_starts(self.frame_y,self.crop_y,self.stride_y,start_y,cover_edge)
        jj = self.get_starts(self.frame_x,self.crop_x,self.stride_x,start_x,cover_edge)
        # no crop available, return the right size to concatenate
        if len(ii) == 0 or len(jj) == 0:
            return np.empty((0,2),dtype=int)
        return np.array([(i,j) for i in ii for j in jj])
    
    def get_crops(self,nframe,epoch=0,crop_fraction=1.):
        '''
        returns:
            [ncrop,3] int array of (iframe,i,j), frame by frame
        '''
        if self.mode == 'fixed':
            ijs = self.get_frame_crops(cover_edge=True)
            iframes = np.repeat(np.arange(nframe),len(ijs))
            return np.column_stack([iframes,np.tile(ijs,(nframe,1))])
        crops = []
        for iframe in range(nframe):
            rng = np.random.default_rng([self.seed,epoch,iframe])
            start_y,start_x = rng.integers(self.stride_y),rng.integers(self.stride_x)
            ijs = self.get_frame_crops(start_y,start_x,cover_edge=False)
            if crop_fraction < 1:
                keep = rng.choice(
                    len(ijs),size=int(len(ijs)*crop_fraction),replace=False)
                ijs = ijs[np.sort(keep)]
            crops.append(np.column_stack([np.full(len(ijs),iframe),ijs]))
        return np.concatenate(crops,axis=0)


class RandomError:
    '''
    random error of the first var (the DD field) of a crop, in loss units 
    (multiplied by its var_scale), used to normalize the L1 loss
    method:
        'clt': sigma0/sqrt(mean num_samples over valid pixels). sigma0 is the 
            error at one layer of coverage (num_samples=1), estimated beforehand 
            for this instrument and grid, in physical units
        'xy_rs': 0.5*std(xy-rs) over valid pixels, times (a+b*grid_size), as v04
    fallback:
        returned, in loss units, when the estimate is not finite
    '''
    def __init__(self,method='clt',sigma0=None,xy_rs_coeffs=(1.02,18.21),fallback=1e3):
        if method not in ['clt','xy_rs']:
            raise ValueError(f'random error method must be clt or xy_rs, not {method}!')
        if method == 'clt' and sigma0 is None:
            raise ValueError('sigma0 is needed for the clt random error!')
        self.method = method
        self.sigma0 = sigma0
        self.xy_rs_coeffs = tuple(xy_rs_coeffs)
        self.fallback = float(fallback)
    
    def __call__(self,valid_mask,scale,num_samples=None,xy=None,rs=None,grid_size=None):
        '''
        valid_mask,num_samples,xy,rs:
            [H,W] tensors of one crop; xy/rs in physical units
        scale:
            var_scale of the DD field
        returns:
            scalar tensor
        '''
        if self.method == 'clt':
            sigma = self.sigma0/torch.sqrt(num_samples[valid_mask].mean())*scale
        else:
            a,b = self.xy_rs_coeffs
            sigma = 0.5*torch.std(xy[valid_mask]-rs[valid_mask])*(a+b*grid_size)*scale
        if not torch.isfinite(sigma):
            sigma = torch.tensor(self.fallback)
        return sigma.float()


class CropDataset(Dataset):
    '''
    a thin view of one (stream, role, fold), optionally in a window of the stream 
    domain, so one stream can host several datasets, e.g., validation subdomains
    stream:
        a loaded Stream; uses frames[role][ifold], channel_names, get_frame_df(role), 
        xgrid, ygrid
    specs:
        list of (interval,offset) of the stream role to use, e.g., 
        [('91D','0D'),('91D','30D')]; None uses all. they must be next to each 
        other in the stream's specs, so the frames are a slice, not a copy
    west,east,south,north:
        window, inclusive on pixel centers like popy trim; None means stream edge
    var_names,var_scales:
        fields returned in loss units; the first is the DD field
    random_error:
        RandomError, or its kwargs
    cdl:
        optional CDL with xgrid, ygrid, df (yearly periods), data [nyear,C,H,W],
        matched to frames by the year of the interval midpoint, nearest otherwise
    crop_kw:
        CropSampler kwargs except frame shape, which is determined from self.frames
    min_valid_pixels:
        crops with fewer valid pixels are skipped
    min_num_samples:
        valid pixel: num_samples > min_num_samples and total_sample_weight > 0;
        positive weight guarantees finite Stream means
    wesnmost:
        global bounds normalizing lon/lat, the same for all streams and grids
    jitter_kw:
        None for no jitter, or {'p_jitter':1.,'lh_span':1.}; the fractional year 
        span is the interval length of each frame
    '''
    def __init__(
        self,stream,role='train',ifold=0,specs=None,
        west=None,east=None,south=None,north=None,
        var_names=['column_amount_DD','column_amount','surface_altitude_DD'],
        var_scales=[1e9,-1e4,-1e6],random_error=None,cdl=None,crop_kw=None,
        min_valid_pixels=3,min_num_samples=0.7,
        wesnmost=(-128,-65,24,50),jitter_kw=None
    ):
        self.logger = logging.getLogger(__name__)
        self.var_names,self.var_scales = list(var_names),list(var_scales)
        if isinstance(random_error,RandomError):
            self.random_error = random_error
        else:
            self.random_error = RandomError(**(random_error or {}))
        self.min_valid_pixels = min_valid_pixels
        self.jitter_kw = jitter_kw
        self.do_jitter = jitter_kw is not None
        
        # window as index slices of the stream grid
        xgrid,ygrid = np.asarray(stream.xgrid),np.asarray(stream.ygrid)
        self.grid_size = float(np.abs(np.median(np.diff(xgrid))))
        self.psf_idx = stream.psf_idx
        xs = self.get_slice(xgrid,west,east)
        ys = self.get_slice(ygrid,south,north)
        self.xgrid,self.ygrid = xgrid[xs],ygrid[ys]
        
        # frames of this fold, specs, and window, a view of stream.frames without copy
        frame_df = stream.get_frame_df(role)
        fs = self.get_frame_slice(frame_df,specs)
        self.frames = stream.frames[role][ifold][fs,:,ys,xs] # [nframe,C,H,W]
        self.nframe = self.frames.shape[0]
        self.channel_idx = {c:i for i,c in enumerate(stream.channel_names)}
        needed = self.var_names+['total_sample_weight','num_samples']
        if self.random_error.method == 'xy_rs':
            needed += [f'{self.var_names[0]}_xy',f'{self.var_names[0]}_rs']
        missing = [c for c in needed if c not in self.channel_idx]
        if missing:
            raise ValueError(f'channels {missing} are not in the stream!')
        # valid mask frames, [nframe,H,W]
        self.valid_mask = (
            self.frames[:,self.channel_idx['num_samples']] > min_num_samples
        ) & (
            self.frames[:,self.channel_idx['total_sample_weight']] > 0
        ) 
        
        # raw time of each frame, for TemporalEncoder in the model
        self.frame_df = frame_df.iloc[fs].reset_index(drop=True)
        if len(self.frame_df) != self.nframe:
            raise ValueError('frame table does not match stream frames!')
        
        # space: normalized lon/lat and grid inverse area (m-2, times DD scale)
        westmost,eastmost,southmost,northmost = wesnmost
        lonmesh,latmesh = np.meshgrid(self.xgrid,self.ygrid)
        self.xmesh = torch.tensor((lonmesh-westmost)/(eastmost-westmost)).float()
        self.ymesh = torch.tensor((latmesh-southmost)/(northmost-southmost)).float()
        self.lonmesh,self.latmesh = lonmesh,latmesh
        self.gia = torch.tensor(
            self.var_scales[0]/(np.cos(np.deg2rad(latmesh))*np.square(self.grid_size*111e3))
        ).float()
        
        # cdl in the same window, matched to frames by year
        self.cdl_data = None
        if cdl is not None:
            cdl_ds = cdl.trim(
                west=self.xgrid[0]-self.grid_size/2,
                east=self.xgrid[-1]+self.grid_size/2,
                south=self.ygrid[0]-self.grid_size/2,
                north=self.ygrid[-1]+self.grid_size/2
            )
            assert np.array_equal(
                np.float32(cdl_ds.xgrid),np.float32(self.xgrid)
            ) and np.array_equal(
                np.float32(cdl_ds.ygrid),np.float32(self.ygrid)
            ), 'cdl grid does not match dataset grid!'
            self.cdl_data = torch.tensor(cdl_ds.data).float()
            cdl_years = np.asarray(cdl_ds.df.index.year)
            frame_years = pd.DatetimeIndex(self.frame_df['mid']).year.values
            self.frame_df['icdl'] = np.abs(
                frame_years[:,None]-cdl_years[None,:]).argmin(axis=1)
        
        # crops
        crop_kw = dict(crop_kw or {})
        crop_kw.setdefault('mode','random' if role == 'train' else 'fixed')
        self.sampler = CropSampler(
            frame_y=len(self.ygrid),frame_x=len(self.xgrid),**crop_kw)
        self.set_epoch(0)
    
    @staticmethod
    def get_frame_slice(frame_df,specs=None):
        '''slice of frames whose (interval,offset) is in specs'''
        if specs is None:
            return slice(None)
        # compare parsed specs, so '91d' and '91D' are the same
        wanted = [
            (Stream.parse_interval(i),Stream.parse_interval(o)) for i,o in specs]
        frame_specs = [
            (Stream.parse_interval(i),Stream.parse_interval(o)) 
            for i,o in zip(frame_df['interval'],frame_df['offset'])
        ]
        idx = np.array([i for i,fsp in enumerate(frame_specs) if fsp in wanted])
        if len(idx) == 0:
            raise ValueError(f'specs {specs} are not in the stream!')
        if np.any(np.diff(idx) != 1):
            raise ValueError(
                f'specs {specs} are not next to each other in the stream specs!')
        return slice(idx[0],idx[-1]+1)
    
    @staticmethod
    def get_slice(grid,low=None,high=None):
        '''index slice of grid centers in [low,high], like popy trim'''
        mask = np.ones(len(grid),dtype=bool)
        if low is not None:
            mask &= grid >= low
        if high is not None:
            mask &= grid <= high
        idx = np.where(mask)[0]
        if len(idx) == 0:
            raise ValueError(f'window [{low},{high}] has no grid point!')
        return slice(idx[0],idx[-1]+1)
    
    def set_epoch(self,epoch,crop_fraction=1.):
        '''draw crops for an epoch and skip crops with too few valid pixels'''
        crops = self.sampler.get_crops(self.nframe,epoch,crop_fraction)
        crop_y,crop_x = self.sampler.crop_y,self.sampler.crop_x
        nvalid = np.zeros(len(crops),dtype=int)
        for iframe in np.unique(crops[:,0]):
            # integral image of one frame, so every crop sum is 4 lookups
            sat = np.pad(self.valid_mask[iframe].cumsum(0).cumsum(1),((1,0),(1,0)))
            rows = np.where(crops[:,0] == iframe)[0]
            i,j = crops[rows,1],crops[rows,2]
            nvalid[rows] = sat[i+crop_y,j+crop_x]-sat[i,j+crop_x]-sat[i+crop_y,j]+sat[i,j]
        keep = nvalid >= self.min_valid_pixels
        self.crops = crops[keep]
        self.nvalid = nvalid[keep]
        self.logger.info(
            f'epoch {epoch}: {len(self.crops)} crops, {(~keep).sum()} skipped')
    
    def __len__(self):
        return len(self.crops)
    
    def __getitem__(self,idx):
        iframe,i,j = (int(v) for v in self.crops[idx])
        win = (slice(i,i+self.sampler.crop_y),slice(j,j+self.sampler.crop_x))
        frame = self.frames[iframe]
        valid_mask = torch.from_numpy(self.valid_mask[iframe][win].copy())
        
        def crop(name):
            return torch.from_numpy(frame[self.channel_idx[name]][win].copy())
        
        out = dict(
            spatial=torch.stack([self.xmesh[win],self.ymesh[win]],dim=0),
            time=self.get_time(iframe),
            valid_mask=valid_mask.unsqueeze(0),
            gia=self.gia[win].unsqueeze(0),
            grid_size=torch.tensor(self.grid_size).float(),
            # psf of the stream; center latitude widens the psf in degree lon
            psf_idx=torch.tensor(self.psf_idx),
            lat=torch.tensor(self.ygrid[win[0]].mean()).float(),
            crop=torch.tensor([iframe,i,j])
        )
        for name,scale in zip(self.var_names,self.var_scales):
            # frames are nan where invalid; where() keeps nan out of the loss
            value = crop(name)*scale
            out[name] = torch.where(valid_mask,value,torch.zeros_like(value)).unsqueeze(0)
        name0 = self.var_names[0]
        if self.random_error.method == 'clt':
            out['random_error'] = self.random_error(
                valid_mask,self.var_scales[0],num_samples=crop('num_samples'))
        else:
            out['random_error'] = self.random_error(
                valid_mask,self.var_scales[0],xy=crop(f'{name0}_xy'),
                rs=crop(f'{name0}_rs'),grid_size=self.grid_size)
        if self.cdl_data is not None:
            out['cdl'] = self.cdl_data[self.frame_df['icdl'].iloc[iframe],:,win[0],win[1]]
        return out
    
    def get_time(self,iframe):
        '''raw time [t,local_hour], jittered within the frame span if enabled'''
        row = self.frame_df.iloc[iframe]
        time = torch.tensor([row['t'],row['local_hour']]).float()
        if not self.do_jitter:
            return time
        span = torch.tensor([row['t_span'],self.jitter_kw.get('lh_span',0.)]).float()
        return jitter_time(time,span,p_jitter=self.jitter_kw.get('p_jitter',1.))
    
    def plot_crop(
        self,ax=None,idx=0,field='column_amount',ichannel=0,**kwargs
    ):
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(12,6),
                                  subplot_kw={"projection": ccrs.PlateCarree()})
        else:
            fig = ax.figure
        x0 = self.lonmesh[0,0];x1 = self.lonmesh[0,-1]
        y0 = self.latmesh[0,0];y1 = self.latmesh[-1,0]
        ax.plot([x0,x0,x1,x1,x0],[y0,y1,y1,y0,y0],'-k')
        
        iframe,i,j = (int(v) for v in self.crops[idx])
        
        xmesh = self.lonmesh[i:i+self.sampler.crop_y,j:j+self.sampler.crop_x]
        ymesh = self.latmesh[i:i+self.sampler.crop_y,j:j+self.sampler.crop_x]

        pc = ax.pcolormesh(
            xmesh,ymesh,
            self[idx][field][ichannel],**kwargs
        )
        fig.colorbar(pc,ax=ax,shrink=0.5)
        ax.add_feature(
            cfeature.STATES.with_scale('50m'),edgecolor='k',linewidth=.5)


def build_streams(config,roles=('train','val','infer'),**load_kw):
    '''
    create and load each stream used by an enabled dataset of these roles. a role 
    of a stream is aggregated only if an enabled dataset needs it
    config:
        Config
    load_kw:
        passed to Stream.load, e.g., log_every, log_acc
    returns:
        dict of stream name -> loaded Stream
    '''
    data = config.data
    streams = {}
    for sc in data.streams:
        needed = [
            role for role in roles
            if any(ds.stream == sc.name and ds.role == role and ds.get('enabled',True)
                   for ds in data.datasets)
        ]
        if not needed:
            continue
        stream = Stream(
            name=sc.name,start=sc.start,end=sc.end,local_hours=sc.local_hours,
            path_pattern=sc.path_pattern,base_year=data.base_year,split_kw=data.split,
            psf_idx=sc.get('psf_idx',0),
            train_kw=sc.get('train') if 'train' in needed else None,
            val_kw=sc.get('val') if 'val' in needed else None,
            infer_kw=sc.get('infer') if 'infer' in needed else None
        )
        if config.experiment.interactive:
            # grid size from the domain, approximately; the exact grid comes with load
            nx = int(round((sc.domain.east-sc.domain.west)/sc.grid_size))
            ny = int(round((sc.domain.north-sc.domain.south)/sc.grid_size))
            memory = stream.estimate_memory(nfield=len(sc.field_names),ny=ny,nx=nx)
            logging.getLogger(__name__).warning(
                f'{sc.name} ({", ".join(needed)}): about {memory["frames"]:.2f} GB of '
                f'frames and {memory["accumulator"]:.2f} GB of accumulators')
        stream.load(
            field_names=sc.field_names,west=sc.domain.west,east=sc.domain.east,
            south=sc.domain.south,north=sc.domain.north,grid_size=sc.grid_size,
            **load_kw
        )
        streams[sc.name] = stream
    return streams


def build_cdl(config):
    '''
    one cdl for all datasets: a file per year over a domain covering all 
    streams, combined to name_codes. each CropDataset trims it to its own grid
    returns:
        CDL, or None if cdl is disabled
    '''
    cc = config.data.cdl
    if not cc.enabled:
        return None
    from CAREER.gridded import CDL
    filenames = [pd.Timestamp(y,1,1).strftime(cc.path_pattern) for y in cc.years]
    missing = [f for f in filenames if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f'cdl files not found: {missing}')
    cdl = CDL(lcc_path=cc.lcc_path)
    cdl.read_nc(filenames,cc.years)
    if not cc.cafo.enabled:
        return cdl.combine(cc.name_codes)
    else:
        ngroup = len(cc.name_codes)
        ncafo = len(cc.cafo.paths)
        cdl = cdl.combine_cafo(cc.name_codes,cc.cafo.paths)
        cdl.data[:,ngroup:ngroup+ncafo] *= cc.cafo.scaling
        return cdl


def build_datasets(config,streams,cdl=None,ifold=0,role=None):
    '''
    a CropDataset for each enabled dataset entry (of a role, if given) whose 
    stream is in streams and has frames of its role. infer datasets always use 
    fold 0, i.e., all dates
    returns:
        dict of dataset name -> CropDataset
    '''
    data = config.data
    wesn = data.wesnmost
    datasets = {}
    for dc in data.datasets:
        if not dc.get('enabled',True) or dc.stream not in streams:
            continue
        if role is not None and dc.role != role:
            continue
        if dc.role not in streams[dc.stream].frames: # role not aggregated
            continue
        sc = config.get_stream(dc.stream)
        datasets[dc.name] = CropDataset(
            streams[dc.stream],role=dc.role,
            ifold=0 if dc.role == 'infer' else ifold,
            specs=dc.specs,**(dc.window or {}),
            var_names=sc.var_names,var_scales=sc.var_scales,
            random_error=sc.random_error,cdl=cdl,
            # crops follow the experiment seed, so seeds of an ensemble differ
            crop_kw=dict(dc.crop,seed=dc.crop.get('seed',config.experiment.seed)),
            min_valid_pixels=dc.min_valid_pixels,min_num_samples=sc.min_num_samples,
            wesnmost=(wesn.westmost,wesn.eastmost,wesn.southmost,wesn.northmost),
            jitter_kw=dc.get('jitter')
        )
    return datasets
