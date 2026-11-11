"""Versioned landing regions in shared (opponent depth, hitter-left lateral) units."""
from dataclasses import dataclass, asdict
import numpy as np

CATEGORIES = ('left','left_short','center_safe','right_short','right','other/unmapped')


@dataclass(frozen=True)
class FourQuadrantsCenterOverride:
    """Complete table partition; central rectangle overrides base quadrants."""
    name: str = 'four_quadrants_center_override'
    depth_split: float = .51
    lateral_split: float = 0.
    center_depth_min: float = .20
    center_depth_max: float = .70
    center_lateral_min: float = -.35
    center_lateral_max: float = .35

    @property
    def categories(self):
        return CATEGORIES[:5]

    def classify(self, depth, lateral):
        d,y=np.broadcast_arrays(np.asarray(depth,float),np.asarray(lateral,float))
        valid=np.isfinite(d)&np.isfinite(y)&(d>=0)&(d<=1)&(np.abs(y)<=1)
        left=y>=self.lateral_split
        deep=d>=self.depth_split
        labels=np.where(left,np.where(deep,'left','left_short'),
                        np.where(deep,'right','right_short')).astype(object)
        center=(d>=self.center_depth_min)&(d<=self.center_depth_max)&(y>=self.center_lateral_min)&(y<=self.center_lateral_max)
        labels=np.where(center,'center_safe',labels)
        return np.where(valid,labels,'invalid/outside')

    def validate(self):
        if not (0<self.depth_split<1 and -1<self.lateral_split<1 and
                0<=self.center_depth_min<self.center_depth_max<=1 and
                -1<=self.center_lateral_min<self.center_lateral_max<=1):
            raise ValueError('Invalid quadrant or center-override bounds')


def region_from_dict(spec):
    cls=FourQuadrantsCenterOverride if spec.get('name')=='four_quadrants_center_override' else Regions
    return cls(**spec)


@dataclass(frozen=True)
class Regions:
    name: str = 'balanced'
    center_lateral: float = .25
    center_depth_min: float = .20
    center_depth_max: float = .65
    side_lateral_min: float = .30
    short_depth_max: float = .45
    deep_depth_min: float = .55

    def classify(self, depth, lateral):
        d,l=np.broadcast_arrays(np.asarray(depth,float),np.asarray(lateral,float))
        result=np.full(d.shape,'other/unmapped',dtype=object)
        valid=np.isfinite(d)&np.isfinite(l)&(d>0)&(d<=1)&(np.abs(l)<=1)
        center=valid&(np.abs(l)<=self.center_lateral)&(d>=self.center_depth_min)&(d<=self.center_depth_max)
        result[center]='center_safe'
        for sign,short,deep in [(1,'left_short','left'),(-1,'right_short','right')]:
            side=valid&(sign*l>=self.side_lateral_min)&~center
            result[side&(d<=self.short_depth_max)]=short
            result[side&(d>=self.deep_depth_min)]=deep
        result[~valid]='invalid/outside'
        return result

    def validate(self):
        if not (0<=self.center_lateral<self.side_lateral_min<=1 and
                0<=self.center_depth_min<self.center_depth_max<=1 and
                0<=self.short_depth_max<=self.deep_depth_min<=1):
            raise ValueError('Invalid or overlapping spatial region bounds')


def definitions():
    return [Regions(),Regions('compact',.20,.25,.55,.35,.40,.60),
            Regions('broad',.35,.15,.70,.40,.50,.50),FourQuadrantsCenterOverride()]


def position_labels(own_lateral_normalized, band_m):
    # Same normalized band for humans (half-width .762 m) and robots (.7625 m).
    y=np.asarray(own_lateral_normalized,float);b=band_m/.7625
    return np.where(~np.isfinite(y),'unknown',np.where(np.abs(y)<=b,'center',np.where(y>0,'left','right')))


def conditional_statistics(frame, regions, band_m=.05):
    import pandas as pd
    rows=[]
    for dataset,f in frame.groupby('dataset'):
        available=bool(f.landing_observed.any())
        p1=position_labels(f.p1_own_lateral,band_m);p2=position_labels(f.p2_own_lateral,band_m)
        known=(p1!='unknown')&(p2!='unknown');total=int(known.sum())
        for a in ('left','center','right'):
            for b in ('left','center','right'):
                sub=f[(p1==a)&(p2==b)];valid=sub[sub.valid_landing.astype('boolean').fillna(False).astype(bool)]
                measured=int(sub.landing_observed.fillna(False).sum())
                labels=regions.classify(valid.depth,valid.lateral);mapped=np.isin(labels,CATEGORIES[:-1]);nm=int(mapped.sum())
                for category in getattr(regions,'categories',CATEGORIES):
                    count=int((labels==category).sum())
                    rows.append(dict(dataset=dataset,p1_position=a,p2_position=b,band_m=band_m,
                        n_attempts=len(sub),n_measured_outcomes=measured,n_valid_landings=len(valid) if available else np.nan,
                        n_mapped=nm if available else np.nan,region=category,region_count=count if available else np.nan,
                        probability_valid=count/len(valid) if len(valid) else np.nan,
                        probability_mapped=count/nm if nm and category!='other/unmapped' else np.nan,
                        context_attempt_frequency=len(sub)/total if total else np.nan,
                        context_attempt_fraction_all=len(sub)/len(f) if len(f) else np.nan,
                        frequency_denominator='known_position_attempts',outcomes_available=available,
                        landing_coverage=len(valid)/len(sub) if len(sub) and measured else np.nan,
                        unknown_context_attempts=int((~known).sum())))
    return pd.DataFrame(rows)
