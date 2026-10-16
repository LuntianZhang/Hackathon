"""Select unique views at uniform azimuths around the observed camera orbit."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def orbit_coordinates(poses):
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 3:
        raise ValueError('Need at least three camera poses')
    centers = poses[:, :3, 3]
    rays = poses[:, :3, 2]
    planes = np.eye(3)[None] - rays[:, :, None]*rays[:, None, :]
    # Closest common point of the optical axes, rather than the trajectory mean.
    pivot = np.linalg.lstsq(planes.sum(axis=0), np.einsum('nij,nj->i', planes, centers), rcond=None)[0]
    _, singular, basis = np.linalg.svd(centers-centers.mean(axis=0), full_matrices=False)
    if singular[1] < 1e-8:
        raise ValueError('Camera positions are collinear; cannot infer an orbit plane')
    xy = (centers-pivot)@basis[:2].T
    angles = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2*np.pi)
    sorted_angles = np.sort(angles)
    gaps = np.diff(np.r_[sorted_angles, sorted_angles[0]+2*np.pi])
    gap_index = int(np.argmax(gaps))
    start = float(sorted_angles[(gap_index+1) % len(angles)])
    largest_gap = float(gaps[gap_index])
    # Use an open arc if the input does not cover a whole turn.
    closed = bool(largest_gap <= np.radians(30))
    span = 2*np.pi if closed else 2*np.pi-largest_gap
    return {'pivot':pivot, 'basis':basis, 'xy':xy, 'angles':angles,
            'start':start, 'span':float(span), 'closed':closed,
            'largest_gap':largest_gap, 'singular_values':singular}


def select_uniform_views(orbit, count, available=None, midpoints=False):
    angles = orbit['angles']
    available = np.arange(len(angles)) if available is None else np.asarray(available, dtype=int)
    if count < 2 or count > len(available):
        raise ValueError('Not enough unique views for the requested count')
    if midpoints:
        targets=orbit['start']+(np.arange(count)+.5)*orbit['span']/count
    else:
        targets = orbit['start'] + np.linspace(0, orbit['span'], count, endpoint=not orbit['closed'])
    delta = np.abs(np.angle(np.exp(1j*(targets[:, None]-angles[available][None]))))
    rows, columns = linear_sum_assignment(delta)
    indices = available[columns]
    error = delta[rows, columns]
    relative = np.mod(angles[indices]-orbit['start'], 2*np.pi)
    return indices.tolist(), {'target_degrees':np.degrees(targets-orbit['start']).tolist(),
        'actual_degrees':np.degrees(relative).tolist(),
        'max_angular_error_degrees':float(np.degrees(error.max())),
        'mean_angular_error_degrees':float(np.degrees(error.mean()))}


def select_common_holdout(orbit, used, count=12, minimum_degrees=1.5):
    used=np.asarray(sorted(set(used)),dtype=int)
    distance=np.abs(np.angle(np.exp(1j*(orbit['angles'][:,None]-orbit['angles'][used][None])))).min(axis=1)
    available=np.flatnonzero(np.degrees(distance)>=minimum_degrees-1e-8)
    indices,distribution=select_uniform_views(orbit,count,available,midpoints=True)
    distribution['minimum_distance_to_any_training_view_degrees']=float(np.degrees(distance[indices]).min())
    return indices,distribution


def fit_similarity(source, target):
    """Least-squares proper Sim(3), mapping source centers into target coordinates."""
    source,target=np.asarray(source,dtype=float),np.asarray(target,dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source)<3:
        raise ValueError('Need at least three paired 3D points')
    mean_source,mean_target=source.mean(0),target.mean(0)
    a,b=source-mean_source,target-mean_target
    if np.linalg.matrix_rank(a)<2:raise ValueError('Camera centers are collinear')
    u,s,v=np.linalg.svd(b.T@a/len(a))
    signs=np.ones(3);signs[-1]=np.sign(np.linalg.det(u@v))
    rotation=u@np.diag(signs)@v
    scale=float(np.dot(s,signs)/np.mean(np.sum(a*a,axis=1)))
    translation=mean_target-scale*rotation@mean_source
    error=source@rotation.T*scale+translation-target
    return rotation,scale,translation,float(np.sqrt(np.mean(np.sum(error*error,axis=1))))
