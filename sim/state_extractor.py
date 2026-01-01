from __future__ import annotations
from typing import Dict, Any, Optional, List
import math 
import numpy as np
import pybullet as p


"""
==================
 logic for state extraction
==================
At Bare Minimum: 
    robot_pose: (x, y, yaw)
    target_dist, target_bearing
    object_dist
    nearest_obstacle_dist

Additionally: 
    front_dist
    left_dist
    right_dist
    at_target
    object_close

Robot: 
    yaw from quaternion
    bearing in [-pi,+pi]

"""


def _wrap_pi(a:float) -> float:
    return (a+math.pi) % (2*math.pi)-math.pi # keep angle inbetween -/+pi


def _yaw_from_xyzw(q_xyzw:np.ndarray) -> float: 
    # pybullet expects (x,y,z,w)
    q = [float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])]
    _,_,yaw = p.getEulerFromQuaternion(q)
    return float(yaw)


def _ray_dists(
        robot_id: int, 
        origin_xyz: np.ndarray,
        yaw: float,
        ray_len: float,
        angles_deg: List[float],
        obstacle_ids: Optional[List[int]] = None,
        )-> List[float]:
    
    x,y,z = float(origin_xyz[0]), float(origin_xyz[1]), float(origin_xyz[2])
    z0 = 0.25 # to avoid negative z


    starts, ends = [], []
    for a in angles_deg:
        ang = yaw + math.radians(a)
        dx, dy = math.cos(ang)*ray_len, math.sin(ang)*ray_len
        starts.append([x, y, z0])
        ends.append([x+dx, y+dy, z0])

    results = p.rayTestBatch(starts, ends)

    dists = []
    for (hit_uid, _, hit_frac, _, _) in results: 
        if hit_uid == -1:
            dists.append(ray_len)
            continue
        if hit_uid == robot_id:
            dists.append(ray_len)
            continue
        #if obstacle_ids is not None and hit_uid not in obstacle_ids:
        #    dists.append(ray_len)
        #    continue
        dists.append(max(0.0, float(hit_frac)*ray_len))
    
    return dists


def extract_symbolic(
        obs: Dict[str, Any],
        *,
        obstacle_ids: Optional[list[int]] = None,
        robot_id: Optional[int] = None,
        ray_len: float = 3.0,
    ) -> Dict[str, Any]:

    # Raw obs -> symbolic state for planning/control.

    r_pos = obs["robot"]["pos"] # shape (3,)
    q = obs["robot"]["orn_xyzw"]  # shape (4,)
    t_pos = obs["target"]["pos"] # shape (3,)
    o_pos = obs["object"]["pos"] # shape (3,) 

    yaw = _yaw_from_xyzw(q)


    # 2D relative vectors on ground plane
    rel_t = (t_pos - r_pos).astype(np.float32)
    rel_o = (o_pos - r_pos).astype(np.float32)

    dx_t, dy_t = float(rel_t[0]), float(rel_t[1])
    dx_o, dy_o = float(rel_o[0]), float(rel_o[1])

    target_dist = math.hypot(dx_t, dy_t) + 1e-9
    object_dist = math.hypot(dx_o, dy_o) + 1e-9

    target_angle = math.atan2(dy_t, dx_t)
    target_bearing = _wrap_pi(target_angle - yaw)

    
    # obstacle distances via rays (front arc)
    angles = [-60.0, -30.0, 0.0, 30.0, 60.0]

    if robot_id is not None:
        dists = _ray_dists(robot_id, 
                           r_pos,
                           yaw,
                           ray_len,
                           angles,
                           obstacle_ids=obstacle_ids
                           )
        nearest = float(min(dists))
        left_dist = float(min(d for a,d in zip(angles, dists) if a>0))
        right_dist = float(min(d for a,d in zip(angles, dists) if a<0))
        front_dist = float(dists[angles.index(0.0)])

    else:
        # Fallback
        # If we can’t raycast, stay safe: pretend things are far but not infinite
        nearest = ray_len
        left_dist = ray_len
        right_dist = ray_len
        front_dist = ray_len
    
    # thresholds 
    # TODO later move to configs/task.yaml
    at_target = target_dist < 0.05 # close enough
    object_close = object_dist < 0.25


    state = {
            "robot_pose": (float(r_pos[0]), float(r_pos[1]), float(yaw)),
            "target_dist": float(target_dist),
            "target_bearing": float(target_bearing),
            "object_dist": float(object_dist),
            "nearest_obstacle_dist": float(nearest),
            "front_dist": float(front_dist),
            "left_dist": float(left_dist),
            "right_dist": float(right_dist),
            "at_target": bool(at_target),
            "object_close": bool(object_close),
        }
    
    
    for k,v in state.items():
        if isinstance(v, tuple):
            for t in v:
                if not math.isfinite(t):
                    raise ValueError(f"Non-Finite {k}:{v}")
        elif isinstance(v, (float, int)):
            if not math.isfinite(float(v)):
                raise ValueError(f"Non-Finite {k}:{v}")

    return state