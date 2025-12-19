from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import time 
import numpy as np
import pybullet as p
import pybullet_data
import yaml 

@dataclass # TODO: learn this 
class Action:
    throttle: float # [-1,1]
    steer: float # [-1,1]


class BulletEnv: 
    """
    Day-1 env:
      - plane + obstacles + target marker + object marker
      - reset() stable
      - step(action) advances sim and returns observation dict
      - robot: simple car from pybullet_data (racecar)
    """

    def __init__(self, task_cfg_path:str, timing_cfg_path:str):
        self.task_cfg = self._load_yaml(task_cfg_path)
        self.timing_cfg = self._load_yaml(timing_cfg_path)

        self.gui = bool(self.task_cfg["sim"].get("gui", True))
        self.dt = float(self.task_cfg["sim"].get("timestep", 0.02)) # TODO we had defined earlier? 
        self.gravity = float(self.task_cfg["sim"].get("gravity",-9.81))

        self.control_hz = float(self.timing_cfg["control"].get("hz", 20))
        self.max_steps = int(self.timing_cfg["control"].get("max_steps_per_episode",2000))

        self.cid: Optional[int] = None
        self.robot_id: Optional[int] = None
        self.target_id: Optional[int] = None
        self.object_id: Optional[int] = None
        self.obstacle_ids = []
        self.step_count = 0 
        self._was_in_collision = False

        # Racecar control joints in pybullet example model
        # TODO: why did we decide them like this ?? 
        self.steering_joints = [0, 2] # front wheels steering 
        self.drive_joints = [8, 15] # rear wheels motor joints (common in this model)


    def connect(self):
        if self.cid is not None:
            return
        self.cid = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.gravity)
        p.setTimeStep(self.dt)

        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(cameraDistance=3.0, 
                                         cameraYaw=45, 
                                         cameraPitch=-35, 
                                         cameraTargetPosition=[0.5, 0, 0]
                                         )
    
    def close(self):
        if self.cid is not None:
            p.disconnect(self.cid)
        self.cid = None


    def reset(self, seed:Optional[int]=None) -> Dict[str, Any]:
        self.connect()
        
        if seed is not None: 
            np.random.seed(seed)
        
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,self.gravity)
        p.setTimeStep(self.dt)

        self.step_count = 0 
        self.obstacle_ids = [] 

        # floor 
        p.loadURDF("plane.urdf")

       # robot 
        self.robot_id = p.loadURDF(
            "racecar/racecar.urdf",
            basePosition=[0,0,0.15],  # slightly higher spawn helps too
            baseOrientation=p.getQuaternionFromEuler([0,0,0]),
        )

        self._setup_racecar_joints()

        # SET DAMPING & FRICTION

        num_joints = p.getNumJoints(self.robot_id)

        # 1. base link
        p.changeDynamics(
            self.robot_id,
            -1,
            lateralFriction=1.2,
            angularDamping=0.8,
            linearDamping=0.04,
        )

        # 2. all joints (especially wheels)
        for j in range(num_joints):
            p.changeDynamics(
                self.robot_id,
                j,
                lateralFriction=1.2,
                angularDamping=0.8,
                linearDamping=0.04,
            )

        # obstacles 
        for ob in self.task_cfg["world"].get("obstacles", []):
            self.obstacle_ids.append(self._spawn_box(
                ob["pos"], ob["size"], rgba=[0.7, 0.2, 0.2, 1]
            ))

        # target marker (visual, non-colliding)
        tgt_pos = self.task_cfg["world"]["target"]["pos"]
        self.target_id = self._spawn_marker(tgt_pos, radius=0.07, rgba=[0.2,0.8,0.2,1])

        # object marker (visual, optional for later)
        obj_pos = self.task_cfg["world"]["object"]["pos"]
        self.object_id = self._spawn_marker(obj_pos, radius=0.06, rgba=[0.2, 0.2, 0.9, 1])

        # settle a few frames to make reset reliable
        for _ in range(10):
            p.stepSimulation()
            if self.gui:
                time.sleep(self.dt)
        
        return self._get_obs()
    
    def _setup_racecar_joints(self):
        assert self.robot_id is not None

        n = p.getNumJoints(self.robot_id)
        name_by_idx = {}
        for j in range(n):
            info = p.getJointInfo(self.robot_id, j)
            name = info[1].decode("utf-8")
            name_by_idx[j] = name

        # Print once to learn what your URDF actually has
        print("\n[Racecar joints]")
        for j, name in name_by_idx.items():
            print(f"  {j:2d}: {name}")

        # Try common pybullet racecar naming patterns
        steering = []
        drive = []

        for j, name in name_by_idx.items():
            lname = name.lower()
            # steering joints often contain "steer"
            if "steer" in lname:
                steering.append(j)
            # wheel joints often contain "wheel"
            if "wheel" in lname:
                drive.append(j)

        # Fallback if no names match (keeps you moving)
        if len(steering) == 0:
            steering = [0, 2] if n > 2 else list(range(min(2, n)))
        if len(drive) == 0:
            # try last few joints as wheels
            drive = list(range(max(0, n - 4), n))

        self.steering_joints = steering[:2]  # usually 2 steering joints
        self.drive_joints = drive[:2]        # keep 2 for now, we can expand later

        print(f"[Selected] steering_joints={self.steering_joints}, drive_joints={self.drive_joints}\n")

    def step(self, action:Action) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        assert self.robot_id is not None, "Call reset() before step()."

        self.step_count +=1

        # clamp actions 
        throttle = float(np.clip(action.throttle, -1.0, 1.0))
        steer = float(np.clip(action.steer, -1.0, 1.0))

        # deadzones to reduce micro-jitter
        if abs(throttle) <= 0.05:
            throttle = 0.0
        if abs(steer) <= 0.05:
            steer = 0.0

        self._apply_car_control(throttle=throttle, steer=steer)
        
        # step sim at env timestep
        p.stepSimulation()
        if self.gui:
            time.sleep(self.dt)

        obs = self._get_obs()

        # day-1: no-reward logic yet
        reward = 0.0 
        done = self.step_count >= self.max_steps
        
        info = {
            "step":self.step_count,
            "collision":self._check_collision_event(),
        }

        return obs, reward, done, info
    

    # ----------------- helpers -----------------
    def _apply_car_control(self, throttle:float, steer:float):
        # steering: set joint positions (small angle)
        max_steer = 0.6 # rad
        steer_angle = max_steer * steer

        n = p.getNumJoints(self.robot_id)

        for j in self.steering_joints:
            if j < 0 or j >= n:
                continue
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=steer_angle,
                force=50,
            )
        
        # throttle: wheel velocity control 
        max_vel = 30.0 
        target_vel = max_vel * throttle

        for j in self.drive_joints:
            if j < 0 or j >= n:
                continue
            p.setJointMotorControl2(
                bodyUniqueId= self.robot_id,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=target_vel,
                force=200,
            )
    
    def _get_obs(self) -> Dict[str, Any]:
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)

        target_pos, _ = p.getBasePositionAndOrientation(self.target_id)
        object_pos, _ = p.getBasePositionAndOrientation(self.object_id)

        return {
            "robot": {
                "pos": np.array(base_pos, dtype=np.float32),
                "orn_xyzw": np.array(base_orn, dtype=np.float32),
                "lin_vel": np.array(lin_vel, dtype=np.float32),
                "ang_vel": np.array(ang_vel, dtype=np.float32),
            },
            "target": {"pos":np.array(target_pos, dtype=np.float32)},
            "object": {"pos":np.array(object_pos, dtype=np.float32)},
        }
    

    def _check_collision_event(self) -> bool:
        """
        Returns True only when a new collision starts.
        """
        in_contact = False
        for oid in self.obstacle_ids:
            if len(p.getContactPoints(self.robot_id, oid)) > 0:
                in_contact = True
                break

        collision_event = in_contact and not self._was_in_collision
        self._was_in_collision = in_contact
        return collision_event
        """if self.robot_id is None:
            return False
        for oid in self.obstacle_ids:
            if len(p.getContactPoints(self.robot_id, oid)) > 0:
                return True
        return False"""

    
    def _spawn_box(self, pos, size, rgba): 
        half = [size[0]/2, size[1]/2, size[2]/2]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
        bid = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
        )
        return bid
    
    def _spawn_marker(self, pos, radius, rgba):
        # purely visual sphere (no collision)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
        mid = p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=vis, basePosition=pos)
        return mid
    
    @staticmethod
    def _load_yaml(path:str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)