"""
Generative Adaptive Sampling Module 

"""

import os
import sys
import os.path as osp
import pickle
import joblib
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sRot
import scipy.ndimage.filters as filters

from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES as joint_names
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from smpl_sim.utils.transform_utils import quat_correct


# ----------------------------- Path + import helpers -----------------------------

def _is_mdm_repo(repo_dir: str) -> bool:
    """MDM repo validity check."""
    return osp.exists(osp.join(repo_dir, "utils", "model_util.py"))


def find_mod_root(anchor_file: Optional[str] = None) -> str:
    """
    Walk upward from anchor_file (preferred) or cwd until we find:
        <Mod>/mdm/utils/model_util.py or <Mod>/motion-diffusion-model/utils/model_util.py
    Return <Mod>.
    """
    cur = osp.abspath(anchor_file or os.getcwd())
    if osp.isfile(cur):
        cur = osp.dirname(cur)

    while True:
        # Try both "mdm" and "motion-diffusion-model" directory names
        for repo_name in ["mdm", "motion-diffusion-model"]:
            mdm_repo = osp.join(cur, repo_name)
            if _is_mdm_repo(mdm_repo):
                return cur
        parent = osp.dirname(cur)
        if parent == cur:
            raise FileNotFoundError(
                "Could not locate Mod root containing mdm/ or motion-diffusion-model/. "
                "Expected: <Mod>/mdm/utils/model_util.py or <Mod>/motion-diffusion-model/utils/model_util.py"
            )
        cur = parent


def setup_mdm_repo(anchor_file: str) -> str:
    """
    Ensure Mod/mdm or Mod/motion-diffusion-model is importable.
    Returns absolute path to the MDM repo folder.
    """
    mod_root = find_mod_root(anchor_file)
    # Try both directory names
    for repo_name in ["mdm", "motion-diffusion-model"]:
        mdm_repo = osp.join(mod_root, repo_name)
        if _is_mdm_repo(mdm_repo):
            mdm_repo = osp.abspath(mdm_repo)
            if mdm_repo not in sys.path:
                sys.path.insert(0, mdm_repo)  # makes `from utils...` resolve to MDM's utils
            return mdm_repo
    raise FileNotFoundError(f"Could not find MDM repo in {mod_root}")


@contextmanager
def pushd(path: str):
    """Temporarily chdir into `path` (some MDM utilities assume repo-relative paths)."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# ----------------------------- Main module -----------------------------

class GenerativeAdaptiveSampling:
    """Generative adaptive sampling using MDM text-to-motion model."""

    MUJOCO_JOINT_NAMES = [
        'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle',
        'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder',
        'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand'
    ]
    # SMPL parent indices (24 joints)
    PARENT = [-1, 0, 0, 1, 2, 3, 4, 5, 6, 0, 9, 10, 11, 12, 11, 14, 15, 16, 17, 11, 19, 20, 21, 22]
    DEFAULT_DIR = None

    def __init__(
        self,
        agent,
        motion_lib,
        text_mapping_path: Optional[str] = None,
        text_base_path: Optional[str] = None,
        index_csv_path: Optional[str] = None,
        num_samples_per_failed: int = 3,
        discriminator_threshold: float = 0.5,
        mdm_model_path: Optional[str] = None,
        mdm_repo_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.agent = agent
        self.motion_lib = motion_lib
        self.num_samples_per_failed = num_samples_per_failed
        self.discriminator_threshold = discriminator_threshold
        self.device = device
        self._mdm_repo_path = mdm_repo_path  # Store for later use

        # --- text mapping ---
        self.text_mapping: Dict[str, str] = {}
        if text_mapping_path and osp.exists(text_mapping_path):
            self.text_mapping = pickle.load(open(text_mapping_path, "rb"))
        elif index_csv_path and osp.exists(index_csv_path):
            self._load_text_from_csv(index_csv_path, text_base_path)

        # --- SMPL robot (data dir: Mod/data/smpl) ---
        mod_root = find_mod_root(__file__)
        smpl_data_dir = osp.join(mod_root, "data", "smpl")
        if not osp.exists(smpl_data_dir):
            # keep a fallback, but this should exist in your layout
            smpl_data_dir = "data/smpl"

        robot_cfg = {
            "mesh": False,
            "model": "smpl",
            "upright_start": True,
            "body_params": {},
            "joint_params": {},
            "geom_params": {},
            "actuator_params": {},
        }
        self.smpl_local_robot = LocalRobot(robot_cfg, data_dir=smpl_data_dir)

        self.smpl_2_mujoco = [joint_names.index(n) for n in self.MUJOCO_JOINT_NAMES if n in joint_names]

        # precompute default directions (normalized)
        if self.DEFAULT_DIR is None:
            d = np.array([
                [0, 0, 0], [-0.1, -0.4, 0], [0.1, -0.4, 0], [0, -0.4, 0], [0, -0.4, 0],
                [0, -0.2, 0.1], [0, -0.2, 0.1], [0, 0, 0.1], [0, 0, 0.1],
                [0, 0.2, 0], [0, 0.2, 0], [0, 0.2, 0], [0, 0.15, 0], [0, 0.1, 0],
                [-0.15, 0, 0], [-0.3, 0, 0], [-0.3, 0, 0], [-0.2, 0, 0], [-0.1, 0, 0],
                [0.15, 0, 0], [0.3, 0, 0], [0.3, 0, 0], [0.2, 0, 0], [0.1, 0, 0],
            ], dtype=np.float32)
            self.DEFAULT_DIR = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8)

        # --- init MDM once ---
        self.mdm_repo = None
        self.mdm_talker = None
        if mdm_model_path and osp.exists(mdm_model_path):
            self.mdm_talker = self._init_local_mdm(mdm_model_path, device_in=device)
        else:
            print(f"Warning: mdm_model_path missing or not found: {mdm_model_path}")

    # ----------------------------- MDM -----------------------------

    def _init_local_mdm(self, mdm_model_path: str, device_in: str = "cuda"):
        """
        Load local MDM once and return a wrapper with generate_motion().
        Assumes <Mod>/motion-diffusion-model exists.
        """
        # Find MDM repo - try multiple methods
        mdm_repo = None
        
        # Method 1: Use provided repo path if available
        if hasattr(self, '_mdm_repo_path') and self._mdm_repo_path and osp.exists(self._mdm_repo_path):
            mdm_repo = osp.abspath(self._mdm_repo_path)
        
        # Method 2: Try to find via setup_mdm_repo
        if mdm_repo is None:
            try:
                mdm_repo = setup_mdm_repo(__file__)
            except (FileNotFoundError, Exception) as e:
                print(f"Could not find MDM repo via setup_mdm_repo: {e}")
        
        # Method 3: Try common locations
        if mdm_repo is None or not osp.exists(mdm_repo):
            common_paths = [
                '/home/yhuang/tracking/ModSkill_release/mdm',
                '/home/yhuang/tracking/motion-diffusion-model',
                osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), 'mdm'),
                osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), 'motion-diffusion-model'),
            ]
            for path in common_paths:
                if path and osp.exists(path) and _is_mdm_repo(path):
                    mdm_repo = osp.abspath(path)
                    break
        
        if mdm_repo is None or not osp.exists(mdm_repo):
            raise FileNotFoundError(
                f"Could not find MDM repository. Tried multiple paths. "
                f"Please ensure MDM repo exists or set mdm_repo_path parameter."
            )
        
        if not _is_mdm_repo(mdm_repo):
            raise ValueError(f"Path {mdm_repo} does not appear to be a valid MDM repository")
        
        self.mdm_repo = mdm_repo
        print(f"[MDM] Using repository: {mdm_repo}")
        
        # Import MDM modules - must be done with MDM repo in sys.path and in correct directory
        # We need to temporarily modify sys.path and change directory
        old_cwd = os.getcwd()
        old_sys_path = list(sys.path)
        
        try:
            # Add MDM repo to sys.path if not already there
            mdm_repo_abs = osp.abspath(mdm_repo)
            
            # Remove any existing instances of this path to avoid duplicates
            while mdm_repo_abs in sys.path:
                sys.path.remove(mdm_repo_abs)
            
            # Insert at the front to prioritize MDM modules over any conflicting ones
            sys.path.insert(0, mdm_repo_abs)
            
            # Change to MDM repo directory for relative imports
            # This is important because some MDM code uses relative paths
            os.chdir(mdm_repo_abs)
            
            # Now import MDM modules - they should resolve correctly from the MDM repo
            # The sys.path modification ensures 'model', 'utils', etc. come from MDM repo
            from utils.fixseed import fixseed
            from utils.model_util import create_model_and_diffusion, load_model_wo_clip
            from utils import dist_util
            from model.cfg_sampler import ClassifierFreeSampleModel
            from data_loaders.tensors import collate
            from sample.generate import load_dataset
            from data_loaders.humanml.scripts.motion_process import recover_from_ric
            
            # Store imports for later use (these are already loaded modules)
            self._mdm_imports = {
                'fixseed': fixseed,
                'create_model_and_diffusion': create_model_and_diffusion,
                'load_model_wo_clip': load_model_wo_clip,
                'dist_util': dist_util,
                'ClassifierFreeSampleModel': ClassifierFreeSampleModel,
                'collate': collate,
                'load_dataset': load_dataset,
                'recover_from_ric': recover_from_ric,
            }
            
        except ImportError as e:
            # Restore state before raising
            os.chdir(old_cwd)
            sys.path[:] = old_sys_path
            raise ImportError(
                f"Failed to import MDM modules from {mdm_repo}. "
                f"Make sure the MDM repository is correctly set up. "
                f"Error: {e}\n"
                f"Current sys.path: {sys.path[:5]}..."
            )
        finally:
            # Restore original directory (but keep sys.path modified for now)
            os.chdir(old_cwd)
            # Note: We keep MDM repo in sys.path for later use, but we'll restore it after model loading
        class MDMConfig:
            dataset = "humanml"
            motion_length = 10.0
            batch_size = 1
            guidance_param = 2.5
            device = device_in
            seed = 0
            unconstrained = False
            model_path = mdm_model_path
            # model defaults
            latent_dim = 512
            layers = 8
            arch = "trans_enc"
            emb_trans_dec = False
            cond_mask_prob = 0.1
            # diffusion defaults
            diffusion_steps = 1000
            noise_schedule = "cosine"
            sigma_small = True
            # loss weights
            lambda_vel = 0.0
            lambda_rcxyz = 0.0
            lambda_fc = 0.0

        cfg = MDMConfig()
        
        # Use stored imports
        fixseed = self._mdm_imports['fixseed']
        create_model_and_diffusion = self._mdm_imports['create_model_and_diffusion']
        load_model_wo_clip = self._mdm_imports['load_model_wo_clip']
        dist_util = self._mdm_imports['dist_util']
        ClassifierFreeSampleModel = self._mdm_imports['ClassifierFreeSampleModel']
        collate = self._mdm_imports['collate']
        load_dataset = self._mdm_imports['load_dataset']
        recover_from_ric = self._mdm_imports['recover_from_ric']

        # Load model - must be done in MDM repo directory
        old_cwd = os.getcwd()
        mdm_repo_abs = osp.abspath(mdm_repo)
        try:
            os.chdir(mdm_repo_abs)
            print(f"[MDM] Loading from: {mdm_model_path}")
            fixseed(cfg.seed)
            n_frames = min(196, int(cfg.motion_length * 20))
            dist_util.setup_dist(cfg.device)
            data = load_dataset(cfg, 196, n_frames)
            model, diffusion = create_model_and_diffusion(cfg, data)
            load_model_wo_clip(model, torch.load(cfg.model_path, map_location="cpu"))

            if cfg.guidance_param != 1:
                model = ClassifierFreeSampleModel(model)

            model.to(cfg.device).eval()
            model_kwargs = collate([{"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames}])[1]
        finally:
            os.chdir(old_cwd)
            # Restore sys.path after model loading
            sys.path[:] = old_sys_path

        class MDMWrapper:
            def __init__(self, model, diffusion, data, model_kwargs, n_frames, repo, device, guidance_param):
                self.model = model
                self.diffusion = diffusion
                self.data = data
                self.model_kwargs = model_kwargs
                self.n_frames = n_frames
                self.repo = repo
                self.device = device
                self.guidance_param = guidance_param

            def generate_motion(self, prompt: str) -> np.ndarray:
                """Return joints as (T, 24, 3) in SMPL-like joint positions."""
                # Use the stored recover_from_ric import from initialization
                # We need to be in the MDM repo directory for some operations
                old_cwd = os.getcwd()
                old_sys_path = list(sys.path)
                
                try:
                    # Ensure MDM repo is in sys.path and at the front
                    if self.repo not in sys.path:
                        sys.path.insert(0, self.repo)
                    elif sys.path[0] != self.repo:
                        # Move to front if not already there
                        sys.path.remove(self.repo)
                        sys.path.insert(0, self.repo)
                    
                    os.chdir(self.repo)
                    
                    # Import recover_from_ric - it should be available from the stored imports
                    # But we'll import it here to ensure it's available in the correct context
                    from data_loaders.humanml.scripts.motion_process import recover_from_ric
                    mk = self.model_kwargs.copy()
                    mk["y"] = self.model_kwargs["y"].copy()
                    mk["y"]["text"] = [prompt]
                    mk["y"]["scale"] = torch.ones(1, device=self.device) * self.guidance_param

                    sample = self.diffusion.p_sample_loop(
                        self.model,
                        (1, self.model.njoints, self.model.nfeats, self.n_frames),
                        clip_denoised=False,
                        model_kwargs=mk,
                        progress=False,
                    )

                    if self.model.data_rep == "hml_vec":
                        n_joints = 22 if sample.shape[1] == 263 else 21
                        sample = self.data.dataset.t2m_dataset.inv_transform(
                            sample.cpu().permute(0, 2, 3, 1)
                        ).float()
                        sample = recover_from_ric(sample, n_joints)
                        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

                    pose_rep = "xyz" if self.model.data_rep in ["xyz", "hml_vec"] else self.model.data_rep
                    mask = None if pose_rep == "xyz" else mk["y"]["mask"].reshape(1, self.n_frames).bool()

                    xyz = self.model.rot2xyz(
                        x=sample,
                        mask=mask,
                        pose_rep=pose_rep,
                        glob=True,
                        translation=True,
                        jointstype="smpl",
                        vertstrans=True,
                        betas=None,
                        beta=0,
                        glob_rot=None,
                        get_rotations_back=False,
                    ).cpu().numpy()

                    # xyz: (1, J, 3, T) -> (T, 22, 3)
                    j = xyz.transpose(0, 3, 1, 2).reshape(1, -1, 22, 3)[0]

                    # add hands (approx) -> (T, 24, 3)
                    hand_len = 0.08824
                    d = (j[:, -2] - j[:, -4])
                    d /= (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8)
                    left = j[:, -2] + d * hand_len

                    d = (j[:, -1] - j[:, -3])
                    d /= (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8)
                    right = j[:, -1] + d * hand_len

                    return np.concatenate([j, left[:, None], right[:, None]], axis=1)  # (T,24,3)
                finally:
                    os.chdir(old_cwd)
                    # Restore sys.path (but keep MDM repo available for future calls)
                    # We don't fully restore to avoid repeated path manipulation
                    if self.repo in sys.path and sys.path[0] != self.repo:
                        sys.path.remove(self.repo)
                        sys.path.insert(0, self.repo)

        print(f"[MDM] Ready (repo={mdm_repo})")
        return MDMWrapper(model, diffusion, data, model_kwargs, n_frames, mdm_repo, cfg.device, cfg.guidance_param)

    # ----------------------------- Text mapping -----------------------------

    def _load_text_from_csv(self, csv_path: str, text_base_path: str):
        import pandas as pd
        df = pd.read_csv(csv_path)
        for path, new_name in zip(df["source_path"], df["new_name"]):
            key = "0-" + path[12:].replace("/", "_")[:-4]
            txt = osp.join(text_base_path, f"{new_name[:-4]}.txt")
            if not osp.exists(txt):
                continue
            segs = [s.split("#")[0].strip() for s in open(txt).read().split("#0.0#0.0")]
            segs = [s for s in segs if len(s) > 1]
            if segs:
                self.text_mapping[key] = segs[0]

    def _get_text_for_failed_key(self, failed_key: str) -> Optional[str]:
        t = self.text_mapping.get(failed_key)
        return t[0] if isinstance(t, list) and t else t

    # ----------------------------- Joints -> motion dict -----------------------------

    def _joints_to_rotations_smpl(self, joints: np.ndarray) -> np.ndarray:
        """(T,24,3) joints -> (T,24,3) axis-angle via bone direction alignment."""
        T = joints.shape[0]
        aa = np.zeros((T, 24, 3), dtype=np.float32)

        for t in range(T):
            for j in range(24):
                p = self.PARENT[j]
                if p < 0:
                    if joints.shape[1] > 9:
                        d = joints[t, 9] - joints[t, 0]
                        d /= (np.linalg.norm(d) + 1e-8)
                        f = np.array([0, 0, 1.0], dtype=np.float32)
                        axis = np.cross(d, f)
                        ang = np.arccos(np.clip(np.dot(d, f), -1, 1))
                        n = np.linalg.norm(axis)
                        if n > 1e-6:
                            aa[t, j] = axis / n * ang
                    continue

                d = joints[t, j] - joints[t, p]
                d /= (np.linalg.norm(d) + 1e-8)
                base = self.DEFAULT_DIR[j]
                axis = np.cross(base, d)
                ang = np.arccos(np.clip(np.dot(base, d), -1, 1))
                n = np.linalg.norm(axis)
                if n > 1e-6:
                    aa[t, j] = axis / n * ang

        return aa

    def _convert_mdm_to_motion_format(self, joints: np.ndarray) -> Optional[Dict]:
        """(T,24,3) -> motion_dict used by discriminator + dumping."""
        try:
            if joints.ndim != 3 or joints.shape[1:] != (24, 3):
                raise ValueError(f"Expected (T,24,3), got {joints.shape}")

            T = joints.shape[0]
            trans = joints[:, 0].copy()  # pelvis

            pose_aa = self._joints_to_rotations_smpl(joints).reshape(T, 72)
            pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((T, 6), dtype=pose_aa.dtype)], axis=1)

            aa_mj = pose_aa.reshape(T, 24, 3)[:, self.smpl_2_mujoco].copy()
            pose_quat = sRot.from_rotvec(aa_mj.reshape(-1, 3)).as_quat().reshape(T, 24, 4)

            beta = np.zeros(10, dtype=np.float32)
            self.smpl_local_robot.load_from_skeleton(torch.from_numpy(beta[None]), gender=[0], objs_info=None)

            # write skeleton xml in a local tmp folder
            tmp_dir = osp.join(osp.dirname(osp.abspath(__file__)), "_tmp_mjcf")
            os.makedirs(tmp_dir, exist_ok=True)
            xml_path = osp.join(tmp_dir, "smpl_humanoid_1.xml")
            self.smpl_local_robot.write_xml(xml_path)

            sk = SkeletonTree.from_mjcf(xml_path)
            root_trans = torch.from_numpy(trans) + sk.local_translation[0]
            st = SkeletonState.from_rotation_and_root_translation(sk, torch.from_numpy(pose_quat), root_trans, is_local=True)

            gq = (sRot.from_quat(st.global_rotation.reshape(-1, 4).numpy())
                  * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(T, -1, 4)

            root_trans = torch.from_numpy(filters.gaussian_filter1d(root_trans.numpy(), 3, axis=0, mode="nearest"))
            gq = np.stack([quat_correct(gq[:, i]) for i in range(gq.shape[1])], axis=1)
            gq = filters.gaussian_filter1d(gq, 2, axis=0, mode="nearest")
            gq /= (np.linalg.norm(gq, axis=-1, keepdims=True) + 1e-8)

            st = SkeletonState.from_rotation_and_root_translation(sk, torch.from_numpy(gq), root_trans, is_local=False)

            return dict(
                pose_quat_global=gq,
                pose_quat=st.local_rotation.numpy(),
                trans_orig=trans,
                root_trans_offset=root_trans.numpy(),
                beta=beta,
                gender="neutral",
                pose_aa=pose_aa,
                fps=30.0,
            )
        except Exception as e:
            print(f"Error converting joints -> motion_dict: {e}")
            return None

    # ----------------------------- Discriminator eval -----------------------------

    def _evaluate_motion_with_discriminator(self, motion_dict: Dict) -> float:
        try:
            env = self.agent.vec_env.env.task
            gq = motion_dict["pose_quat_global"]
            rt = motion_dict["root_trans_offset"]
            fps = motion_dict.get("fps", 30.0)
            T = len(gq)
            if T < 10:
                return 0.0

            n = min(30, T - 1)
            dt = 1.0 / fps
            idx = np.sort(np.random.choice(T, n, replace=False))

            root_pos = rt[idx]
            root_rot = gq[idx, 0]

            root_vel = np.zeros((n, 3))
            root_ang = np.zeros((n, 3))
            for i, t in enumerate(idx):
                t1 = min(t + 1, T - 1)
                root_vel[i] = (rt[t1] - rt[t]) / dt
                q0, q1 = sRot.from_quat(root_rot[i]), sRot.from_quat(gq[t1, 0])
                rv = (q0.inv() * q1).as_rotvec()
                a = np.linalg.norm(rv)
                if a > 1e-6:
                    root_ang[i] = (a / dt) * (rv / a)

            pose_aa = motion_dict["pose_aa"]
            dof_pos = pose_aa[idx]
            dof_vel = np.zeros_like(dof_pos)
            for i, t in enumerate(idx):
                t1 = min(t + 1, T - 1)
                dof_vel[i] = (pose_aa[t1] - pose_aa[t]) / dt

            key_body_ids = getattr(env, "_key_body_ids", None)
            if key_body_ids is None:
                key_body_ids = np.arange(min(5, gq.shape[1]))
            elif isinstance(key_body_ids, torch.Tensor):
                key_body_ids = key_body_ids.cpu().numpy()
            key_pos = root_pos[:, None, :].repeat(len(key_body_ids), axis=1)  # placeholder

            to = lambda x: torch.from_numpy(x).float().to(self.device)
            root_pos_t, root_rot_t = to(root_pos), to(root_rot)
            root_vel_t, root_ang_t = to(root_vel), to(root_ang)
            dof_pos_t, dof_vel_t = to(dof_pos), to(dof_vel)
            key_pos_t = to(key_pos)

            beta = motion_dict.get("beta", np.zeros(10))
            smpl_params = torch.from_numpy(np.concatenate([beta, np.zeros(6)])).float().to(self.device)
            smpl_params = smpl_params.unsqueeze(0).repeat(n, 1)

            if hasattr(self.motion_lib, "_motion_limb_weights") and len(self.motion_lib._motion_limb_weights) > 0:
                limb_w = self.motion_lib._motion_limb_weights[0:1].repeat(n, 1)
            else:
                limb_w = torch.zeros(n, 0, device=self.device)

            humanoid_type = getattr(env, "humanoid_type", "smpl")
            if humanoid_type in ["smpl", "smplh", "smplx"]:
                from phc.env.tasks.humanoid_amp import build_amp_observations_smpl
                amp_obs = build_amp_observations_smpl(
                    root_pos_t, root_rot_t, root_vel_t, root_ang_t,
                    dof_pos_t, dof_vel_t, key_pos_t, smpl_params, limb_w,
                    getattr(env, "dof_subset", None),
                    getattr(env, "_local_root_obs", True),
                    getattr(env, "_amp_root_height_obs", True),
                    hasattr(env, "dof_subset") and env.dof_subset is not None,
                    getattr(env, "_has_shape_obs_disc", False),
                    getattr(env, "_has_limb_weight_obs_disc", False),
                    getattr(env, "_has_upright_start", True),
                )
            else:
                from phc.env.tasks.humanoid_amp import build_amp_observations
                amp_obs = build_amp_observations(
                    root_pos_t, root_rot_t, root_vel_t, root_ang_t,
                    dof_pos_t, dof_vel_t, key_pos_t,
                    getattr(env, "_local_root_obs", True),
                    getattr(env, "_amp_root_height_obs", True),
                    dof_pos_t.shape[1], None
                )

            with torch.no_grad():
                obs = self.agent._preproc_amp_obs(amp_obs) if hasattr(self.agent, "_preproc_amp_obs") else amp_obs
                logits = self.agent.model.a2c_network.eval_disc(obs)
                return torch.sigmoid(logits).mean().item()

        except Exception as e:
            print(f"Error evaluating motion with discriminator: {e}")
            return 0.0

    # ----------------------------- Main entrypoint -----------------------------

    def generate_and_add_motions(self, failed_keys: List[str]) -> Dict[str, int]:
        stats = {"generated": 0, "accepted": 0, "added": 0}
        if self.mdm_talker is None:
            print("Error: MDM not initialized; cannot generate.")
            return stats

        accepted: Dict[str, List[Dict]] = {}
        for failed_key in tqdm(failed_keys, desc="Generating motions"):
            prompt = self._get_text_for_failed_key(failed_key)
            if not prompt:
                continue

            keep = []
            for i in range(self.num_samples_per_failed):
                joints = self.mdm_talker.generate_motion(prompt)  # (T,24,3)
                stats["generated"] += 1

                md = self._convert_mdm_to_motion_format(joints)
                if md is None:
                    continue

                p = self._evaluate_motion_with_discriminator(md)
                if p >= self.discriminator_threshold:
                    keep.append(md)
                    stats["accepted"] += 1
                    print(f"Accepted {failed_key} (p={p:.3f})")

            if keep:
                accepted[failed_key] = keep

        stats["added"] = self._dump_generated(accepted)
        return stats

    def _dump_generated(self, accepted: Dict[str, List[Dict]]) -> int:
        out = {f"{k}_gen_{i}": md for k, lst in accepted.items() for i, md in enumerate(lst)}
        if not out:
            return 0
        path = osp.join(self.agent.network_path, f"generated_motions_epoch_{self.agent.epoch_num:010d}.pkl")
        joblib.dump(out, path)
        print(f"Saved {len(out)} generated motions to {path}")
        print("Note: merge/reload motion library to use them.")
        return len(out)
