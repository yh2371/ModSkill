<p align="center">
  <h1 align="center">ModSkill: Physical Character Skill Modularization</h1>
  <h3 align="center">ICCV 2025</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2502.14140">Paper</a> | <a href="https://yh2371.github.io/modskill/">Project Page</a> </h3>
  <div align="center"></div>
</p>

<div align="center">
  <img src="assets/teaser.png" />
  <img src="assets/pipeline.png" />
</div>

## Preparation

### Installation

Install the required dependencies:

```bash
pip install -r requirement.txt
```

### Download Data and Checkpoints

Download datasets and pretrained checkpoints using the provided script:

```bash
bash download_data.sh
```

This script will download:
- HumanML3D text descriptions and mappings to AMASS motions for generative adaptive sampling
- Pretrained model checkpoints
- Required assets and resources

The data will be saved to `sample_data/` and checkpoints to `output/HumanoidIm/` directories. Please follow [PHC](https://github.com/ZhengyiLuo/PHC) for steps to prepare AMASS training/test datasets.

## Training

To train a model, use the following command:

```bash
python modskill/run_hydra.py \
    learning=<learning_config> \
    exp_name=<experiment_name> \
    env=<env_config> \
    robot=<robot_config> \
    env.motion_file=<path_to_motion_file> \
    env.num_envs=<num_envs> \
    epoch=-1
```

For example:

```bash
python modskill/run_hydra.py \
    learning=im_pnn_bigger \
    exp_name=modskill_attn \
    env=env_im_pnn \
    robot=smpl_humanoid \
    env.motion_file=./sample_data/train.pkl \
    env.num_envs=3072 \
    epoch=-1
```

## Validation/Evaluation

To evaluate a trained model, set `test=True im_eval=True`:

```bash
python modskill/run_hydra.py \
    learning=<learning_config> \
    exp_name=<experiment_name> \
    epoch=<checkpoint_epoch> \
    test=True \
    env=<env_config> \
    env.motion_file=<path_to_motion_file> \
    env.num_envs=<num_envs> \
    env.obs_v=6 \
    headless=True \
    im_eval=True
```

For example:

```bash
python modskill/run_hydra.py \
    learning=im_pnn_bigger \
    exp_name=modskill_attn \
    epoch=-1 \
    test=True \
    env=env_im_pnn \
    env.motion_file=./sample_data/test.pkl \
    robot.freeze_hand=True \
    robot.box_body=False \
    env.num_envs=1 \
    env.obs_v=6 \
    headless=True \
    im_eval=True
```

## BibTeX
If you find our work helpful or use our code, please consider citing:
```bibtex
@inproceedings{huang2025modskill,
  title={Modskill: Physical character skill modularization},
  author={Huang, Yiming and Dou, Zhiyang and Liu, Lingjie},
  booktitle={IEEE/CVF International Conference on Computer Vision},
  year={2025}
  }
```

## Acknowledgments

We would like to acknowledge the following amazing works that this paper builds upon:

- **MDM**: [MDM](https://github.com/GuyTevet/motion-diffusion-model) 

- **PHC**: [PHC](https://github.com/ZhengyiLuo/PHC)
