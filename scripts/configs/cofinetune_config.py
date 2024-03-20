from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

FINETUNING_KWARGS = {
    "name": "bridge_dataset",
    "data_dir": "/home/ubuntu/tensorflow_datasets/",

    "image_obs_keys": {"primary": "image_0", "wrist": None},
    "state_obs_keys": ["state", None],
    "language_key": "language_instruction",
    "action_proprio_normalization_type": "normal",
    # All actions are relative deltas, except for the last one (gripper) which is absolute
    # Specifying this is only necessary if you want to predict > 1 step into the future
    "absolute_action_mask": [False, False, False, False, False, False, True],
    # standardize_fn is dynamically loaded from a file
    # for example: "experiments/kevin/custom_standardization_transforms.py:aloha_dataset_transform"
    "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:bridge_dataset_transform",
    # If the default data loading speed is too slow, try these:
    # "num_parallel_reads": 8,  # for reading from disk / GCS
    # "num_parallel_calls": 16,  # for initial dataset construction
}

def get_config(config_string="full,multimodal"):
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    # Fill this in for your own dataset!

    # There should be two image keys
    # first image key should be the third-person view (None if not used)
    # and second image key should be the wrist view (None if not used)

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    elif mode == "head_mlp_only":
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )
    elif mode == "frozen_transformer":
        frozen_keys = ("octo_transformer.BlockTransformer_0.*",)
    else:
        raise ValueError("Invalid mode")

    max_steps = FieldReference(50000)
    window_size = FieldReference(default=1)

    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy = None
        keep_image_prob = 0.0
    elif task == "multimodal":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")

    traj_transform_kwargs = dict(
        window_size=window_size,
        future_action_window_size=3,
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
        # If the default data loading speed is too slow, try these:
        # num_parallel_calls=16,  # for less CPU-intensive ops
    )
    workspace_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    wrist_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    frame_transform_kwargs = dict(
        resize_size={
            "primary": (256, 256),  # workspace (3rd person) camera is at 256x256
            "wrist": (128, 128),  # wrist camera is at 128x128
        },
        image_augment_kwargs=[
            workspace_augment_kwargs,
            wrist_augment_kwargs,
        ],
    )

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        num_steps=max_steps,
        log_interval=100,
        eval_interval=5000,
        save_interval=5000,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="octo_cofinetune", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs=get_dataset_config(traj_transform_kwargs, frame_transform_kwargs, shuffle_buffer_size=1000, batch_size=256*2),
        cofinetuning_split=0.5, # probability of sampling from the finetuning set vs the pretraining set
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=3e-4,
                warmup_steps=2000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,  # if you are using grad accumulation, you need to adjust max_steps accordingly
        ),
        prefetch_num_batches=0,
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),
        viz_kwargs=dict(
            eval_batch_size=128,
            trajs_for_metrics=100,
            trajs_for_viz=8,
            samples_per_state=8,
        ),
    )
    config["batch_size"] = config["dataset_kwargs"]["batch_size"]

    return ConfigDict(config)

# Copied over from config.py with minor modificaiton
# Passing in config.dataset_kwargs.oxe_kwargs.data_dir=/mnt/nfs/data/tensorflow_dataset --config.dataset_kwargs.oxe_kwargs.data_mix=oxe_magic_soup
# should get you a dataset resembling the pretraining dataset
def get_dataset_config(traj_transform_kwargs, frame_transform_kwargs, shuffle_buffer_size, batch_size):
    return {
        # oxe_kwargs will generate dataset_kwargs_list and sampling weights
        "oxe_kwargs": dict(
            data_mix=placeholder(str),
            data_dir=placeholder(str),
            load_camera_views=("primary", "wrist"),
            load_depth=False,
        ),
        "cofinetuning_kwargs": FINETUNING_KWARGS,
        "traj_transform_kwargs": traj_transform_kwargs,
        "frame_transform_kwargs": frame_transform_kwargs,
        "traj_transform_threads": 48,  # shared between all datasets
        "traj_read_threads": 48,  # shared between all datasets
        "shuffle_buffer_size": shuffle_buffer_size,  # shared between all datasets
        "batch_size": batch_size,
    }