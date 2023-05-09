default_args = {
  "env_buffer":{
    "max_buffer_size": 1000000
  },
  "model_buffer":{
    "max_buffer_size": 2000000
  },
  "agent":{
    "gamma": 0.99,
    "target_smoothing_tau": 0.005,
    "alpha": 0.2,
    "reward_scale": 1.0,
    "q_network":{
      "hidden_dims": [256,256],
      "act_fn": "relu",
      "out_act_fn": "identity",
      "optimizer_class": "Adam",
      "learning_rate":0.0003,
    },
    "policy_network":{
      "hidden_dims": [256,256],
      "deterministic": True,
      "act_fn": "relu",
      "out_act_fn": "identity",
      "re_parameterize": True,
      "optimizer_class": "Adam",
      "learning_rate":0.0003
    },
    "entropy":{
      "automatic_tuning": True,
      "target_entropy": -1,
      "learning_rate": 0.0003,
      "optimizer_class": "Adam"
    }
  },

  "transition_model":{
    "use_weight_decay": True,
    "optimizer_class": "Adam",
    "learning_rate":0.001,
    "holdout_ratio": 0.2,
    "inc_var_loss": True,
    "model":{
      "hidden_dims": [200, 200, 200, 200],
      "decay_weights": [0.000025, 0.00005, 0.000075, 0.000075, 0.0001],
      "act_fn": "swish",
      "out_act_fn": "identity",
      "num_elite": 5,
      "ensemble_size": 7
    }
  },
  "rollout_step_scheduler":{
    "initial_val": 1,
    "target_val": 1,
    "start_timestep": 20,
    "end_timestep": 150,
    "schedule_type": "linear"
  },
  "trainer":{
    "max_epoch": 125,
    "agent_batch_size": 256,
    "model_batch_size": 256,
    "rollout_batch_size": 100000,
    "rollout_mini_batch_size": 10000,
    "model_retain_epochs": 1,
    "num_env_steps_per_epoch": 1000,
    "train_model_interval": 250,
    "train_agent_interval": 1,
    "max_trajectory_length":1000,
    "eval_interval": 1000,
    "num_eval_trajectories": 10,
    "snapshot_interval": 100000,
    "warmup_timesteps": 5000,
    "save_video_demo_interval": -1,
    "log_interval": 250,
    "model_env_ratio": 0.95,
    "hold_out_ratio": 0.1,
    "num_agent_updates_per_env_step": 2,
    "max_model_update_epochs_to_improve": 5,
    "max_model_train_iterations": "None"
  }
  
}
