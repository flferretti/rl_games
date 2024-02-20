from jym_envs import create_jax_env
from rl_games.torch_runner import Runner
from rl_games.envs.jym import JymEnv

if __name__ == "__main__":
    cartpole_config = {
        "params": {
            "algo": {"name": "a2c_continuous"},
            "config": {
                "bound_loss_type": "regularisation",
                "bounds_loss_coef": 0.0,
                "clip_value": True,
                "critic_coef": 4,
                "e_clip": 0.2,
                "entropy_coef": 0.0,
                "env_config": {"env_name": "cartpole", "seed": 5},
                "env_name": "jym",
                "gamma": 0.99,
                "grad_norm": 1.0,
                "horizon_length": 8,
                "kl_threshold": 0.008,
                "learning_rate": "3e-4",
                "lr_schedule": "adaptive",
                "max_epochs": 5000,
                "mini_epochs": 4,
                "minibatch_size": 1024,  # 32768
                "name": "cartpole-jym",
                "normalize_advantage": True,
                "normalize_input": True,
                "normalize_value": True,
                "num_actors": 128,
                "player": {"render": True},
                "ppo": True,
                "reward_shaper": {"scale_value": 0.1},
                "schedule_type": "standard",
                "score_to_win": 20000,
                "tau": 0.95,
                "truncate_grads": True,
                "use_smooth_clamp": True,
                "value_bootstrap": True,
            },
            "model": {"name": "continuous_a2c_logstd"},
            "network": {
                "mlp": {
                    "activation": "elu",
                    "initializer": {"name": "default"},
                    "units": [256, 128, 64],
                },
                "name": "actor_critic",
                "separate": False,
                "space": {
                    "continuous": {
                        "fixed_sigma": True,
                        "mu_activation": "None",
                        "mu_init": {"name": "default"},
                        "sigma_activation": "None",
                        "sigma_init": {"name": "const_initializer", "val": 0},
                    }
                },
            },
            "seed": 5,
        }
    }

    env_name = "cartpole"
    configs = {
        "cartpole": cartpole_config,
    }
    networks = {
        "cartpole": "runs/cartpole/nn/cartpole-jym.pth",
    }

    config = configs[env_name]
    network_path = networks[env_name]
    config["params"]["config"]["full_experiment_name"] = env_name
    config["params"]["config"]["max_epochs"] = 1000

    runner = Runner()
    runner.load(config)
    runner.run(
        {
            "train": True,
        }
    )
