from ray import tune

space = {
    "model.init_args.arch.init_args.kernel_size": tune.choice([3, 7, 15]),
    "model.init_args.arch.init_args.layers": tune.choice(
        [[2, 3, 4], [3, 4, 6, 3], [3, 4, 6, 5, 4, 4, 3]]
    ),
    "model.learning_rate": tune.loguniform(1e-4, 1e-2),
    "model.pct_lr_ramp": tune.uniform(0.05, 0.7),
    "data.waveform_prob": tune.uniform(0.2, 0.6),
}
