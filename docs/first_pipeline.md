First Pipeline
==============

```{eval-rst}
.. note::
    It is highly recommended that you have completed the [ml4gw quickstart](https://github.com/ml4gw/quickstart/) instructions, or installed the equivalent software, before running the sandbox pipeline.
```

```{eval-rst}
.. note::
    It is assumed that you have already built each [project's container](./projects.md)
```

The Aframe `Sandbox` pipelines strings together `luigi` / `law` tasks to run an end-to-end workflow.
In short, the `Sandbox` pipeline will

1. Generate training data 
2. Generate testing data
3. Train or Tune a model
4. Export trained weights to TensorRT
5. Perform inference using Triton
6. Calculate sensitive volume

## Configuration
The `Sandbox` pipeline is configured by two main configuration files. A `.cfg` file is used by `law`, and contains the parameters
for the data generation, export, and inference tasks. See the [`bbh.cfg`](../aframe/pipelines/sandbox/configs/bbh.cfg) for a complete example.

Details of the training configuration is handled by [`lightning`](https://lightning.ai/docs/pytorch/stable/), which 
uses a `.yaml` file. See the [`config.yaml`](../../../projects/train/config.yaml) that lives at the root of the train project for an example. 

```{eval-rst}
.. note::
    When running the `Sandbox` pipeline, parameters that are common between training and other tasks (e.g. ifos, highpass, fduration) are specified once in the `.cfg` and automatically passed to the downstream training or tuning `config.yaml` by `luigi`/`law`.
```

## Initialize a Pipeline
The `aframe-init` command line tool can be used to initialize a directory with configuration files for a fresh run. 
In the specified directory, `aframe-init` will create default `.cfg` and `.yaml` configuration files, as well as a `run.sh` file for launching the pipeline.

```{eval-rst}
.. tip::
    When running a new "experiment", it is recommended to use `aframe-init` to initialize a new directory. This way, all the configuration associated with the experiment is isolated, and the experiment is reproducible.
```


While in the root directory, a sandbox pipeline can be initialized with

```bash
poetry run aframe-init offline  --mode sandbox --directory ~/aframe/my-first-run/ 
```

You can also initialize a directory for launching the tune pipeline

```bash
poetry run aframe-init offline --mode tune --directory ~/aframe/my-first-tune-run/ 
```

Now, you can navigate to the experiment directory and edit the configuration files as you wish.

## Running the Pipeline
```{eval-rst}
.. note:: 
    Running the sandbox pipeline out-of-the-box requires access to an enterprise-grade GPU(s) (e.g. P100, V100, T4, A[30,40,100], etc.). There are several nodes on the LIGO Data Grid which meet these requirements_**.
```

In the experiment directory a `run.sh` file will be created that looks like 

```bash
#!/bin/bash
# Export environment variables
export AFRAME_TRAIN_DATA_DIR=/home/albert.einstein/aframe/my-first-run/data/train
export AFRAME_TEST_DATA_DIR=/home/albert.einstein/aframe/my-first-run/data/test
export AFRAME_TRAIN_RUN_DIR=/home/albert.einstein/aframe/my-first-run/training
export AFRAME_CONDOR_DIR=/home/albert.einstein/aframe/my-first-run/condor
export AFRAME_RESULTS_DIR=/home/albert.einsteinaframe/my-first-run/results
export AFRAME_TMPDIR=/home/albert.einsteinaframe/my-first-run/tmp/

# launch pipeline; modify the gpus, workers etc. to suit your needs
# note that if you've made local code changes not in the containers
# you'll need to add the --dev flag!
LAW_CONFIG_FILE=/home/albert.einstein/aframe/my-first-run/sandbox.cfg poetry run --directory /home/albert.einstein/projects/aframev2 law run aframe.pipelines.sandbox.Sandbox --workers 5 --gpus 0
```

Environment variables are automatically set based on the specified experiment directory. These environment variables
are ingested by the `law` tasks and control where various pipeline artifacts are stored.

- `AFRAME_TRAIN_DATA_DIR` Training data storage
- `AFRAME_TEST_DATA_DIR` Testing data storage
- `AFRAME_TRAIN_RUN_DIR` Training artifact storage
- `AFRAME_CONDOR_DIR` Condor submit files and logs
- `AFRAME_RESULTS_DIR` Inference and sensitive volume results
- `AFRAME_TMPDIR` Intermediate data product storage 

The bottom of the `run.sh` contains the command that launches the pipeline. The `workers` argument specifies how many `luigi` workers to use. This controls how many concurrent tasks can be launched. It is useful to specify more than 1 worker when you have several tasks that are not dependent on one another. The default of 5 should be plenty.

The `gpus` argument controls which gpus to use for training and inference. Under the hood, the pipeline is simply setting
the `CUDA_VISIBLE_DEVICES` environment variable. `gpus` should be specified as a comma separated list.

The pipeline can now be kicked off by executing the `run.sh` 

```bash
bash ~/aframe/my-first-run/run.sh
```

```{eval-rst}
.. tip:: 
    The end to end pipeline can take a few days to run. 
    If you wish to launch an analysis with the freedom of ending
    your ssh session, use a tool like [`tmux`](https://github.com/tmux/tmux/wiki) or [`screen`](https://www.gnu.org/software/screen/manual/screen.html)
```

The most time consuming steps are training and performing inference. If you wish to reduce these timescales for testing the end-to-end analysis, consider altering the following arguments:
- [`max_epochs`](../../../projects/train/config.yaml#92) in the training `yaml` configuration file
- the amount of analyzed background livetime ([`Tb`](./configs/base.cfg#17)) 
- the number of injections ([`num_injections`](./configs/base.cfg#101))