# **LaTac: Latent Tactics for Robust Multi-Agent Coordination Under Intermittent Communication**

## **experiment instructions**

### **Installation instructions**
See `requirments.txt` file for more information about how to install the dependencies.
```python
conda create -n latac python=3.10.0 -y
conda activate latac
pip install -r requirements.txt
```

### **Run an experiment**

You can execute the following command to run ACORM on SMAC benchmark with a map config, such as `3s_vs_5z`:

```python
python src/main.py --config llt --env_config sc2 with --env_name 3s_vs_5z --max_train_steps 5000000
```

or you can execute the following command to run ACORM on MPE benchmark with a map config, such as `cn_3v3_classical`:
```python
python src/main.py --config llt --env_config mpe with --map_name cn_3v3_classical
```

All results will be stored in the `LaTac/results` folder. You can see the console output, config, and tensorboard logging in the `LaTac/results/tb_logs` folder.

## **License**

Code licensed under the Apache License v2.0.

