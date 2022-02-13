# diffusions

## Setup
```
  conda create -n example_env python=3.8
  conda activate example_env
  pip install git+https://github.com/JTT94/jtils.git
  pip install tensorflow tensorflow_datasets jax flax jaxlib tqdm matplotlib scikit-learn

```

## Run Example notebooks 
```
    pip install jupyter 
    
```

## Procedure
- Training consists of two steps:
  - Sample data $$x_0$$
  - Sample random time $t \sim Uniform(1,2,3\ldots,T)$
  - Apply noise from SDE to generate $x_t$
  - Train a model based on objective $E[||model(x_t, t) - noise||^2]$

- Sampling consists of applying the model sequentially through time and scaling the output

