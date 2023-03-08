# SMB
Scalable Bayesian Modelling: A comparison

## Setting up the environment

```sh
conda env create -f environment.yaml
```

```sh
conda activate sbm
```

## Using the template

In the folder `notebooks`, you can find the file `template.ipynb` where you can add the code to get your data and models to create a benchmark for your specific use case. Cells preceded by the message **✍🏽 User input required** should be filled, the other cells can be optionally modified according to your needs.

The sampling results are saved by default in the path `data/results`.

The folder also has the file `example.ipynb`, with an example using the template.


## Executing in Google Colab

The template can be executed in Google Colab. Before executing the code, follow these steps:

1. Change runtime in `Runtime > Change runtime type` if you want to execute the notebook using GPU.
2. Uncomment the first cell which makes sure Colab has the correct versions and required files.
3. Set the variable `output_path` to `data/results` or to a folder you know exists in the environment.
