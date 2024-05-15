## Tutorial for property guidance pipeline

### Step 1: Prepare the dataset

Prepare the raw data as

```
|-- data
    |-- properties
	    |-- <property>
            	|-- cif
            	|-- <raw_data>.csv
```

The csv file should at least contain the following 3 columns

```
material_id, cif, <prop>
```

``<prop>`` can be arbitrary property types, like formation_energy_per_atom in mp_20.


Split the raw data via the following script

```
python scripts/make_split.py --dir data/properties/<property> --csv <raw_data>.csv
```

The default setting will shuffle the dataset in random seed 42 and split it into train.csv, val.csv and test.csv with ratio 8:1:1. 

### Step 2: Train a property prediction model as an efficient evaluator

```
python diffcsp/run.py model=prediction data=property data.subdir=<property> data.prop=<prop> data.task=<task> data.opt_target=<opt_target> exptag=<property>_<prop> expname=prediction
```

The trained model is saved in ``singlerun/<property>_<prop>/prediction``. The default 3D encoder is DimeNet++, and one can change it into more powerful encoders (e.g. Equiformer).

``<task>`` can be chosen from classification/regression.

``<opt_target>`` have different meanings for different tasks:

For classification, ``<opt_target>`` means the required class index to generation.
For regression, ``<opt_target> = 1`` means to generate candidates with higher property (like Tc for superconductors), while ``<opt_target> = -1`` means to generate candidates with lower property (like formation energy)

### Step 3: Pre-Train an unconditional diffusion model

```
python diffcsp/run.py data=<dataset> model=diffusion_w_type expname=<expname>
```

### Step 4: Train a time-dependent guidance model

```
python diffcsp/run.py model=energy data=property data.subdir=<property> data.prop=<prop> data.task=<task> data.opt_target=<opt_target> exptag=<property>_<prop> expname=guidance
```

The trained model is saved in ``singlerun/<property>_<prop>/guidance``.

### Step 5: Generate candidates with guidance

```
python scripts/optimization.py --model_path singlerun/<property>_<prop>/guidance --uncond_path <uncond_model_path>
```

The above command will yield ``eval_opt.pt`` under the ``singlerun/<property>_<prop>/guidance`` directory, which contains 500 optimized structures.

### Step 6: Evaluate the trained model and optimized samples

```
python scripts/eval_optimization.py --dir singlerun/<property>_<prop>
```

The results are logged in ``singlerun/<property>_<prop>/results`` as 

```
|-- results
    |-- summary.log
    |-- results.csv
    |-- cif
        |-- xx.cif
        ...
```

``summary.log`` summaries the results of the property prediction & guidance model. 

### Step 7 (Optional): Multi-Property Optimization

```
python scripts/multi_optimization.py --uncond_path <uncond_model> -cond_paths <cond_model1> <cond_model2> ... <cond_modelN> --augs aug1 aug2 ... augN
```

And one could develop the pipeline as above to evaluate the performance on each property.
