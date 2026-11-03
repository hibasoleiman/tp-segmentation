Practical Work on Segmentation - HourGlass vs U-net
=========================================

## Goals

- implement HourGlass network
- implement U-net
- compare both methods
- optionally, you can test generative models on this basis: auto-encoder, VAE

## Evaluation
The practical will be evaluated in groups of two.
It is expected that you write a report with the plots/results of the 
different experiments. Submit the code with the report.

A good report :
* briefly details the experiment setting 
* illustrates the results (with screenshots, tables)
* briefly discusses the outcome of the experiment

```
Ask the teacher for the due date of the practical session.
```
## Getting started

The practical will take place on the Ensimag / Grenoble INP educational
GPU cluster with manual [here](https://gitlab.ensimag.fr/ensicompute/docs/-/wikis/Utilisation-de-la-plateforme-ensicompute).
In a nutshell, first you log in to the slurm server:

```
$ ssh -YK nash.ensimag.fr
```

Then, you can execute PyTorch scripts by prefixing them with the srun command as follows: 
(If the flag x11 does not work, you can remove it)

```
$ srun --gres=shard:1 --cpus-per-task=8 --mem=12GB --x11=all python3 train.py
```


### Dataset
1. Data is contained from the tictoc dance sequences.
   - **dataset** : The dataset contains the images and the masks stored on
     Ensimag machines in folder `/matieres/5MMVORF/04-dataset`, referred
     throughout as `dataset` folder below for simplicity. 
     
   - [`train.py`](train.py) is the file containing the main training code.

### References

The original articles where some of the architectures were described.

[1] U-Net: Convolutional Networks for Biomedical Image Segmentation ;
Olaf Ronneberger, Philipp Fischer, Thomas Brox ; 2015

## Part 1 : Examine the code

A first naive attempt to segment is provided through the SimpleConv architecture,
which runs the image through a series of convolutions.
A first stack of convolutions encodes increasingly more features,
and a second decodes progressively less features, until it predicts
just one value per pixel, which corresponds to our segmentation per pixel.

1. Run the training code and predict.
2. Does it converge easily? 
SimpleConv converges slowly: the loss decreases from ≈0.68 (close to ln 2 ≈ 0.693, random guessing) to ≈0.55 after 20 epochs and is still decreasing, without reaching a plateau. Train and test losses are almost identical, so there is no overfitting; the model is underfitting because its receptive field (≈9×9 pixels) is too small to capture the global shape of a person.


## Part 2 : Hourglass convolutional neural network

Implement the following Hourglass convolutional neural network, 
starting with a copy of the SimpleConv class. In the following diagram, 
you see what we want to achieve: instead of layering naive convolutions, 
we want to create a bottleneck, where the number of features is reduced 
and creates a 'latent space' encoding the abstract space of people segmentations.
To this goal, you must add pooling layers in the network encoder part
to progressively reduce the size of the feature map. Reciprocally,
in the decoder loop, you must upscale the features such that the size of 
the output image coincides with the input size (same number of upconvolutions).
Consult the documentation of [ConvTranspose2D](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html) in PyTorch.

![alt the hourglass][hourglass]

[hourglass]: hourglass_scheme.png "HourGlass scheme"

**Questions**

1. What do you observe with convergence?
2. Vary the number of features and depth encoding layers. For each of these attempts,
can you predict what is the size of the bottleneck map? Verify your predictions
by printing its actual size. Comment on how the performance varies
with the depth of the network / size of the bottleneck.


## Part 3 : U-net convolutional neural network
U-Net was invented to address precision problems of the original hourglass network. 
Implement the following U-net convolutional neural network, 
with 3 encoding and 3 decoding layers proposed, starting with a copy of your Hourglass.

![alt the unet][unet]

[unet]: unet_scheme.png "U-Net scheme"

**Questions**

1. What do you observe with convergence? 
2. Compare the obtained results with the Hourglass
3. As with the Hourglass, vary the number of features and depth
encoding layers and further compare both architectures.



#To be removed
# TP Segmentation – Command Cheat Sheet

Nov 2, 2026 · @Heba

## 1. Start of every session

Every lab session starts the same way: VPN on, log in to nash, go into the TP folder.

1. Turn on the Ensimag or grenet.fr VPN (not needed on Ensimag TP room PCs).
2. Open a laptop terminal (`Ctrl+Alt+T`) and log in. Login is lowercase; the password is typed blind (nothing shows).

```bash
ssh sleimanh@nash.ensimag.fr
```

3. Go into the TP folder:

```bash
cd ~/TP-segmentation
```

Check which window you are in by its prompt:

| Window | Prompt starts with | Used for |
| --- | --- | --- |
| Laptop terminal | `(base) hebasleiman@...` | copying files (`scp`) |
| Nash terminal | `(opt) sleimanh@nash` | training (`srun ...`) |

To test the connection if ssh hangs (ping does not work on nash, this does):

```bash
timeout 5 bash -c '</dev/tcp/nash.ensimag.fr/22' && echo "OK: nash reachable" || echo "BLOCKED: turn on VPN"
```

## 2. Pick a free GPU server

GPUs are shared with other students, so check which server is free before training. Run this on nash:

```bash
for n in $(sinfo -N -h -o "%N" | sort -u); do echo -n "$n: "; timeout 20 srun -w $n --gres=shard:1 --cpus-per-task=1 --mem=1GB nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>&1 | tr '\n' ' '; echo; done
```

How to read it: each server has 6 GPUs of 23040 MiB. `0 MiB, 23040 MiB` = free; `22000 MiB, 23040 MiB` = full. `ampere` and `tesla` are not available to you: ignore them.

What happened on 2 Nov 2026:

| Server | State | Note |
| --- | --- | --- |
| turing-2, turing-4 | full | CUDA out of memory |
| turing-5 | free but broken | scikit-learn error: No module named 'narwhals' |
| turing-6 | free, works | used for the SimpleConv run |
| turing-7 to turing-11 | free | not tested yet |

This changes every day, so re-run the check each session. In the commands below, replace `turing-6` with the free server you picked.

## 3. Train and predict

Train first (about 2 minutes for SimpleConv), then predict. Both run on nash, inside `~/TP-segmentation`.

```bash
MPLBACKEND=Agg srun -w turing-6 --gres=shard:1 --cpus-per-task=8 --mem=12GB python3 train.py
```

```bash
MPLBACKEND=Agg srun -w turing-6 --gres=shard:1 --cpus-per-task=8 --mem=12GB python3 predict.py
```

Run `predict.py` before renaming `output/`: it loads the model from the folder named `output`.

| Piece | Meaning |
| --- | --- |
| `MPLBACKEND=Agg` | save plots as PNG files instead of opening windows |
| `srun` | run on a GPU server, not on nash |
| `-w turing-6` | run on this exact server (`--exclude=turing-2` = any server except this one) |
| `--gres=shard:1` | one share of a GPU (required) |
| `--cpus-per-task=8` | 8 CPU cores (always set it, or you reserve the whole server) |
| `--mem=12GB` | 12 GB RAM (always set it, same reason) |

What you get in `output/`: `best_model.pth` (trained model), `convergence_plot.png` (loss curves), `test_data.csv` (test image list), and 10 `predict_plot_*.png` (image, true mask, predicted mask, probability).

To switch network, edit the model line in `train.py`:

```python
model = SimpleConv((3,16),(16,1))
# model = HourGlass((3,16,32,64),(64,32,16,1))
# model = UNet((3,16,32,64),(64,32,16,1))
```

## 4. Save each experiment

Rename `output/` after every train + predict, or the next run overwrites it. `mv` = rename.

```bash
mv output output_hourglass_d3
ls
```

Suggested names, one folder per experiment:

- `output_simpleconv` (done)
- `output_hourglass_d3`, `output_hourglass_d2`, `output_hourglass_d4` (d = number of pooling levels)
- `output_unet_d3`, `output_unet_d2`, ...
- add channels if you vary them too, e.g. `output_unet_d3_c32`

Keep the same batch size and number of epochs for every network so the comparison is fair.

## 5. Copy files between laptop and nash

`scp` = secure copy: `scp WHAT WHERE`. Always run it in the **laptop** terminal; `-r` copies a whole folder.

Laptop to nash (sending code):

```bash
scp -r TP-segmentation sleimanh@nash.ensimag.fr:~/
scp model/network.py sleimanh@nash.ensimag.fr:~/TP-segmentation/model/
```

Nash to laptop (getting results back):

```bash
scp -r sleimanh@nash.ensimag.fr:~/TP-segmentation/output_simpleconv ~/
xdg-open ~/output_simpleconv
```

The originals stay on nash, so a photo deleted on the laptop can be copied again. With VS Code Remote-SSH you can skip `scp`: edit files on nash directly and click PNGs to view them.

## 6. Fixes to the TP code

Three fixes are needed; two are done and one is still to do. Run them on nash, inside `~/TP-segmentation`. `sed -i 's/OLD/NEW/' file` = find-and-replace saved directly in the file.

| Fix | File | Status | Why |
| --- | --- | --- | --- |
| num\_workers = 8 | train.py | done | `os.cpu_count()` started more loader workers than the 8 CPUs reserved |
| weights\_only=False | predict.py | done | PyTorch 2.12 refuses to load a full saved model by default |
| Loss step count | train.py | to do | test loss was shown about 25% too high |
| Batch size 64 | config.py | only if needed | use it only if you get CUDA out of memory |

### Fix 1: number of data-loading workers (done)

```bash
sed -i 's/num_workers=os.cpu_count()/num_workers=8/g' train.py
```

### Fix 2: loading the saved model (done)

`train.py` saves the whole model (structure + weights). Since PyTorch 2.6, `torch.load` accepts only plain weights by default, so `predict.py` crashed. `weights_only=False` means "not only weights": the weights are still loaded. Safe because the file is your own.

```bash
sed -i 's/torch.load(config.BEST_MODEL_PATH)/torch.load(config.BEST_MODEL_PATH, weights_only=False)/' predict.py
```

### Fix 3: the loss averaging bug (to do, before HourGlass)

The code divides the summed loss by `len(dataset) // BATCH_SIZE`, which drops the last, smaller batch. Test set: 1205 images = 5 batches, but it divides by 1205 // 256 = 4, so the test loss is about 25% too high (0.657 shown, about 0.53 real). Train is about 4% too high (27 batches, divides by 26). The gap between the curves looked like overfitting but was mostly this bug: worth one sentence in the report.

```bash
sed -i 's|trainSteps = len(trainDS) // config.BATCH_SIZE|trainSteps = len(trainLoader)|; s|testSteps = len(testDS) // config.BATCH_SIZE|testSteps = len(testLoader)|' train.py
grep -n "Steps =" train.py
```

The `grep` should print `trainSteps = len(trainLoader)` and `testSteps = len(testLoader)`. Then retrain SimpleConv (2 minutes) so all three networks use the same correct loss.

### Optional: smaller batches

```bash
sed -i 's/BATCH_SIZE = 256/BATCH_SIZE = 64/' model/config.py
grep BATCH_SIZE model/config.py
```

Batch 64 = about 106 weight updates per epoch instead of 26, so curves change. If you switch, use 64 for every network and mention it in the report.

## 7. Common errors

| Error | Cause | Fix |
| --- | --- | --- |
| `torch.OutOfMemoryError: CUDA out of memory` | another job fills that GPU | run the server check (section 2), use `-w` with a free server |
| `No module named 'narwhals'` | broken install on that server (turing-5) | use another server |
| `No module named 'torch'` | you ran python on nash, not on a GPU server | put `srun ...` in front |
| `weights_only` error in predict.py | Fix 2 not applied | apply Fix 2 (section 6) |
| `Password:` asked again | wrong username or password | login is lowercase `sleimanh`; type the password blind |
| `Connection closed` | too many failed tries | just run `ssh` again |
| ssh hangs, nothing happens | VPN off | turn on the VPN, test with the command in section 1 |
| `EOG-WARNING ... thumbnail failed` | image viewer was open during scp | harmless, ignore |
| `FileNotFoundError` for best\_model.pth | `output/` already renamed | rename back: `mv output_xxx output` |

## 8. Terminal basics

| Keys or command | What it does |
| --- | --- |
| `Ctrl+Shift+V` | paste in the terminal (`Ctrl+V` does not work) |
| `Ctrl+Shift+C` | copy selected text from the terminal |
| `Ctrl+C` | cancel the running command |
| `Ctrl+Alt+T` | open a new terminal |
| `ls` | list files in the current folder |
| `cd folder` / `cd ..` | go into a folder / go up one level |
| `pwd` | show which folder you are in |
| `mv old new` | rename a file or folder |
| `nano file` | simple editor: `Ctrl+O` Enter to save, `Ctrl+X` to quit |
| `squeue -u sleimanh` | see your jobs on the cluster (R = running) |
| `scancel <job number>` | stop one job |
| `exit` | log out of nash |
