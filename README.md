# CEO 

<img src="https://image.freepik.com/free-icon/suit-and-tie-outfit_318-42494.jpg" width="150">

Central Exploit Organizer (CEO) extracts, collects, analyzes and build predictive models to guide Cyber Reasoning Systems (CRS) toward the discovery of vulnerabilities in binary programs. It aims to be the missing component in modern CRS.
CEO works collecting data to predict which is the best *action* that a CRS can perform. Given a test case (binary + input), our CRS can perform several *actions* using the following techniques:

* Symbolic execution with [Manticore](https://github.com/trailofbits/manticore).
* Smart fuzzing with [American Fuzzy Lop](http://lcamtuf.coredump.cx/afl/).
* Blind fuzzing with [GRR](https://github.com/trailofbits/grr).

Every time our CRS performs an action with a fixed amount of resources (time or memory), there is one of these results:

* r0: it fails to start.
* r1: it produces no additional test cases.
* r2: it produces additional test cases.
* r3: it finds a valuable test case (e.g, a crash).

CEO aims to predict the techinque (and parameters) we should use in a test case to obtain the desire result. It is illustrated in this overview diagram:

![overview](https://github.com/trailofbits/ceo/blob/master/docs/discovery-overview.png)


This repository contains a prototype that will only work with CGC binaries. If you want to test it, you can find a large set of precompiled CGC binaries [here](https://github.com/zardus/cgc-bins).

## Requirements

* Python 2.7 with setuptools
* GRR requirements: [gflags](https://github.com/gflags/gflags). 
* AFL requirements [(with QEMU support!)](https://github.com/ggrieco-tob/afl-cgc/blob/master/qemu_mode/build_qemu_support.sh#L33).

Other required Python packages are: [manticore (my fork)](https://github.com/ggrieco-tob/manticore), 
[scipy](https://scipy.org/), [scikit-learn](http://scikit-learn.org/) and 
[imbalanced-learn](http://imbalanced-learn.org) but these are automatically installed
by our [script](https://github.com/ggrieco-tob/ceo/blob/master/install.sh).

In Debian/Ubuntu, you can run:

```
# apt-get install libgflags-dev bison glib2
```

## Installation

To compile and install locally the required external tools (afl-cgc, grr and manticore) 
execute:

```
$ ./install.sh
```

### Quickstart using a (small) corpus 

```
$ wget https://github.com/ggrieco-tob/ceo/releases/download/0.1/cgc-corpus.tar.xz
$ tar -xf cgc-corpus.tar.xz
$ cd corpus
$ mkdir bins
$ wget "https://github.com/angr/binaries/blob/master/tests/cgc/PIZZA_00001?raw=true" -O bins/PIZZA_00001
$ printf "bins/PIZZA_00001" > test.txt
$ mkdir -p PIZZA_00001/inputs
$ printf "AAA" > PIZZA_00001/inputs/file
$ ceo-bin test.txt test
```

Then, CEO will output the predictions:

```
[+] Predicting best action for test case:
('PIZZA_00001', 'file', 'AAA')
[+] Extracting features
[+] Finding best predictor
[+] For grr, the best predictor is:
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
[+] Possible outcomes are:
['splice'] → r2
['splice_chunked'] → r1
['dropper'] → r2
['random'] → r1
['bitflip1'] → r1
['bitflip2'] → r1
['bitflip3'] → r2
['bitflip4'] → r2
['bitflip5'] → r1
['bitflip6'] → r2
['bitflip7'] → r2
['bitflip8'] → r2
['bitflip2_2'] → r1
['bitflip3_2'] → r2
['bitflip4_2'] → r1
['bitflip5_2'] → r2
['bitflip6_2'] → r1
['bitflip7_2'] → r1
['bitflip8_2'] → r2
['bitflip4_4'] → r1
['bitflip6_4'] → r2
['bitflip8_4'] → r1
['bitflip8_8'] → r2
['inf_bitflip_random'] → r1
['random'] → r1
['inf_radamsa_chunked'] → r2
['inf_radamsa_spliced'] → r1
['inf_radamsa_concat'] → r1
['inf_chunked_repeat'] → r2
...
```

Using these predictions, we can decide to use GRR with an specific mutator.

### Training

1. Create a text file with one executable program path per line.
2. Optionally, create directories named as the filenames of the binaries to execute. 
   with a directory named "input" to give the initial inputs. For instance:
   ```
   mkdir -p CROMU_00001/inputs/
   printf AAA > CROMU_00001/inputs/file
   ```
   Otherwise, CEO will use a [list of *special* strings as initial inputs](https://github.com/minimaxir/big-list-of-naughty-strings) 
2. If your target file is "train.txt", execute:
   ```
    $ ceo-bin train.txt init
   ```
    
3. Now you can go to every "fold" directory and run ceo independely in each one. For instance: 
   ```
   $ cd fold-0
   $ ceo-bin train.txt train
   ```

### Predicting

1. After training, open a shell in the directory with the "fold" subdirectories.
2. Create a text file with one executable program path per line.
3. Create directories named as the filenames of the binaries to execute. 
   with a directory named "input" to give the initial inputs. For instance:
   ```
   mkdir -p NRFIN_00001/inputs/
   printf AAA > NRFIN_00001/inputs/file
   ```
4. If your target file is "test.txt", execute:
   ```
    $ ceo-bin test.txt test
   ```
