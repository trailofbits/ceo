# CEO 

<img src="https://image.freepik.com/free-icon/suit-and-tie-outfit_318-42494.jpg" width="150">

Central Exploit Organizer (CEO) extracts, collects, analyzes and build predictive models to guide Cyber Reasoning Systems (CRS) toward the discovery of vulnerabilities in binary programs. It aims to be the missing component in modern CRS.
CEO works collecting data to predict which is the best *action* that a CRS can perform. Given a test case (binary + input), our CRS can perform several *actions* using the following techniques:

* Symbolic execution with [Manticore](https://github.com/trailofbits/manticore).
* Smart fuzzing with [American Fuzzy Lop](http://lcamtuf.coredump.cx/afl/).
* Blind fuzzing with [GRR](https://github.com/trailofbits/grr).

Every time our CRS performs an action with a fixed amount of resources (time or memory), there is one of these results:

* r0: it fails to start or it produces no new test cases.
* r1: it produces additional test cases.
* r2: it finds a valuable test case (e.g, a crash).

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
$ wget https://github.com/trailofbits/ceo/releases/download/0.1/cgc-corpus.tar.xz
$ tar -xf cgc-corpus.tar.xz
$ cd corpus
$ mkdir bins
$ wget "https://github.com/angr/binaries/blob/master/tests/cgc/PIZZA_00001?raw=true" -O bins/PIZZA_00001
$ printf "bins/PIZZA_00001" > test.txt
$ mkdir -p PIZZA_00001/inputs
$ printf "AAA" > PIZZA_00001/inputs/file
$ ceo-bin test.txt test
```

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
