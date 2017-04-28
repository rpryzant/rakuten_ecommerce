

## juman_tokenization.py

#### Description

This file tokenizes japanese text with the Juamn CLI iterface. 

#### Dependancies

1. Install Juman:

```
wget -O juman7.0.1.tar.bz2 "http://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=http://nlp.ist.i.kyoto-u.ac.jp/nl-resource/juman/juman-7.01.tar.bz2&name=juman-7.01.tar.bz2"
bzip2 -dc juman7.0.1.tar.bz2  | tar xvf -
cd juman-7.01
./configure
make
[sudo] make install
```

2. Install tqdm: `pip install tqdm`



#### Usage

```
python juman_tokenization.py [input file] [output file path] (-v vocab path)  (-pos)
```

This will tokenize the `input file` with Juman and write to the `output file path`. Flags in parentheses `(...)` are **optional**. They are as follows:

* `-pos`: attach POS tags to each word in the output.
* `-v [vocab file path]`: write a vocab file as well.

This is an example of a full run. This fun will tokenize `japanese_text.txt` and write its output to `japanese_text.tokenized`. It will also write the vocabulary to `text.vocab`. POS tags will be attached to each word.


```
python juman_tokenization.py japanese_text.txt japanese_text.tokenized -v text.vocab -pos
```

