## Using this code.
#### author: Rey Cruz-Torres

The first time the code is run, the flag ```process_g4_hits``` must be passed, so that the Geant hits are stored in a hdf5 file:

```
python3 synrad_hits.py --process_g4_hits
```

After that first time, it is faster to read in the dictionary in this hdf5 file than generating it on the fly. Thus, don't include that flag anymore. From that point on do something like:

```
python3 synrad_hits.py --analyze --nevents 10000 --int_window 100.e-09
```
