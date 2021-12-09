# fine

## Training

In lots of places, you might see that we "split on identity instead of fingerprint." An example will help to clarify. If we have 50 identities with 5 fingerprints each, a split on identity (assuming a split of 0.8 for training, and 0.2 for testing) would mean that train becomes `[40, 5, ...]` and test becomes `[10, 5, ...]`. If we had split on fingerprint, then training would become `[50, 4, ...]` and testing would become `[50, 1, ...]`.

When training, you can specify three parameters via the command line:

1. Train/test split. This will be used to split on both both on identity and on fingerprint.
2. Partial ratio. This needs to be documented better, but essentially, it represents what fraction of the width/height of each image are used to generate a partial.
3. Identity split. Whether to use the data split on identity for training. Historically, we've only trained on fingerprints, i.e. running _without_ this flag.

An example command to run is as follows:

`python3 src/train.py --train-test-split 0.6 --partial-ratio 0.75 --identity-split`
