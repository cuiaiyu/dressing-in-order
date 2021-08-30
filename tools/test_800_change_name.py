"""
For ADGAN released generated images, this file rename the image files to our evaluation protocol.
"""
import os, shutil
from tqdm import tqdm 

test_pairs_fn = "fasion-pairs-test.csv"
from_dir = "test_800/images/"
to_dir = "adgan_800"

if not os.path.exists(to_dir):
    os.mkdir(to_dir)

with open(test_pairs_fn) as f:
    pairs = f.readlines()

print("%d pairs found." % len(pairs[1:]))

for pair in tqdm(pairs[1:]):
    i, from_key, to_key = pair.split(",")
    to_fn = "generated_%d.jpg" % (int(i)+1)
    from_fn = "%s___%s_vis.jpg" % (from_key, to_key[:-1])
    shutil.copy(os.path.join(from_dir, from_fn), os.path.join(to_dir, to_fn))
