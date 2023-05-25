# download data from https://github.com/yumingj/DeepFashion-MultiModal
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataroot', type=str, default="data", help='data root')

args = parser.parse_args()

if not os.path.exists(args.dataroot):
  os.system("mkdir {}".format(args.dataroot))

def download_from_gdrive(dst_root, fn, gdrive_path, iszip=True):
  if not os.path.exists(dst_root):
    os.system("mkdir {}".format(dst_root))
  if not os.path.exists("{}/{}".format(dst_root, fn)):
    os.system("gdown {}".format(gdrive_path))
    if iszip:
      os.system("unzip {}.zip".format(fn))
      os.system("rm {}.zip".format(fn))
    os.system("mv {} {}/".format(fn, dst_root))
  print("download {}.".format(fn))

## download data
# https://drive.google.com/drive/folders/11wWszW1kskAyMIGJHBBZzHNKN3os6pu_
download_from_gdrive(args.dataroot, "testM_lip", "1toeQwAe57LNPTy9EWGG0u1XfTI7qv6b1")
download_from_gdrive(args.dataroot, "trainM_lip", "1OAsHXiyQRGCCZltWtBUj_y4xV8aBKLk5")
download_from_gdrive(args.dataroot,"standard_test_anns.txt","19nJSHrQuoJZ-6cSl3WEYlhQv6ZsAYG-X",iszip=False)

# DeepFashion-MultiModal https://github.com/yumingj/DeepFashion-MultiModal
download_from_gdrive(args.dataroot, "images", "1U2PljA7NE57jcSSzPs21ZurdIPXdYZtN")

# https://drive.google.com/drive/folders/1BX3Bxh8KG01yKWViRY0WTyDWbJHju-SL
download_from_gdrive(args.dataroot,"fasion-pairs-test.csv","12fZKGf0kIu5OX3mjC-C3tptxrD8sxm7x",iszip=False)
download_from_gdrive(args.dataroot,"fasion-annotation-test.csv","1MxkVFFtNsWFshQp_TA7qwIGEUEUIpYdS",iszip=False)
download_from_gdrive(args.dataroot,"fasion-annotation-train.csv","1CkINRpr4L7E-YCEbBE_RS-ainIHn8v3P",iszip=False)
download_from_gdrive(args.dataroot,"fasion-pairs-train.csv","13FrzVtjo0bRFJZn3f5VrV2R0FSWHikNF",iszip=False)
download_from_gdrive(args.dataroot,"test.lst","1yi7xg1nJ9Ts5RbA_WsKq5lDUCwrXwGyn",iszip=False)
download_from_gdrive(args.dataroot,"train.lst","1sbIw7M-CpLlT9L1kQfkdami-IPoHHWpC",iszip=False)

# filter images (exclude training data and rename the files)
if not os.path.exists(args.dataroot + "/test"):
  os.mkdir(args.dataroot + "/test")
if not os.path.exists(args.dataroot + "/train"):
  os.mkdir(args.dataroot + "/train")
target_fns = [fn[:-4] for fn in os.listdir(args.dataroot + "/testM_lip")]
for fn in tqdm(os.listdir(args.dataroot + "/images")):
  if not fn.endswith(".jpg"):
    continue
  elements = fn.split("-")
  elements[2] = elements[2].replace("_","")
  last_elements = elements[-1].split("_")
  elements[-1] = last_elements[0] + "_" + last_elements[1] + last_elements[2]
  new_fn = "fashion"+"".join(elements)

  if new_fn[:-4] in target_fns:
    os.system("mv {} {}".format(args.dataroot + "/images/"+fn, args.dataroot + "/test/"+new_fn))
  else:
    os.system("mv {} {}".format(args.dataroot + "/images/"+fn, args.dataroot + "/train/"+new_fn))
  os.system("rm -rf {}/images".format(args.dataroot))

