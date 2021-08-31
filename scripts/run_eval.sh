export OUTPUT_DIR=checkpoints/DIOR_32
export NAME=DIOR_32
export LOAD_EP=latest
export NET_G=dior
export NET_E=adgan
export NGF=32
export ROOT=...
export DATAROOT=...


# generate images
python generate_all.py --model dior --dataroot $DATAROOT \
--name $NAME --epoch $LOAD_EP --eval_output_dir $OUTPUT_DIR  \
--netE $NET_E --netG $NET_G --ngf $NGF \
--n_cpus 4 --gpu_ids 0  --batch_size 8 


# ssim
python tools/compute_ssim.py --output_dir $OUTPUT_DIR'_'$LOAD_EP 

# fid and lpips
export REAL_DIR=$DATA_DIR/ttrain
export GT_DIR=$DATA_DIR/test
export RESULT_DIR=$OUTPUT_DIR'_'$LOAD_EP 

python3 -m  tools.metrics \
--output_path=$RESULT_DIR \
--fid_real_path=$REAL_DIR \
--gt_path=$GT_DIR \
--name=./fashion
