export NGF=32
export DATAROOT=...
export NET_G=dior
export NAME=dior
export PRETRAINED_FLOWNET_PATH=...

python train.py --model dior \
--name $NAME --dataroot $DATAROOT \
--batch_size 8 --lr 1e-4 --init_type orthogonal \
--loss_coe_seg 0 \
--netG $NET_G --ngf $NGF \
--netD gfla --ndf 32 --n_layers_D 4 \
--n_epochs 20002 --n_epochs_decay 0 --lr_update_unit 4000 \
--print_freq 200 --display_freq 10000 --save_epoch_freq 10000 --save_latest_freq 2000 \
--n_cpus 8 --gpu_ids 0 \
--flownet_path $PRETRAINED_FLOWNET_PATH --frozen_flownet \
--random_rate 0 --warmup --perturb

rm -rf checkpoints/$NAME/latest_net_D*

python train.py --model dior \
--name $NAME --dataroot $DATAROOT \
--batch_size 8 --lr 1e-4 --init_type orthogonal \
--loss_coe_seg 0.1 \
--netG $NET_G --ngf $NGF \
--netD gfla --ndf 32 --n_layers_D 4 \
--n_epochs 160002 --n_epochs_decay 0 --lr_update_unit 4000 \
--print_freq 200 --display_freq 10000 --save_epoch_freq 10000 --save_latest_freq 2000 \
--n_cpus 8 --gpu_ids 0 --continue_train \
--flownet_path $PRETRAINED_FLOWNET_PATH --frozen_flownet \
--random_rate 0.8 --perturb


rm -rf checkpoints/$NAME/latest_net_D*

python train.py --model dior \
--name $NAME --dataroot $DATAROOT \
--batch_size 8 --lr 1e-5 --init_type orthogonal \
--loss_coe_seg 0.1 \
--netG $NET_G --ngf $NGF \
--netD gfla --ndf 32 --n_layers_D 4 \
--n_epochs 240002 --n_epochs_decay 60000 --lr_update_unit 4000 \
--print_freq 200 --display_freq 10000 --save_epoch_freq 10000 --save_latest_freq 2000 \
--n_cpus 8 --gpu_ids 0 --continue_train \
--epoch iter_160000 --epoch_count 160001 \
--flownet_path $PRETRAINED_FLOWNET_PATH \
--random_rate 0.8 --perturb


#iter 80k random 25 -> 80
#iter 120k lr 1e-4 -> 5e-5
