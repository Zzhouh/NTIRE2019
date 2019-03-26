requirements:

1.anaconda3

2.other packages: pytorch 0.4, tqdm, glob, imageio, matplotlib



please run the command:

CUDA_VISIBLE_DEVICES="0" python recon.py --scale 1 --pre_train ../experiment/model/model/model_best.pt  --model RAN2 --dir_data ${testdir}/Test_LR --n_resblocks 4 --n_resgroups 12 --chop --nvis --n_GPUs 1

then the HR images will be generated in the directory "experiment/KR_result"