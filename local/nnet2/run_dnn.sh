#!/bin/bash
# Modified from kaldi/egs/wsj/ for use with the voxforge_ru dataset

stage=0
train_stage=-100
temp_dir=
# This trains only unadapted (just cepstral mean normalized) features,
# and uses various combinations of VTLN warping factor and time-warping
# factor to artificially expand the amount of data.




. ./cmd.sh
. ./path.sh
! cuda-compiled && cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
. utils/parse_options.sh  # to parse the --stage option, if given

if [ $# != 0 ]; then
  echo "Usage: $0 [--stage <stage> --train-stage <train-stage>]"
  echo "Options: "
  echo "    --stage <stage>              # controls partial re-runs"
  echo "    --train-stage <train-stage>  #  use with --stage 2 to control partial rerun of training"
  echo "    --temp-dir <temp-dir>        # e.g. --temp-dir /export/my-machine/dpovey/wsj-temp-5b"
  echo "                                 # (puts temporary data including MFCC and egs at this location)"
  exit 1;
fi

set -e

if [ $stage -le 0 ]; then 
  echo "Stage 0"
  # Create the training data.
  
  if [ ! -z "$temp_dir" ]; then
    mkdir -p $temp_dir/mfcc_5b_gpu
    featdir=$temp_dir/mfcc_5b_gpu
  else
    featdir=`pwd`/mfcc/nnet5b_gpu; 
    mkdir -p $featdir
  fi
  fbank_conf=conf/fbank_40.conf
  echo "--num-mel-bins=40" > $fbank_conf
  steps/nnet2/get_perturbed_feats.sh --cmd "$train_cmd" \
    $fbank_conf $featdir exp/perturbed_fbanks data/train data/train_perturbed_fbank &
  steps/nnet2/get_perturbed_feats.sh --cmd "$train_cmd" --feature-type mfcc \
    conf/mfcc.conf $featdir exp/perturbed_mfcc data/train data/train_perturbed_mfcc &
  wait
fi

if [ $stage -le 1 ]; then
  echo "Stage 1"
  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    data/train_perturbed_mfcc data/lang exp/tri3b exp/tri3b_ali_perturbed_mfcc
fi 

if [ $stage -le 2 ]; then
  echo "Stage 2"
  if [ ! -z "$temp_dir" ] && [ ! -e exp/nnet5b_gpu/egs ]; then
    mkdir -p exp/nnet5b_gpu
    mkdir -p $temp_dir/nnet5b_gpu/egs
    ln -s $temp_dir/nnet5b_gpu/egs exp/nnet5b_gpu/
  fi

  steps/nnet2/train_block.sh --stage "$train_stage" \
     --num-threads 1 --max-change 40.0 --minibatch-size 512 --num-jobs-nnet 8 \
     --parallel-opts "--gpu 1" \
     --initial-learning-rate 0.0075 --final-learning-rate 0.00075 \
     --num-epochs 10 --num-epochs-extra 5 \
     --cmd "$decode_cmd" \
     --hidden-layer-dim 1536 \
     --num-block-layers 3 --num-normal-layers 3 \
      data/train_perturbed_fbank data/lang exp/tri3b_ali_perturbed_mfcc exp/nnet5b_gpu  || exit 1
fi

if [ $stage -le 3 ]; then # create testing fbank data.
  echo "Stage 3"
  featdir=`pwd`/mfcc
  fbank_conf=conf/fbank_40.conf
  for x in test; do 
    rm -r data/${x}_fbank
    cp -r data/$x data/${x}_fbank
    rm -r ${x}_fbank/split* || true
    steps/make_fbank.sh --fbank-config "$fbank_conf" --nj 8 \
      --cmd "$train_cmd" data/${x}_fbank exp/make_fbank/$x $featdir  || exit 1;
    steps/compute_cmvn_stats.sh data/${x}_fbank exp/make_fbank/$x $featdir  || exit 1;
  done
fi

if [ $stage -le 4 ]; then
  echo "Stage 4"
  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 10 \
     exp/tri3b/graph data/test_fbank exp/nnet5b_gpu/decode_bd_test

fi



exit 0;

