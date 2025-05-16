#!/bin/bash
start=$(date +%s)

export cuda=3

export P="a blue jay standing on a large basket of rainbow macarons"

export P1="'Blue Jay: Bright blue, small'"
export P2="'Basket: Large, woven'"
export P3="'Macarons: Rainbow-colored, assorted'"
export N_obj=3
export PO=[["$P1"],["$P2"],["$P3"]]
export NP="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"

export P12="Blue Jay is standing on Basket"
export P23="Basket contains Macarons"
export P32="Macarons are inside Basket "
export P13="Blue Jay is above Macarons"

export PG=[["$P12"],["$P23"],["$P32"],["$P13"]]
export E=[[0,1],[1,2],[2,1],[0,2]]
# export C=[[-0.25,0.1,0.],[0.24,0.12,0.],[0.25,0.13,0.2],[0.28,-0.16,0.2]]
# export R=[0.5,0.5,0.3,0.3]

# Name save folder:
export TG="06_bluejay_macaron"

# 1. Coarse stage:
python launch.py --config configs/gd-if.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$P" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$PO" system.prompt_global="$PG" system.edge_list=$E system.guidance.guidance_scale=[200.,100.] system.guidance.guidance_scale_milestones=[2000,] system.optimizer.params.geometry.lr=0.01 data.resolution_milestones=[2000,] trainer.max_steps=4600

# 2. Fine stage:
export RP="a 4K DSLR high-resolution high-quality photo of "$P""
export RP1="'a 4K DSLR high-resolution high-quality photo of a Blue Jay: Bright blue, small'"
export RP2="'a 4K DSLR high-resolution high-quality photo of a Basket: Large, woven'"
export RP3="'a 4K DSLR high-resolution high-quality photo of Macarons: Rainbow-colored, assorted'"
export RP12="a 4K DSLR high-resolution high-quality photo of "$P12""
export RP23="a 4K DSLR high-resolution high-quality photo of "$P23""
export RP32="a 4K DSLR high-resolution high-quality photo of "$P32""
export RP13="a 4K DSLR high-resolution high-quality photo of "$P13""
export RPO=[["$RP1"],["$RP2"],["$RP3"]]

export RPG=[["$RP12"],["$RP23"],["$RP32"],["$RP13"]]

# Avoid OOM: data.batch_size=1 data.width=128 data.height=128
python launch.py --config configs/gd-sd-refine.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$RP" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$RPO" system.prompt_global="$RPG" system.edge_list=$E resume=examples/gd-if/$TG/ckpts/last.ckpt data.batch_size=1 data.width=128 data.height=128 trainer.max_steps=10000 trainer.val_check_interval=200

end=$(date +%s)
echo "Total time: $((end - start)) seconds" | tee run_time_$TG.log