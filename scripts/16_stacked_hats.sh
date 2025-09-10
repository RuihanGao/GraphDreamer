#!/bin/bash
start=$(date +%s)

export cuda=2

export P="a cute plush toy dog with five different colored hats stacked on its head"

export P1="'Plush Toy Dog: Cute, soft.'"
export P2="'Red and Blue Hats: Colorful, stacked.'"
export P3="'Green and Yellow Hats: Bright, stacked.'"
export P4="'Purple Hat: Distinct, topmost.'"
export N_obj=4
export PO=[["$P1"],["$P2"],["$P3"],["$P4"]]
export NP="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"

export P32="Green and Yellow Hats stack on top of Red and Blue Hats"
export P43="Purple Hat stacks on top of Green and Yellow Hats"
export P21="Red and Blue Hats are placed on Plush Toy Dog"
export P12="Plush Toy Dog is wearing Red and Blue Hats"

export PG=[["$P32"],["$P43"],["$P21"],["$P12"]]
export E=[[2,1],[3,2],[1,0],[0,1]]
# export C=[[-0.25,0.1,0.],[0.24,0.12,0.],[0.25,0.13,0.2],[0.28,-0.16,0.2]]
# export R=[0.5,0.5,0.3,0.3]

# Name save folder:
export TG="16_stacked_hats"

# 1. Coarse stage:
python launch.py --config configs/gd-if.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$P" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$PO" system.prompt_global="$PG" system.edge_list=$E system.guidance.guidance_scale=[200.,100.] system.guidance.guidance_scale_milestones=[2000,] system.optimizer.params.geometry.lr=0.01 data.resolution_milestones=[2000,] trainer.max_steps=4600

# 2. Fine stage:
export RP="a 4K DSLR high-resolution high-quality photo of "$P""
export RP1="'a 4K DSLR high-resolution high-quality photo of Plush Toy Dog.'"
export RP2="'a 4K DSLR high-resolution high-quality photo of Red and Blue Hats: red, blue, soft.'"
export RP3="'a 4K DSLR high-resolution high-quality photo of Green and Yellow Hats: green, yellow, soft.'"
export RP4="'a 4K DSLR high-resolution high-quality photo of Purple Hat: purple, soft.'"
export RP32="a 4K DSLR high-resolution high-quality photo of "$P32""
export RP43="a 4K DSLR high-resolution high-quality photo of "$P43""
export RP21="a 4K DSLR high-resolution high-quality photo of "$P21""
export RP12="a 4K DSLR high-resolution high-quality photo of "$P12""

export RPO=[["$RP1"],["$RP2"],["$RP3"],["$RP4"]]
export RPG=[["$RP32"],["$RP43"],["$RP21"],["$RP12"]]

# Avoid OOM: data.batch_size=1 data.width=128 data.height=128
python launch.py --config configs/gd-sd-refine.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$RP" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$RPO" system.prompt_global="$RPG" system.edge_list=$E resume=examples/gd-if/$TG/ckpts/last.ckpt data.batch_size=1 data.width=128 data.height=128 trainer.max_steps=10000 trainer.val_check_interval=200

end=$(date +%s)
echo "Total time: $((end - start)) seconds" | tee run_time_$TG.log