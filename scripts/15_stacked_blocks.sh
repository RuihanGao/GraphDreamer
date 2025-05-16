#!/bin/bash
start=$(date +%s)

export cuda=3

export P="A stack of colorful wooden blocks arranged vertically, featuring red, blue, yellow, green, orange, and purple pieces, balanced on a flat surface"

export P1="'Red and Blue Blocks: Wooden, primary colors.'"
export P2="'Yellow and Green Blocks: Wooden, bright colors.'"
export P3="'Orange and Purple Blocks: Wooden, vibrant colors.'"
export P4="'Flat Surface: Smooth, even, light-colored.'"
export N_obj=4
export PO=[["$P1"],["$P2"],["$P3"],["$P4"]]
export NP="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"

export P12="Red and Blue Blocks are stacked on Yellow and Green Blocks"
export P23="Yellow and Green Blocks are stacked on Orange and Purple Blocks "
export P34="Orange and Purple Blocks are resting on Flat Surface "
export P14="Red and Blue Blocks are above Flat Surface"

export PG=[["$P12"],["$P23"],["$P34"],["$P14"]]
export E=[[0,1],[1,2],[2,3],[0,3]]
# export C=[[-0.25,0.1,0.],[0.24,0.12,0.],[0.25,0.13,0.2],[0.28,-0.16,0.2]]
# export R=[0.5,0.5,0.3,0.3]

# Name save folder:
export TG="15_stacked_blocks"

# 1. Coarse stage:
python launch.py --config configs/gd-if.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$P" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$PO" system.prompt_global="$PG" system.edge_list=$E system.guidance.guidance_scale=[200.,100.] system.guidance.guidance_scale_milestones=[2000,] system.optimizer.params.geometry.lr=0.01 data.resolution_milestones=[2000,] trainer.max_steps=4600

# 2. Fine stage:
export RP="a 4K DSLR high-resolution high-quality photo of "$P""
export RP1="'a 4K DSLR high-resolution high-quality photo of red and blue blocks: red, blue, wooden.'"
export RP2="'a 4K DSLR high-resolution high-quality photo of yellow and green blocks: yellow, green, wooden.'"
export RP3="'a 4K DSLR high-resolution high-quality photo of orange and purple blocks: orange, purple, wooden.'"
export RP4="'a 4K DSLR high-resolution high-quality photo of a flat surface: flat, smooth, light-colored.'"
export RP12="a 4K DSLR high-resolution high-quality photo of "$P12""
export RP23="a 4K DSLR high-resolution high-quality photo of "$P23""
export RP34="a 4K DSLR high-resolution high-quality photo of "$P34""
export RP14="a 4K DSLR high-resolution high-quality photo of "$P14""

export RPO=[["$RP1"],["$RP2"],["$RP3"],["$RP4"]]
export RPG=[["$RP12"],["$RP23"],["$RP34"],["$RP14"]]

# Avoid OOM: data.batch_size=1 data.width=128 data.height=128
python launch.py --config configs/gd-sd-refine.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$RP" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$RPO" system.prompt_global="$RPG" system.edge_list=$E resume=examples/gd-if/$TG/ckpts/last.ckpt data.batch_size=1 data.width=128 data.height=128 trainer.max_steps=10000 trainer.val_check_interval=200

end=$(date +%s)
echo "Total time: $((end - start)) seconds" | tee run_time_$TG.log