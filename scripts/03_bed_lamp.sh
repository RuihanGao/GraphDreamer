#!/bin/bash
start=$(date +%s)

export cuda=2

export P="a warm setup with a bed, bedside tables, plant, a few books, and a small decorative lamp."
export P1="'a bed: warm, cozy'"
export P2="'bedside tables: wooden, matching'"
export P3="'plant: green, lively'"
export P4="'books: a few, stacked'"
export P5="'decorative lamp: small, warm light'"
export N_obj=5
export PO=[["$P1"],["$P2"],["$P3"],["$P4"],["$P5"]]
export NP="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"

export P12="Bed is between Bedside Tables"
export P21="Bedside Tables are beside Bed"
export P52="Lamp is on Bedside Table"
export P42="Books are on Bedside Table"
export P31="Plant is near Bed"
export P54="Lamp is next to Books"

export PG=[["$P12"],["$P52"],["$P42"],["$P31"],["$P54"]] # ["$P21"],
export E=[[0,1],[4,1],[3,1],[2,0],[4,3]] # [1,0],
# export C=[[-0.25,0.1,0.],[0.24,0.12,0.],[0.25,0.13,0.2],[0.28,-0.16,0.2]]
# export R=[0.5,0.5,0.3,0.3]

# Name save folder:
export TG="03_bed_lamp"

# 1. Coarse stage:
python launch.py --config configs/gd-if.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$P" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$PO" system.prompt_global="$PG" system.edge_list=$E system.guidance.guidance_scale=[200.,100.] system.guidance.guidance_scale_milestones=[2000,] system.optimizer.params.geometry.lr=0.01 data.resolution_milestones=[2000,] trainer.max_steps=4600

# 2. Fine stage:
export RP="a 4K DSLR high-resolution high-quality photo of "$P""
export RP1="'a 4K DSLR high-resolution high-quality photo of a bed: warm, cozy'"
export RP2="'a 4K DSLR high-resolution high-quality photo of bedside tables: wooden, matching'"
export RP3="'a 4K DSLR high-resolution high-quality photo of a plant: green, lively'"
export RP4="'a 4K DSLR high-resolution high-quality photo of books: a few, stacked'"
export RP5="'a 4K DSLR high-resolution high-quality photo of a decorative lamp: small, warm light'"
export RP12="a 4K DSLR high-resolution high-quality photo of "$P12""
export RP21="a 4K DSLR high-resolution high-quality photo of "$P21""
export RP52="a 4K DSLR high-resolution high-quality photo of "$P52""
export RP42="a 4K DSLR high-resolution high-quality photo of "$P42""
export RP31="a 4K DSLR high-resolution high-quality photo of "$P31""
export RP54="a 4K DSLR high-resolution high-quality photo of "$P54""
export RPO=[["$RP1"],["$RP2"],["$RP3"],["$RP4"],["$RP5"]]
export RPG=[["$RP12"],["$RP52"],["$RP42"],["$RP31"],["$RP54"]] # ["$RP21"],

# Avoid OOM: data.batch_size=1 data.width=128 data.height=128
python launch.py --config configs/gd-sd-refine.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$RP" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$RPO" system.prompt_global="$RPG" system.edge_list=$E resume=examples/gd-if/$TG/ckpts/last.ckpt data.batch_size=1 data.width=128 data.height=128 trainer.max_steps=10000 trainer.val_check_interval=200

end=$(date +%s)
echo "Total time: $((end - start)) seconds" | tee run_time_$TG.log