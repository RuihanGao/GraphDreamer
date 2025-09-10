#!/bin/bash
start=$(date +%s)

export cuda=1

export P="a cozy cartoon setup with a bed, beside table, lamp, bookshelf, and a small dog on the bed."
export P1="'a bed: cozy, cartoon-style'"
export P2="'bedside table and lamp: wooden, small, lamp on top'"
export P3="'a bookshelf: filled with books'"
export P4="'a dog: small, cartoon-style'"
export N_obj=4
export PO=[["$P1"],["$P2"],["$P3"],["$P4"]]
export NP="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"

export P21="Bedside Table and Lamp are beside Bed"
export P41="Dog is on Bed"
export P31="Bookshelf is near Bed"
export P23="Bedside Table and Lamp are near Bookshelf"

export PG=[["$P21"],["$P41"],["$P31"],["$P23"]]
export E=[[1,0],[3,0],[2,0],[1,2]]
# export C=[[-0.25,0.1,0.],[0.24,0.12,0.],[0.25,0.13,0.2],[0.28,-0.16,0.2]]
# export R=[0.5,0.5,0.3,0.3]

# Name save folder:
export TG="01_bed_bookshelf_v2"

# 1. Coarse stage:
python launch.py --config configs/gd-if.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$P" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$PO" system.prompt_global="$PG" system.edge_list=$E system.guidance.guidance_scale=[200.,100.] system.guidance.guidance_scale_milestones=[2000,] system.optimizer.params.geometry.lr=0.01 data.resolution_milestones=[2000,] trainer.max_steps=4600

# 2. Fine stage:
export RP="a 4K DSLR high-resolution high-quality photo of "$P""
export RP1="'a 4K DSLR high-resolution high-quality photo of a bed: cozy, cartoon-style'"
export RP2="'a 4K DSLR high-resolution high-quality photo of bedside table and lamp: wooden, small, lamp on top'"
export RP3="'a 4K DSLR high-resolution high-quality photo of a bookshelf: filled with books'"
export RP4="'a 4K DSLR high-resolution high-quality photo of a dog: small, cartoon-style'"
export RP21="a 4K DSLR high-resolution high-quality photo of "$P21""
export RP41="a 4K DSLR high-resolution high-quality photo of "$P41""
export RP31="a 4K DSLR high-resolution high-quality photo of "$P31""
export RP23="a 4K DSLR high-resolution high-quality photo of "$P23""

export RPO=[["$RP1"],["$RP2"],["$RP3"],["$RP4"]]
export RPG=[["$RP21"],["$RP41"],["$RP31"],["$RP23"]]

# Avoid OOM: data.batch_size=1 data.width=128 data.height=128
python launch.py --config configs/gd-sd-refine.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$RP" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$RPO" system.prompt_global="$RPG" system.edge_list=$E resume=examples/gd-if/$TG/ckpts/last.ckpt data.batch_size=1 data.width=128 data.height=128 trainer.max_steps=10000 trainer.val_check_interval=200

end=$(date +%s)
echo "Total time: $((end - start)) seconds" | tee run_time_$TG.log