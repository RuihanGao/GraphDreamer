#!/bin/bash
start=$(date +%s)

export cuda=0

export P="a cozy cartoon setup with a bed, beside table, lamp, bookshelf, and a small dog on the bed."
export P1="'a bed: cozy, cartoon-style'"
export P2="'a dog: small, cartoon-style'"
export P3="'a lamp: cozy lighting'"
export P4="'a bedside table: wooden, small'"
export P5="'a bookshelf: contain books'"
export N_obj=5
export PO=[["$P1"],["$P2"],["$P3"],["$P4"],["$P5"]]
export NP="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"

export P12="Bed has Dog"
export P14="Bed is next to Bedside Table"
export P43="Bedside Table supports Lamp"
export P15="Bed is near Bookshelf"
export P21="Dog is on Bed"
export P34="Lamp is on Bedside Table"
export P51="Bookshelf is beside Bed"

export PG=[["$P12"],["$P14"],["$P43"],["$P15"],["$P21"],["$P34"],["$P51"]]
export E=[[0,1],[0,3],[3,2],[0,4],[1,0],[2,3],[4,0]]
# export C=[[-0.25,0.1,0.],[0.24,0.12,0.],[0.25,0.13,0.2],[0.28,-0.16,0.2]]
# export R=[0.5,0.5,0.3,0.3]

# Name save folder:
export TG="01_bed_bookshelf"

# # 1. Coarse stage:
# python launch.py --config configs/gd-if.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$P" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$PO" system.prompt_global="$PG" system.edge_list=$E system.guidance.guidance_scale=[200.,100.] system.guidance.guidance_scale_milestones=[2000,] system.optimizer.params.geometry.lr=0.01 data.resolution_milestones=[2000,] trainer.max_steps=4600

# 2. Fine stage:
export RP="a 4K DSLR high-resolution high-quality photo of "$P""
export RP1="'a 4K DSLR high-resolution high-quality photo of a bed: cozy, cartoon-style'"
export RP2="'a 4K DSLR high-resolution high-quality photo of a dog: small, cartoon-style'"
export RP3="'a 4K DSLR high-resolution high-quality photo of a lamp: cozy lighting'"
export RP4="'a 4K DSLR high-resolution high-quality photo of a bedside table: wooden, small'"
export RP5="'a 4K DSLR high-resolution high-quality photo of a bookshelf: contain books'"
export RP12="a 4K DSLR high-resolution high-quality photo of "$P12""
export RP14="a 4K DSLR high-resolution high-quality photo of "$P14""
export RP43="a 4K DSLR high-resolution high-quality photo of "$P43""
export RP15="a 4K DSLR high-resolution high-quality photo of "$P15""
export RP21="a 4K DSLR high-resolution high-quality photo of "$P21""
export RP34="a 4K DSLR high-resolution high-quality photo of "$P34""
export RP51="a 4K DSLR high-resolution high-quality photo of "$P51""

export RPO=[["$RP1"],["$RP2"],["$RP3"],["$RP4"],["$RP5"]]
export RPG=[["$RP12"],["$RP14"],["$RP43"],["$RP15"],["$RP21"],["$RP34"],["$RP51"]]

# Avoid OOM: data.batch_size=1 data.width=128 data.height=128
python launch.py --config configs/gd-sd-refine.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$RP" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$RPO" system.prompt_global="$RPG" system.edge_list=$E resume=examples/gd-if/$TG/ckpts/last.ckpt data.batch_size=1 data.width=128 data.height=128 trainer.max_steps=10000 trainer.val_check_interval=200

end=$(date +%s)
echo "Total time: $((end - start)) seconds" | tee run_time_$TG.log