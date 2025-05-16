#!/bin/bash
start=$(date +%s)

export cuda=1

export P="a luxurious CG scene with a couch, coffee table, small art piece, a cozy rug."
export P1="'a couch: luxurious, computer-generated'"
export P2="'a coffee table: luxurious, computer-generated'"
export P3="'art piece: small, decorative'"
export P4="'a rug: cozy, soft texture'"
export N_obj=4
export PO=[["$P1"],["$P2"],["$P3"],["$P4"]]
export NP="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"


export P12="Couch is behind Coffee Table"
export P24="Coffee Table is on Rug"
export P32="Art Piece is on Coffee Table"
export P42="Rug is under Coffee Table"

export PG=[["$P12"],["$P24"],["$P32"],["$P42"]]
export E=[[0,1],[1,3],[2,1],[3,1]]
# export C=[[-0.25,0.1,0.],[0.24,0.12,0.],[0.25,0.13,0.2],[0.28,-0.16,0.2]]
# export R=[0.5,0.5,0.3,0.3]

# Name save folder:
export TG="02_sofa_coffeetable"

# 1. Coarse stage:
python launch.py --config configs/gd-if.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$P" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$PO" system.prompt_global="$PG" system.edge_list=$E system.guidance.guidance_scale=[200.,100.] system.guidance.guidance_scale_milestones=[2000,] system.optimizer.params.geometry.lr=0.01 data.resolution_milestones=[2000,] trainer.max_steps=4600

# 2. Fine stage:
export RP="a 4K DSLR high-resolution high-quality photo of "$P""
export RP1="'a 4K DSLR high-resolution high-quality photo of a couch: luxurious, computer-generated'"
export RP2="'a 4K DSLR high-resolution high-quality photo of a coffee table: luxurious, computer-generated'"
export RP3="'a 4K DSLR high-resolution high-quality photo of an art piece: small, decorative'"
export RP4="'a 4K DSLR high-resolution high-quality photo of a rug: cozy, soft texture'"
export RP12="a 4K DSLR high-resolution high-quality photo of "$P12""
export RP24="a 4K DSLR high-resolution high-quality photo of "$P24""
export RP32="a 4K DSLR high-resolution high-quality photo of "$P32""
export RP42="a 4K DSLR high-resolution high-quality photo of "$P42""
export RPO=[["$RP1"],["$RP2"],["$RP3"],["$RP4"]]
export RPG=[["$RP12"],["$RP24"],["$RP32"],["$RP42"]]

# Avoid OOM: data.batch_size=1 data.width=128 data.height=128
python launch.py --config configs/gd-sd-refine.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$RP" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$RPO" system.prompt_global="$RPG" system.edge_list=$E resume=examples/gd-if/$TG/ckpts/last.ckpt data.batch_size=1 data.width=128 data.height=128 trainer.max_steps=10000 trainer.val_check_interval=200

end=$(date +%s)
echo "Total time: $((end - start)) seconds" | tee run_time_$TG.log