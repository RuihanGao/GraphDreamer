#!/bin/bash
start=$(date +%s)

export cuda=0

export P="a Wizard standing in front of a Wooden Desk, gazing into a Crystal Ball perched atop the Wooden Desk, with a Stack of Ancient Spell Books perched atop the Wooden Desk"

export P1="'Wizard: Mysterious, wise, magical'"
export P2="'a Wooden Desk: Old, rustic, wooden'"
export P3="'a Crystal Ball: Glowing, mystical'"
export P4="'Ancient Spell Books: Dusty, leather-bound, aged'"
export N_obj=4
export PO=[["$P1"],["$P2"],["$P3"],["$P4"]]
export NP="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"

# scene graph output from chatgpt
# export P12="Wizard is standing in front of Wooden Desk"
# export P23="Wooden Desk supports Crystal Ball"
# export P24="Wooden Desk supports Ancient Spell Books"
# export P34="Crystal Ball is beside Ancient Spell Books"

# copied from wizard_study.sh
export P12="a Wizard standing before a Wooden Desk, cartoon, blender"
export P23="a Crystal Ball perched atop a Wooden Desk, cartoon, blender"
export P24="a Stack of Ancient Spell Books perched atop a Wooden Desk, cartoon, blender"
export P34="a Stack of Ancient Spell Books next to a Crystal Ball, cartoon, blender"

export PG=[["$P12"],["$P23"],["$P24"],["$P34"]] # ["$P32"],["$P42"],["$P13"],
export E=[[0,1],[1,2],[1,3],[2,3]]
# export C=[[-0.25,0.1,0.],[0.24,0.12,0.],[0.25,0.13,0.2],[0.28,-0.16,0.2]]
# export R=[0.5,0.5,0.3,0.3]

# Name save folder:
export TG="05_wizard_study_v2"

# 1. Coarse stage:
python launch.py --config configs/gd-if.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$P" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$PO" system.prompt_global="$PG" system.edge_list=$E system.guidance.guidance_scale=[200.,100.] system.guidance.guidance_scale_milestones=[2000,] system.optimizer.params.geometry.lr=0.01 data.resolution_milestones=[2000,] trainer.max_steps=4600

# 2. Fine stage:
export RP="a 4K DSLR high-resolution high-quality photo of "$P""
export RP1="'a 4K DSLR high-resolution high-quality photo of a Wizard: Mysterious, wise, magical'"
export RP2="'a 4K DSLR high-resolution high-quality photo of a Wooden Desk: Old, rustic, wooden'"
export RP3="'a 4K DSLR high-resolution high-quality photo of a Crystal Ball: Glowing, mystical'"
export RP4="'a 4K DSLR high-resolution high-quality photo of Ancient Spell Books: Dusty, leather-bound, aged'"
export RP12="a 4K DSLR high-resolution high-quality photo of "$P12""
export RP23="a 4K DSLR high-resolution high-quality photo of "$P23""
export RP24="a 4K DSLR high-resolution high-quality photo of "$P24""
export RP34="a 4K DSLR high-resolution high-quality photo of "$P34""
export RPO=[["$RP1"],["$RP2"],["$RP3"],["$RP4"]]
export RPG=[["$RP12"],["$RP23"],["$RP24"],["$RP34"]]

# Avoid OOM: data.batch_size=1 data.width=128 data.height=128
python launch.py --config configs/gd-sd-refine.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$RP" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$RPO" system.prompt_global="$RPG" system.edge_list=$E resume=examples/gd-if/$TG/ckpts/last.ckpt data.batch_size=1 data.width=128 data.height=128 trainer.max_steps=10000 trainer.val_check_interval=200

end=$(date +%s)
echo "Total time: $((end - start)) seconds" | tee run_time_$TG.log