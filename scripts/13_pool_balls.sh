#!/bin/bash
start=$(date +%s)

export cuda=0

export P="A round blue inflatable pool filled with small blue, white, and turquoise plastic balls features a toy sailboat, two yellow rubber ducks, a green turtle, and an orange starfish"

export P1="'Pool: Round, blue, inflatable.'"
export P2="'Plastic Balls: Small, blue, white, turquoise.'"
export P3="'Rubber Toys: Two yellow ducks, green turtle, orange starfish.'"
export P4="'Sailboat: Toy, floating.'"
export N_obj=4
export PO=[["$P1"],["$P2"],["$P3"],["$P4"]]
export NP="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"

export P12="Pool is filled with Plastic Balls "
export P13="Pool contains Rubber Toys"
export P14="Pool contains Sailboat"
export P34="Rubber Toys are beside Sailboat "

export PG=[["$P12"],["$P13"],["$P14"],["$P34"]]
export E=[[0,1],[0,2],[0,3],[2,3]]
# export C=[[-0.25,0.1,0.],[0.24,0.12,0.],[0.25,0.13,0.2],[0.28,-0.16,0.2]]
# export R=[0.5,0.5,0.3,0.3]

# Name save folder:
export TG="13_pool_balls"

# 1. Coarse stage:
python launch.py --config configs/gd-if.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$P" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$PO" system.prompt_global="$PG" system.edge_list=$E system.guidance.guidance_scale=[200.,100.] system.guidance.guidance_scale_milestones=[2000,] system.optimizer.params.geometry.lr=0.01 data.resolution_milestones=[2000,] trainer.max_steps=4600

# 2. Fine stage:
export RP="a 4K DSLR high-resolution high-quality photo of "$P""
export RP1="'a 4K DSLR high-resolution high-quality photo of a pool: round, blue, inflatable.'"
export RP2="'a 4K DSLR high-resolution high-quality photo of plastic balls: small, blue, white, turquoise.'"
export RP3="'a 4K DSLR high-resolution high-quality photo of rubber toys: two yellow ducks, green turtle, orange starfish.'"
export RP4="'a 4K DSLR high-resolution high-quality photo of a sailboat: toy, floating.'"
export RP12="a 4K DSLR high-resolution high-quality photo of "$P12""
export RP13="a 4K DSLR high-resolution high-quality photo of "$P13""
export RP14="a 4K DSLR high-resolution high-quality photo of "$P14""
export RP34="a 4K DSLR high-resolution high-quality photo of "$P34""

export RPO=[["$RP1"],["$RP2"],["$RP3"],["$RP4"]]
export RPG=[["$RP12"],["$RP13"],["$RP14"],["$RP34"]]

# Avoid OOM: data.batch_size=1 data.width=128 data.height=128
python launch.py --config configs/gd-sd-refine.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$RP" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$RPO" system.prompt_global="$RPG" system.edge_list=$E resume=examples/gd-if/$TG/ckpts/last.ckpt data.batch_size=1 data.width=128 data.height=128 trainer.max_steps=10000 trainer.val_check_interval=200

end=$(date +%s)
echo "Total time: $((end - start)) seconds" | tee run_time_$TG.log