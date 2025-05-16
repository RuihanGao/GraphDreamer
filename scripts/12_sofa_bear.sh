#!/bin/bash
start=$(date +%s)

export cuda=0

export P="A brown leather sofa decorated with plush toys, including a large teddy bear, a gray elephant, a white rabbit, a yellow giraffe, and two throw pillows, sits in a cozy room with two round burgundy floor cushions in front"

export P1="'Sofa: Brown, leather'"
export P2="'Plush Toys and Pillows: Teddy bear, gray elephant, white rabbit, yellow giraffe, throw pillows'"
export P3="'Floor Cushions: Round, burgundy.'"
export N_obj=3
export PO=[["$P1"],["$P2"],["$P3"]]
export NP="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"

export P12="Sofa is decorated with Plush Toys and Pillows "
export P31="Floor Cushions are in front of Sofa"
export P21="Plush Toys and Pillows are on Sofa"


export PG=[["$P12"],["$P31"],["$P21"]]
export E=[[0,1],[2,0],[1,0]]
# export C=[[-0.25,0.1,0.],[0.24,0.12,0.],[0.25,0.13,0.2],[0.28,-0.16,0.2]]
# export R=[0.5,0.5,0.3,0.3]

# Name save folder:
export TG="12_sofa_bear"

# 1. Coarse stage:
python launch.py --config configs/gd-if.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$P" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$PO" system.prompt_global="$PG" system.edge_list=$E system.guidance.guidance_scale=[200.,100.] system.guidance.guidance_scale_milestones=[2000,] system.optimizer.params.geometry.lr=0.01 data.resolution_milestones=[2000,] trainer.max_steps=4600

# 2. Fine stage:
export RP="a 4K DSLR high-resolution high-quality photo of "$P""
export RP1="'a 4K DSLR high-resolution high-quality photo of a sofa: brown, leather'"
export RP2="'a 4K DSLR high-resolution high-quality photo of plush toys and pillows: teddy bear, gray elephant, white rabbit, yellow giraffe, throw pillows.'"
export RP3="'a 4K DSLR high-resolution high-quality photo of floor cushions: round, burgundy.'"
export RP12="a 4K DSLR high-resolution high-quality photo of "$P12""
export RP31="a 4K DSLR high-resolution high-quality photo of "$P31""
export RP21="a 4K DSLR high-resolution high-quality photo of "$P21""

export RPO=[["$RP1"],["$RP2"],["$RP3"]]
export RPG=[["$RP12"],["$RP31"],["$RP21"]]

# Avoid OOM: data.batch_size=1 data.width=128 data.height=128
python launch.py --config configs/gd-sd-refine.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$RP" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$RPO" system.prompt_global="$RPG" system.edge_list=$E resume=examples/gd-if/$TG/ckpts/last.ckpt data.batch_size=1 data.width=128 data.height=128 trainer.max_steps=10000 trainer.val_check_interval=200

end=$(date +%s)
echo "Total time: $((end - start)) seconds" | tee run_time_$TG.log