#!/bin/bash
start=$(date +%s)

export cuda=2

export P="On a table, there is a vase with a bouquet of flowers. Beside it, there is a plate of cake"

export P1="'Table: Flat, sturdy'"
export P2="'Vase: Tall, decorative'"
export P3="'Bouquet of Flowers: Colorful, fresh'"
export P4="'Plate: Round, ceramic'"
export P5="'Cake: Sweet, layered'"
export N_obj=5
export PO=[["$P1"],["$P2"],["$P3"],["$P4"],["$P5"]]
export NP="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"

export P21="Vase is on Table"
export P32="Bouquet of Flowers is inside Vase"
export P41="Plate is on Table"
export P13="Cake is on the Plate"
export P42="Plate is beside Vase"
export P52="Cake is beside Vase"

export PG=[["$P21"],["$P32"],["$P41"],["$P13"],["$P42"],["$P52"]]
export E=[[1,0],[2,1],[3,1],[0,2],[3,1],[4,1]]
# export C=[[-0.25,0.1,0.],[0.24,0.12,0.],[0.25,0.13,0.2],[0.28,-0.16,0.2]]
# export R=[0.5,0.5,0.3,0.3]

# Name save folder:
export TG="07_vase_table"

# 1. Coarse stage:
python launch.py --config configs/gd-if.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$P" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$PO" system.prompt_global="$PG" system.edge_list=$E system.guidance.guidance_scale=[200.,100.] system.guidance.guidance_scale_milestones=[2000,] system.optimizer.params.geometry.lr=0.01 data.resolution_milestones=[2000,] trainer.max_steps=4600

# 2. Fine stage:
export RP="a 4K DSLR high-resolution high-quality photo of "$P""
export RP1="'a 4K DSLR high-resolution high-quality photo of a Table: Flat, sturdy'"
export RP2="'a 4K DSLR high-resolution high-quality photo of a Vase: Tall, decorative'"
export RP3="'a 4K DSLR high-resolution high-quality photo of a Bouquet of Flowers: Colorful, fresh'"
export RP4="'a 4K DSLR high-resolution high-quality photo of a Plate: Round, ceramic'"
export RP5="'a 4K DSLR high-resolution high-quality photo of a Cake: Sweet, layered'"
export RP21="a 4K DSLR high-resolution high-quality photo of a Vase on a Table"
export RP32="a 4K DSLR high-resolution high-quality photo of a Bouquet of Flowers inside a Vase"
export RP41="a 4K DSLR high-resolution high-quality photo of a Plate on a Table"
export RP13="a 4K DSLR high-resolution high-quality photo of a Cake on the Plate"
export RP42="a 4K DSLR high-resolution high-quality photo of a Plate beside a Vase"
export RP52="a 4K DSLR high-resolution high-quality photo of a Cake beside a Vase"

export RPO=[["$RP1"],["$RP2"],["$RP3"],["$RP4"],["$RP5"]]
export RPG=[["$RP21"],["$RP32"],["$RP41"],["$RP13"],["$RP42"],["$RP52"]]

# Avoid OOM: data.batch_size=1 data.width=128 data.height=128
python launch.py --config configs/gd-sd-refine.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$RP" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$RPO" system.prompt_global="$RPG" system.edge_list=$E resume=examples/gd-if/$TG/ckpts/last.ckpt data.batch_size=1 data.width=128 data.height=128 trainer.max_steps=10000 trainer.val_check_interval=200

end=$(date +%s)
echo "Total time: $((end - start)) seconds" | tee run_time_$TG.log