#!/bin/bash
start=$(date +%s)

export cuda=2

export P="A wooden table with a red fire extinguisher sits in front of a metal shelving unit with a gray cap on one of the shelves, flanked by large cardboard sheets on one side and a wheeled cart holding potted green plants on the other"

export P1="'Wooden Table with Fire Extinguisher: Wooden, red fire extinguisher.'"
export P2="'Metal Shelving Unit with Cap: Metal, gray cap'"
export P3="'Cardboard Sheets: Large, flat.'"
export P4="'Wheeled Cart with Potted Plants: Wheeled, potted green plants.'"
export N_obj=4
export PO=[["$P1"],["$P2"],["$P3"],["$P4"]]
export NP="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"

export P12="Wooden Table with Fire Extinguisher is in front of Metal Shelving Unit with Cap "
export P32="Cardboard Sheets are beside Metal Shelving Unit with Cap"
export P42="Wheeled Cart with Potted Plants is beside Metal Shelving Unit with Cap"
export P14="Wooden Table with Fire Extinguisher is on the left of Wheeled Cart with Potted Plants"

export PG=[["$P12"],["$P32"],["$P42"],["$P14"]]
export E=[[0,1],[2,1],[3,1],[0,3]]
# export C=[[-0.25,0.1,0.],[0.24,0.12,0.],[0.25,0.13,0.2],[0.28,-0.16,0.2]]
# export R=[0.5,0.5,0.3,0.3]

# Name save folder:
export TG="14_warehouse_cart"

# 1. Coarse stage:
python launch.py --config configs/gd-if.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$P" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$PO" system.prompt_global="$PG" system.edge_list=$E system.guidance.guidance_scale=[200.,100.] system.guidance.guidance_scale_milestones=[2000,] system.optimizer.params.geometry.lr=0.01 data.resolution_milestones=[2000,] trainer.max_steps=4600

# 2. Fine stage:
export RP="a 4K DSLR high-resolution high-quality photo of "$P""
export RP1="'a 4K DSLR high-resolution high-quality photo of a wooden table with a red fire extinguisher: wooden, red fire extinguisher.'"
export RP2="'a 4K DSLR high-resolution high-quality photo of a metal shelving unit with a gray cap: metal, gray cap.'"
export RP3="'a 4K DSLR high-resolution high-quality photo of large cardboard sheets: large, flat.'"
export RP4="'a 4K DSLR high-resolution high-quality photo of a wheeled cart with potted plants: wheeled, potted green plants.'"
export RP12="a 4K DSLR high-resolution high-quality photo of "$P12""
export RP32="a 4K DSLR high-resolution high-quality photo of "$P32""
export RP42="a 4K DSLR high-resolution high-quality photo of "$P42""
export RP14="a 4K DSLR high-resolution high-quality photo of "$P14""

export RPO=[["$RP1"],["$RP2"],["$RP3"],["$RP4"]]
export RPG=[["$RP12"],["$RP32"],["$RP42"],["$RP14"]]

# Avoid OOM: data.batch_size=1 data.width=128 data.height=128
python launch.py --config configs/gd-sd-refine.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$RP" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$RPO" system.prompt_global="$RPG" system.edge_list=$E resume=examples/gd-if/$TG/ckpts/last.ckpt data.batch_size=1 data.width=128 data.height=128 trainer.max_steps=10000 trainer.val_check_interval=200

end=$(date +%s)
echo "Total time: $((end - start)) seconds" | tee run_time_$TG.log