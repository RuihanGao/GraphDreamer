#!/bin/bash
start=$(date +%s)

# add cuda device
export cuda=0
export P="On a table, there is a vase with a bouquet of flowers. Beside it, there is a plate of cake"
export P1="a table"
export P2="a vase with a bouquet of flowers"
export P3="a plate of cake"
export P12="On a table, there is a vase with a bouquet of flowers"
export P23="a plate of cake beside a vase with a bouquet of flowers"
export P13="On a table, there is a plate of cake"
export PG=[["$P12"],["$P23"],["$P13"]]
export E=[[0,1],[1,2],[0,2]]
export CD=0.

export TG="vase_table"

# 1. Coarse stage:
python launch.py --config configs/gd-if.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=0. system.geometry.num_objects=3 system.prompt_processor.prompt="$P" system.prompt_processor.front_threshold=45. system.prompt_processor.back_threshold=45. system.prompt_obj=[["$P1"],["$P2"],["$P3"]] system.prompt_obj_neg=[["$P3"],["$P2"],["$P1"]] system.obj_use_view_dependent=true system.geometry.sdf_center_dispersion=$CD system.guidance.guidance_scale=[50.,20.] system.guidance.guidance_scale_milestones=[2000,] system.optimizer.params.geometry.lr=0.001 system.prompt_global="$PG" system.edge_list=$E

# 2. Fine stage:
export RP="a 4K DSLR photo of "$P", high-resolution high-quality"
export RP1="a 4K DSLR photo of "$P1", high-resolution high-quality"
export RP2="a 4K DSLR photo of "$P2", high-resolution high-quality"
export RP3="a 4K DSLR photo of "$P3", high-resolution high-quality"

export RP12="a 4K DSLR high-resolution high-quality photo of "$P12""
export RP23="a 4K DSLR high-resolution high-quality photo of "$P23""
export RP13="a 4K DSLR high-resolution high-quality photo of "$P13""
export RPG=[["$RP12"],["$RP23"],["$RP13"]]

python launch.py --config configs/gd-sd-refine.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=0. system.geometry.num_objects=3 system.prompt_processor.prompt="$RP" system.prompt_obj=[["$RP1"],["$RP2"],["$RP3"]] system.obj_use_view_dependent=true system.prompt_obj_neg=[["$P3"],["$P2"],["$P1"]] system.geometry.sdf_center_dispersion=$CD resume=examples/gd-if/$TG/ckpts/epoch=0-step=10000.ckpt system.prompt_global="$RPG" system.edge_list=$E

end=$(date +%s)
echo "Total time: $((end - start)) seconds" | tee run_time_$TG.log