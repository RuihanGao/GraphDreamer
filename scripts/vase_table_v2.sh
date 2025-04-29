#!/bin/bash
start=$(date +%s)

# add cuda device
export cuda=1
export P="On a table, there is a vase with a bouquet of flowers. Beside it, there is a plate of cake"
export P1="a table"
export P2="a vase"
export P3="a bouquet of flowers, colorful, fresh"
export P4="a plate"
export P5="a cake, delicious, sweet"

export P12="A vase on a table "
export P23="a bouquet of flowers in a vase"
export P41="a plate on a table"
export P45="a cake on a plate"
export P24="a plate is beside a vase"
export PG=[["$P12"],["$P23"],["$P13"],["$P41"],["$P45"],["$P24"]]
export E=[[0,1],[1,2],[0,2],[3,0],[3,4],[1,3]]
export CD=0.

export TG="vase_table_v3"

# # 1. Coarse stage:
# python launch.py --config configs/gd-if.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=0. system.geometry.num_objects=5 system.prompt_processor.prompt="$P" system.prompt_processor.front_threshold=45. system.prompt_processor.back_threshold=45. system.prompt_obj=[["$P1"],["$P2"],["$P3"],["$P4"],["$P5"],] system.prompt_obj_neg=[["$P5"],["$P4"],["$P3"],["$P2"],["$P1"]] system.obj_use_view_dependent=true system.geometry.sdf_center_dispersion=$CD system.guidance.guidance_scale=[50.,20.] system.guidance.guidance_scale_milestones=[2000,] system.optimizer.params.geometry.lr=0.001 system.prompt_global="$PG" system.edge_list=$E

# 2. Fine stage:
export RP="a 4K DSLR photo of "$P", high-resolution high-quality"
export RP1="a 4K DSLR photo of "$P1", high-resolution high-quality"
export RP2="a 4K DSLR photo of "$P2", high-resolution high-quality"
export RP3="a 4K DSLR photo of "$P3", high-resolution high-quality"
export RP4="a 4K DSLR photo of "$P4", high-resolution high-quality"
export RP5="a 4K DSLR photo of "$P5", high-resolution high-quality"

export RP12="a 4K DSLR high-resolution high-quality photo of "$P12""
export RP23="a 4K DSLR high-resolution high-quality photo of "$P23""
export RP41="a 4K DSLR high-resolution high-quality photo of "$P41""
export RP45="a 4K DSLR high-resolution high-quality photo of "$P45""
export RP24="a 4K DSLR high-resolution high-quality photo of "$P24""
export RPG=[["$RP12"],["$RP23"],["$RP13"],["$RP41"],["$RP45"],["$RP24"]]

# python launch.py --config configs/gd-sd-refine.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=0. system.geometry.num_objects=5 system.prompt_processor.prompt="$RP" system.prompt_obj=[["$RP1"],["$RP2"],["$RP3"],["$RP4"],["$RP5"]] system.obj_use_view_dependent=true system.prompt_obj_neg=[["$P5"],["$P4"],["$P3"],["$P2"],["$P1"]] system.geometry.sdf_center_dispersion=$CD resume=examples/gd-if/$TG/ckpts/epoch=0-step=10000.ckpt system.prompt_global="$RPG" system.edge_list=$E

# end=$(date +%s)
# echo "Total time: $((end - start)) seconds" | tee run_time_$TG.log`



# 3. Export stage: run test to export 3D output to mesh
# NOTE: things to change compared to Stage 2: (1) "--test" tag (2) remove "system.loss.lambda_entropy=0." (3) resume path
python launch.py --config configs/gd-sd-refine.yaml --test --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.geometry.num_objects=5 system.prompt_processor.prompt="$RP" system.prompt_obj=[["$RP1"],["$RP2"],["$RP3"],["$RP4"],["$RP5"]] system.obj_use_view_dependent=true system.prompt_obj_neg=[["$P5"],["$P4"],["$P3"],["$P2"],["$P1"]] system.geometry.sdf_center_dispersion=$CD resume=examples/gd-sd-refine/$TG/ckpts/epoch=0-step=20000.ckpt system.prompt_global="$RPG" system.edge_list=$E