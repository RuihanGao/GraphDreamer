#!/bin/bash
start=$(date +%s)

# add cuda device
export cuda=1
export P="Horizontal perspective, showing the complete objects. A headphone stand with a headphone on it."
export P1="a headphone stand"
export P2="a headphone"
export CD=0.

export TG="headphone_stand"

# # 1. Coarse stage:
# python launch.py --config configs/gd-if.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=0. system.geometry.num_objects=2 system.prompt_processor.prompt="$P" system.prompt_processor.front_threshold=45. system.prompt_processor.back_threshold=45. system.prompt_obj=[["$P1"],["$P2"]] system.prompt_obj_neg=[["$P2"],["$P1"]] system.obj_use_view_dependent=true system.geometry.sdf_center_dispersion=$CD system.guidance.guidance_scale=[50.,20.] system.guidance.guidance_scale_milestones=[2000,] system.optimizer.params.geometry.lr=0.001 

# 2. Fine stage:
export RP="a 4K DSLR photo of "$P", high-resolution high-quality"
export RP1="a 4K DSLR photo of "$P1", high-resolution high-quality"
export RP2="a 4K DSLR photo of "$P2", high-resolution high-quality"

# python launch.py --config configs/gd-sd-refine.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=0. system.geometry.num_objects=2 system.prompt_processor.prompt="$RP" system.prompt_obj=[["$RP1"],["$RP2"]] system.obj_use_view_dependent=true system.prompt_obj_neg=[["$P2"],["$P1"]] system.geometry.sdf_center_dispersion=$CD resume=examples/gd-if/$TG/ckpts/epoch=0-step=10000.ckpt

# end=$(date +%s)
# echo "Total time: $((end - start)) seconds" | tee run_time_$TG.log


# 3. Export stage: run test to export 3D output to mesh
# NOTE: things to change compared to Stage 2: (1) "--test" tag (2) remove "system.loss.lambda_entropy=0." (3) resume path
python launch.py --config configs/gd-sd-refine.yaml --test --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.geometry.num_objects=2 system.prompt_processor.prompt="$RP" system.prompt_obj=[["$RP1"],["$RP2"]] system.obj_use_view_dependent=true system.prompt_obj_neg=[["$P2"],["$P1"]] system.geometry.sdf_center_dispersion=$CD resume=examples/gd-sd-refine/$TG/ckpts/epoch=0-step=20000.ckpt
