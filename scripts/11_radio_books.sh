#!/bin/bash
start=$(date +%s)

export cuda=0

export P="A vintage wooden radio with a small cow figurine on top sits on a stack of three hardcover books, next to a wooden cup holding colorful pencils"

export P1="'Radio: Vintage, wooden'"
export P2="'Cow Figurine: Small, decorative.'"
export P3="'Book Stack: Three hardcover books.'"
export P4="'Cup with Pencils: Wooden cup, colorful pencils.'"
export N_obj=4
export PO=[["$P1"],["$P2"],["$P3"],["$P4"]]
export NP="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"

export P13="Radio sits on Book Stack "
export P21="Cow Figurine is on top of Radio "
export P43="Cup with Pencils is next to Book Stack"
export P14="Radio is next to Cup with Pencils"

export PG=[["$P13"],["$P21"],["$P43"],["$P14"]]
export E=[[0,2],[1,0],[3,2],[0,3]]
# export C=[[-0.25,0.1,0.],[0.24,0.12,0.],[0.25,0.13,0.2],[0.28,-0.16,0.2]]
# export R=[0.5,0.5,0.3,0.3]

# Name save folder:
export TG="11_radio_books"

# 1. Coarse stage:
python launch.py --config configs/gd-if.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$P" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$PO" system.prompt_global="$PG" system.edge_list=$E system.guidance.guidance_scale=[200.,100.] system.guidance.guidance_scale_milestones=[2000,] system.optimizer.params.geometry.lr=0.01 data.resolution_milestones=[2000,] trainer.max_steps=4600

# 2. Fine stage:
export RP="a 4K DSLR high-resolution high-quality photo of "$P""
export RP1="'a 4K DSLR high-resolution high-quality photo of a radio: vintage, wooden'"
export RP2="'a 4K DSLR high-resolution high-quality photo of a cow figurine: small, decorative.'"
export RP3="'a 4K DSLR high-resolution high-quality photo of a book stack: three hardcover books.'"
export RP4="'a 4K DSLR high-resolution high-quality photo of a cup with pencils: wooden cup, colorful pencils.'"
export RP13="a 4K DSLR high-resolution high-quality photo of "$P13""
export RP21="a 4K DSLR high-resolution high-quality photo of "$P21""
export RP43="a 4K DSLR high-resolution high-quality photo of "$P43""
export RP14="a 4K DSLR high-resolution high-quality photo of "$P14""

export RPO=[["$RP1"],["$RP2"],["$RP3"],["$RP4"]]
export RPG=[["$RP13"],["$RP21"],["$RP43"],["$RP14"]]

# Avoid OOM: data.batch_size=1 data.width=128 data.height=128
python launch.py --config configs/gd-sd-refine.yaml --train --gpu $cuda exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=$N_obj system.prompt_processor.prompt="$RP" system.prompt_processor.negative_prompt="$NP" system.prompt_obj="$RPO" system.prompt_global="$RPG" system.edge_list=$E resume=examples/gd-if/$TG/ckpts/last.ckpt data.batch_size=1 data.width=128 data.height=128 trainer.max_steps=10000 trainer.val_check_interval=200

end=$(date +%s)
echo "Total time: $((end - start)) seconds" | tee run_time_$TG.log