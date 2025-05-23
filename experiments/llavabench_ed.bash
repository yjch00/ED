export CUDA_VISIBLE_DEVICES=4

seed=67
dataset_name="llavabench"
model_path="LLAVA_PATH"
ed_alpha=0.5
ed_beta=0.5
ed_tau=4

image_folder="IMAGE_PATH"


python ./eval/llava_ed.py \
--model-path ${model_path} \
--question-file ./data/llava_bench.json \
--image-folder ${image_folder} \
--answers-file ./output/${dataset_name}_description_ed.jsonl \
--use_description \
--use_ed \
--ed_alpha $ed_alpha \
--ed_beta $ed_beta \
--ed_tau $ed_tau \
--seed ${seed}
