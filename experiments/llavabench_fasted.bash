seed=67
dataset_name="llavabench"
model_path="LLAVA_PATH"
ed_alpha=0.5
ed_beta=0.5

image_folder="IMAGE_PATH"


python ./eval/llava_fasted.py \
--model-path ${model_path} \
--question-file ./data/llava_bench.json \
--image-folder ${image_folder} \
--answers-file ./output/${dataset_name}_description_fasted.jsonl \
--use_description \
--use_ed \
--ed_alpha $ed_alpha \
--ed_beta $ed_beta \
--seed ${seed}
