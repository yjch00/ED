seed=67
dataset_name="coco"
model_path="LLAVA_PATH"
ed_alpha=0.5
ed_beta=0.5
ed_tau=2

if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder="COCO_IMGAE_PATH"
else
  image_folder="GQA_IMAGE_PATH"
fi

types=("random" "popular" "adversarial" )


for type in "${types[@]}"
do
  python ./eval/llava_ed.py \
  --model-path ${model_path} \
  --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
  --image-folder ${image_folder} \
  --answers-file ./output/${dataset_name}_pope_${type}_ed.jsonl \
  --use_ed \
  --ed_alpha $ed_alpha \
  --ed_beta $ed_beta \
  --ed_tau $ed_tau \
  --seed ${seed}
done
