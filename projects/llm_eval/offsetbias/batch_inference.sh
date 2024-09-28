model_name="NCSOFT/Llama-3-OffsetBias-8B"
data_name="NCSOFT/offsetbias"
reverse=True

SET=$(seq 0 4)
for i in $SET
do
    output_dir="./data/test_reverse_${i}.json"
    # Run the evaluation script
    python inference.py --model_path $model_name \
        --input_file $data_name \
        --output_file $output_dir \
        --reverse $reverse
done
