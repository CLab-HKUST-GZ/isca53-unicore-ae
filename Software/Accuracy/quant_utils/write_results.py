import os


def write_results(
    ppl: float,
    model_name: str,
    dataset: str,
    wq_bits: int,
    wq_datatype: str,
    wq_groupsize: int,
    result_table: str = "table1",
    result_precision_tag: str = "",
    result_method: str = "",
):
    parts = [p for p in model_name.replace("\\", "/").split("/") if p]
    if len(parts) >= 2:
        model_file = f"{parts[-2]}__{parts[-1]}.txt"
    elif len(parts) == 1:
        model_file = f"{parts[0]}__.txt"
    else:
        model_file = "model__.txt"

    if result_precision_tag and result_method:
        output_path = f"./results/{result_table}/{result_precision_tag}/{result_method}/{model_file}"
    else:
        output_path = f"./results/{dataset}/{model_file}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.writelines(f'{dataset} perplexity: {ppl} \n')
    
    print('Successfully written results. \n\n')
