from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3-8b-instruct-hf',
        path='/home/xhai/bianjr/hf-models/Higgs-Llama-3-70B',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    )
]
