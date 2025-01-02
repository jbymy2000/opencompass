from mmengine.config import read_base

with read_base():
    #from opencompass.configs.models.qwen.hf_qwen_7b_chat import beyond_dialogue
    #from .models.hf_llama.hf_llama3_8b_instruct import models as llama3_80
    from .models.chatglm.hf_chatglm3_6b import models as hf_chatglm3_6b
    #from .models.qwen2_5.lmdeploy_qwen2_5_model_stock_0_2 import models as lmdeploy_qwen2_5
    #print("模型导入成功")
    #from .datasets.FewCLUE_cluewsc.FewCLUE_cluewsc_gen import cluewsc_datasets
    #from .datasets.CLUE_C3.CLUE_C3_gen import C3_datasets
    from .datasets.rolebench.instruction_generalization_zh import instruction_generalization_zh_datasets
    from .datasets.rolebench.instruction_generalization_eng import instruction_generalization_eng_datasets
    #from .datasets.rolebench.role_generalization_eng import role_generalization_eng_datasets
    from .datasets.IFEval.IFEval_gen import ifeval_datasets



    #from .datasets.cmmlu.cmmlu_gen import cmmlu_datasets
    #from .datasets.ceval.ceval_gen import ceval_datasets
    #from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    #from .datasets.FewCLUE_chid.FewCLUE_chid_ppl import chid_datasets
    


datasets = []
datasets += instruction_generalization_zh_datasets
models = hf_chatglm3_6b
