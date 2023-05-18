from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import logging





class ModelGenerate:
    def __init__(self,model_name:str,config:dict,preprocess:callable=None):
        self.model_name = model_name
        self.config = config
        self.model, self.tokenizer = self.load_model_and_tokenizer(model_name)
        if preprocess:
            print(f"Preprocess function...")
            self.preprocess = preprocess

    def load_model_and_tokenizer(self,model_name:str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        if config.model_type == "t5":
            model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)
        if self.config.device:
            model.to(self.config.device)
        return model,tokenizer

    def generate(self,input_prompt:str):
        input_prompt : str = self.preprocess(input_prompt)
        input_ids = self.tokenizer.encode(input_prompt, return_tensors='pt')
        if self.config.device:
            input_ids = input_ids.to(self.config.device)
        output = self.model.generate(input_ids, max_length=self.config.max_len , do_sample=True, top_k=50)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
        

