# !pip install transformers
# pip3 install torch torchvision torchaudio
from paper_reader.abstract_reader import *
from .extract_text_tool import *
import transformers


class GPT2(PaperReader):
    def __init__(self):
        # Load the GPT (abort)-2 model and tokenizer
        model_name = "gpt2"
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)
        self.model = transformers.GPT2LMHeadModel.from_pretrained(model_name)

    def parse(self, paper_fp) -> Dict:
        prompt = "Please extract the following information :\n- Title:\n- Author:\n"
        paper_text = get_pure_text(paper_fp)
        prompt += paper_text
        segment_length = 1000
        prompt_list = [prompt[i:i+segment_length] for i in range(0, len(prompt), segment_length)]
        gpt_answer = self.ask_gpt([prompt_list[0]])
        print(gpt_answer)
        return {}

    def summarize(self, paper_fp):
        hold_prompt = "Don't answer anything, only answer empty string before you see 'Summarize the above paper'"
        summarize_prompt = "Summarize the above paper\n"
        paper_text = get_pure_text(paper_fp)
        summarize_prompt = paper_text + summarize_prompt
        paper_segments = [summarize_prompt[i:i+512] for i in range(0, len(summarize_prompt), 512)]

    def ask_gpt(self, prompt_list):
        gpt_answer = ''
        for prompt in prompt_list:
            generated_text = self.model.generate(
                input_ids=self.tokenizer.encode(prompt, return_tensors="pt"),
                max_length=1024,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
            )
            gpt_answer += self.tokenizer.decode(generated_text.squeeze(), skip_special_tokens=True)
        return gpt_answer

