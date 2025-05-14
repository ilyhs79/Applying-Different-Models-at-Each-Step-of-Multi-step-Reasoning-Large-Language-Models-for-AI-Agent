# Basic LLM node that calls for a Large Language Model for completion.
import os

import openai

from nodes.Node import Node
from nodes.NodeCofig import *
from utils.util import *
from alpaca.lora import AlpacaLora

openai.api_key = os.environ["OPENAI_API_KEY"]


class LLMNode(Node):
    def __init__(self, name="BaseLLMNode", model_name="gpt-4", stop=None, input_type=str, output_type=str):
        super().__init__(name, input_type, output_type)
        self.model_name = model_name
        self.stop = stop

        # Initialize to load shards only once
        if self.model_name in LLAMA_WEIGHTS:
            self.al = AlpacaLora(lora_weights=self.model_name)

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        
        response = self.call_llm(input, self.stop)
        completion = response["output"]
        #print("응답답답답", completion,"\n")
        if log:
            return response
        
        return completion

    def call_llm(self, prompt, stop):
        
        if self.model_name in OPENAI_COMPLETION_MODELS:
            
            #response = openai.Completion.create(
            response = openai.completions.create(
                model=self.model_name,
                prompt=prompt,
                temperature=OPENAI_CONFIG["temperature"],
                max_tokens=OPENAI_CONFIG["max_tokens"],
                top_p=OPENAI_CONFIG["top_p"],
                frequency_penalty=OPENAI_CONFIG["frequency_penalty"],
                presence_penalty=OPENAI_CONFIG["presence_penalty"],
                stop=stop
                )
            
            print ("사용모델 : ", self.model_name, "\n")
            #print ("질문 : ", prompt, "\n")
            #print ("답변 : ", response, "\n")

            return {"input": prompt,
                    "output": response.choices[0].text,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens}

            
            '''            
            return {"input": prompt,
                    "output": response["choices"][0]["text"],
                    "prompt_tokens": response["usage"]["prompt_tokens"],
                    "completion_tokens": response["usage"]["completion_tokens"]}
            '''

        elif self.model_name in OPENAI_CHAT_MODELS:
            #o1이랑 o1 mini는 형식이 달라서 따로 써줌
            print (self.model_name)
            if self.model_name == 'o1' or self.model_name == 'o1-mini' :                
                messages = [{"role": "user", "content": prompt}]
                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    #temperature=OPENAI_CONFIG["temperature"],
                    #max_tokens=OPENAI_CONFIG["max_tokens"],
                    #top_p=OPENAI_CONFIG["top_p"],
                    #frequency_penalty=OPENAI_CONFIG["frequency_penalty"],
                    #presence_penalty=OPENAI_CONFIG["presence_penalty"],
                    #stop=stop
                    )
                
                print ("사용모델 : ", self.model_name, "\n")
                #print ("질문 : ", prompt, "\n")
                #print ("답변 : ", response, "\n")
                
                return {"input": prompt,
                        "output": response.choices[0].message.content,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens}
                
                
            #o1이랑 o1 mini 말고 나머지들, 원래 모델들
            else :
                messages = [{"role": "user", "content": prompt}]
                #response = openai.ChatCompletion.create(
                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=OPENAI_CONFIG["temperature"],
                    max_tokens=OPENAI_CONFIG["max_tokens"],
                    top_p=OPENAI_CONFIG["top_p"],
                    frequency_penalty=OPENAI_CONFIG["frequency_penalty"],
                    presence_penalty=OPENAI_CONFIG["presence_penalty"],
                    stop=stop
                    )
                
                print ("사용모델 : ", self.model_name, "\n")
                #print ("질문 : ", prompt, "\n")
                #print ("답변 : ", response, "\n")

                return {"input": prompt,
                        "output": response.choices[0].message.content,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens}

                '''
                return {"input": prompt,
                        "output": response["choices"][0]["message"]["content"],
                        "prompt_tokens": response["usage"]["prompt_tokens"],
                        "completion_tokens": response["usage"]["completion_tokens"]}
                '''
        elif self.model_name in LLAMA_WEIGHTS:
            
            instruction, input = prompt[0], prompt[1]
            output, prompt = self.al.lora_generate(instruction, input)
            return {"input": prompt,
                    "output": output,
                    "prompt_tokens": len(prompt)/4,
                    "completion_tokens": len(output)/4
            }

        else:
            
            raise ValueError("Model not supported")
