from prompts import fewshots

OPENAI_COMPLETION_MODELS = ["gpt-3.5-turbo-instruct"] #["gpt-3.5-turbo"] #text-davinci-003
OPENAI_CHAT_MODELS = ["gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", "gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini"]
LLAMA_WEIGHTS = ["tloen/alpaca-lora-7b", "rewoo/planner_7B"]

DEFAULT_EXEMPLARS_COT = {"hotpot_qa": fewshots.HOTPOTQA_COT,
                         "trivia_qa": fewshots.TRIVIAQA_COT,
                         "gsm8k": fewshots.GSM8K_COT,
                         "physics_question": fewshots.TRIVIAQA_COT,
                         "sports_understanding": fewshots.TRIVIAQA_COT,
                         "strategy_qa": fewshots.TRIVIAQA_COT,
                         "sotu_qa": fewshots.TRIVIAQA_COT}

DEFAULT_EXEMPLARS_REACT = {"hotpot_qa": fewshots.HOTPOTQA_REACT,
                           "trivia_qa": fewshots.TRIVIAQA_REACT,
                           "gsm8k": fewshots.GSM8K_REACT,
                           "physics_question": fewshots.GSM8K_REACT,
                           "sports_understanding": fewshots.GSM8K_REACT,
                           "strategy_qa": fewshots.GSM8K_REACT,
                           "sotu_qa": fewshots.GSM8K_REACT}

DEFAUL_EXEMPLARS_PWS = {"hotpot_qa": fewshots.HOTPOTQA_PWS_BASE,
                        "trivia_qa": fewshots.TRIVIAQA_PWS,
                        "gsm8k": fewshots.GSM8K_PWS,
                        "physics_question": fewshots.GSM8K_PWS,
                        "sports_understanding": fewshots.SPORTS_UNDERSTANDING,
                        "strategy_qa": fewshots.STRATEGY_PWS,
                        "sotu_qa": fewshots.HOTPOTQA_PWS_BASE}


def get_token_unit_price(model):
    if model in OPENAI_COMPLETION_MODELS:
        return 0.00002
    elif model in OPENAI_CHAT_MODELS:
        if model == "gpt-3.5-turbo-0125": #US$0.50 / 1M input tokens, US$1.50 / 1M output tokens
            return 0.0000005
        if model == "gpt-3.5-turbo-1106": #US$1 / 1M input tokens, US$2 / 1M output tokens
            return 0.000001
        elif model == "gpt-3.5-turbo-instruct": #US$1.5 / 1M input tokens, US$2 / 1M output token
            return 0.0000015
        elif model == "gpt-4o": #US$5.00 / 1M input tokens, US$15.00 / 1M output tokens
            return 0.000005
        elif model == "gpt-4o-mini": #US$0.150 / 1M input tokens, US$0.600 / 1M output token
            return 0.00000015
        elif model == "o1-preview": #US$15 / 1M input tokens, US$60 / 1M output token
            return 0.000015
        elif model == "o1-mini": #US$3 / 1M input tokens, US$12 / 1M output token
            return 0.000003

    elif model in LLAMA_WEIGHTS:
        return 0.0
    else:
        raise ValueError("Model not found")
