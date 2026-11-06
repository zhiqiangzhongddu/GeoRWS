from yacs.config import CfgNode as CN


def set_llm_cfg(cfg):
    cfg.llm = CN()  # Configuration block
    cfg.llm.provider = 'openai'  # openai | fake
    cfg.llm.name = 'gpt-4o'  # Model name
    cfg.llm.base_url = ''  # API endpoint
    cfg.llm.api_key_env = 'OPENAI_API_KEY'  # API key environment variable
    cfg.llm.temperature = 1.0  # Sampling temperature
    cfg.llm.top_p = 1.0  # Sampling top-p
    cfg.llm.max_tokens = 256  # Maximum tokens
    cfg.llm.max_workers = 8  # Concurrent workers
    cfg.llm.max_retries = 6  # Maximum retries
