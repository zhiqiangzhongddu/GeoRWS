def llm_tag(cfg) -> str:
    name = cfg.llm.name.replace('/', '_')
    return 'fake-' + name if cfg.llm.provider == 'fake' else name
