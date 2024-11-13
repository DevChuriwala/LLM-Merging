# Install dependencies
```
pip3 install -r docs/requirements.txt
python3 setup.py install
pip3 install evaluate peft
```

# Set environment variable for Hugging Face authentication
```
export HF_AUTH_TOKEN="<TOKEN>"
```

# Run one of the LLM merging scripts with different methods
```
python3 llm_merging/main.py -m flant5_avg
python3 llm_merging/main.py -m llama_avg
python3 llm_merging/main.py -m tiny_llama_avg
```