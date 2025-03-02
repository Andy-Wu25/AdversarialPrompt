import os
import re
import json
import logging
from typing import List, Dict
import numpy as np
from utils.config import Config
from utils.helpers import load_json, save_file, set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/cross_context_prompt_injector.log"),
        logging.StreamHandler()
    ]
)

# Set seed for reproducibility
set_seed()

class CrossContextPromptInjector:
    def __init__(self, config: Config):
        self.config = config
        self.prompt_code_pairs, self.cwe_list = self.load_data()
        logging.info("CrossContextPromptInjector initialized.")

    def load_data(self) -> List[Dict]:
        data_path = self.config.data_path
        cwe_path = self.config.list_of_cwes_path

        logging.info(f"Loading data from {data_path}")
        logging.info(f"Loading CWE list from {cwe_path}")
        try:
            data = load_json(data_path)
            cwe_list = load_json(cwe_path)
            return data, cwe_list
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise

    def generate_injection_prompts(self):
        n_prompts = self.config.number_of_injection_prompts
        for lang in ['c', 'py']:
            for cwe in self.cwe_list[lang]:
                logging.info(f"Generating injection prompts for CWE: {cwe}")
                self.create_cross_context_prompts(lang, cwe, n_prompts)

    def create_cross_context_prompts(self, lang: str, cwe: str, n_prompts: int):
        len_data_per_cwe = len(self.prompt_code_pairs[lang][cwe]['codes'])
        for i in range(n_prompts):
            prompt_content = ""
            prompt_content += f"# Cross-context injection for {cwe}\n"
            prompt_content += self.prompt_code_pairs[lang][cwe]['codes'][i % len_data_per_cwe]

            # Generate multi-context scenarios
            prompt_content += "\n# Simulate cross-context vulnerability\n"
            prompt_content += "# Example: User input in SQL, then executed as Python code\n"
            prompt_content += "user_input = 'malicious'\n"
            prompt_content += "query = f'SELECT * FROM users WHERE name = "{user_input}"'\n"
            prompt_content += "exec(query)\n"

            # Add more diverse contexts
            prompt_content += "\n# Potential HTML Injection\n"
            prompt_content += "html_content = f'<div>{user_input}</div>'\n"
            prompt_content += "render(html_content)\n"

            prompt_file_name = os.path.join(self.config.injection_prompts_path, f"injection_{cwe}_{i}.{lang}")
            if not os.path.exists(os.path.dirname(prompt_file_name)):
                os.makedirs(os.path.dirname(prompt_file_name))
            with open(prompt_file_name, 'w') as f:
                f.write(prompt_content)
            logging.info(f"Created Cross-Context Injection prompt file: {prompt_file_name}")


def main():
    config = Config()
    os.makedirs(config.injection_prompts_path, exist_ok=True)
    injector = CrossContextPromptInjector(config)
    injector.generate_injection_prompts()

if __name__ == "__main__":
    main()
