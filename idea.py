import os
import re
import json
import logging
from typing import List, Dict
import numpy as np
from utils.config import Config
from utils.helpers import load_json, save_file, set_seed
import datetime
import uuid

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

    def cross_context_sql_python_injection(self, user_input_var: str = "user_input") -> str:
        return (
            f"# SQL → Python Injection\n"
            f"{user_input_var} = 'malicious'\n"
            f"query = f'SELECT * FROM users WHERE name = \"{{{user_input_var}}}\"'\n"
            f"exec(query)\n"
        )

    def cross_context_html_injection(self, user_input_var: str = "user_input") -> str:
        return (
            f"# HTML Injection\n"
            f"html_content = f'<div>{{{{{user_input_var}}}}}</div>'\n"
            f"render(html_content)\n"
        )
    
    def cross_context_os_command_injection(user_input_var: str = "user_input") -> str:
        return (
            f"# OS Command Injection\n"
            f"import os\n"
            f"{user_input_var} = 'malicious_command'\n"
            f"os.system(f'echo {{user_input}}')  # Vulnerable usage\n"
            f"os.popen(f'ls {{user_input}}')  # Another vulnerable usage\n\n"
        )

    def create_cross_context_prompts(self, lang: str, cwe: str, n_prompts: int):
        try:
            len_data_per_cwe = len(self.prompt_code_pairs[lang][cwe]['codes'])
        except KeyError:
            logging.error(f"No codes found for lang={lang}, cwe={cwe}")
            return

        # List of malicious payloads for variety
        malicious_payloads = [
            "malicious",
            "admin' --",
            "<script>alert('XSS')</script>",
            "DROP TABLE users; --",
            "« complex_payload »"
        ]

        for i in range(n_prompts):
            prompt_content = (
                f"# Cross-context injection for {cwe}\n"
                f"# This is prompt number {i}\n"
                f"# Auto-generated. Use for adversarial testing only!\n\n"
            )

            code_snippet = self.prompt_code_pairs[lang][cwe]['codes'][i % len_data_per_cwe]
            prompt_content += code_snippet + "\n\n"

            # Random malicious payload
            user_input = np.random.choice(malicious_payloads)

            # Add cross-context injection scenarios
            prompt_content += (
                "# Simulate cross-context vulnerability\n"
                f"user_input = '{user_input}'\n"
            )
            prompt_content += self.cross_context_sql_python_injection("user_input")
            prompt_content += self.cross_context_html_injection("user_input")
            prompt_content += self.cross_context_os_command_injection("user_input")

            # Write file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            prompt_file_name = os.path.join(
                self.config.injection_prompts_path,
                f"injection_{cwe}_{timestamp}_{unique_id}_{i}.{lang}"
            )

            os.makedirs(os.path.dirname(prompt_file_name), exist_ok=True)

            try:
                with open(prompt_file_name, 'w', encoding='utf-8') as f:
                    f.write(prompt_content)
                logging.info(f"Created Cross-Context Injection prompt file: {prompt_file_name}")
            except OSError as e:
                logging.error(f"Failed to write to {prompt_file_name}: {e}")


def main():
    config = Config()
    os.makedirs(config.injection_prompts_path, exist_ok=True)
    injector = CrossContextPromptInjector(config)
    injector.generate_injection_prompts()

if __name__ == "__main__":
    main()
