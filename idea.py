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
        logging.FileHandler("logs/few_shot_prompt_generator.log"),
        logging.StreamHandler()
    ]
)

# Set seed for reproducibility
set_seed()

class FewShotPromptGenerator:
    def __init__(self, config: Config):
        """
        Initialize the FewShotPromptGenerator with configuration settings.
        
        Args:
            config (Config): Configuration object containing settings and parameters.
        """
        self.config = config
        self.prompt_code_pairs, self.cwe_list = self.load_data()
        logging.info("FewShotPromptGenerator initialized.")

    def load_data(self) -> List[Dict]:
        """
        Load the prompt_code_pairs.json dataset.
        
        Returns:
            List[Dict]: A list of prompt-code pair dictionaries.
        """
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
    def get_random_indices(self, len_data_per_cwe: int, last_indices: List[int], n_examples_per_cwe: int, n_fs_prompts: int) -> List[List[int]]:
        """
        Get random indices for few-shot prompts.

        Args:
            len_data_per_cwe (int): Length of data per CWE.
            last_indices (List[int]): List of indices to be used as last examples.
            n_examples_per_cwe (int): Number of examples per prompt.
            n_fs_prompts (int): Number of few-shot prompts to generate.

        Returns:
            List[List[int]]: List of random indices for each prompt.
        """
        all_indices = []
        while len(all_indices) < n_fs_prompts:
            indices = np.random.choice(len_data_per_cwe, len_data_per_cwe, replace=False)
            
            last_index = last_indices[len(all_indices) % len(last_indices)]
            index_of_current_last = np.where(indices == last_index)[0][0]
            if index_of_current_last != len(indices) - 1:
                indices[index_of_current_last], indices[-1] = indices[-1], indices[index_of_current_last]

            if all(not np.array_equal(np.asarray(indices), np.asarray(ids)) for ids in all_indices):
                all_indices.append(indices[:n_examples_per_cwe])

        return all_indices
    
    
    def generate_prompts(self):
        """
        Generate few-shot prompts using three different methods.
        """
        n_prompts = self.config.number_of_fs_prompts
        n_examples_per_cwe = self.config.number_of_examples_per_cwe
        for lang in ['c','py']:
            for cwe in self.cwe_list[lang]:
                last_indices = []
                for i in range(n_examples_per_cwe):
                    if self.prompt_code_pairs[lang][cwe]['file_names'][i].find('CWE') == -1:
                        last_indices.append(i)
                    else:
                        continue
                logging.info(f"Generating prompts for CWE: {cwe}")
                self.fs_codes_generate(lang, cwe, n_prompts, n_examples_per_cwe, last_indices)
                # self.fs_prompts_generate(cwe, n_prompts, n_examples_per_cwe)
                # self.os_prompt_generate(cwe)

    def fs_codes_generate(self, lang: str, cwe: str, n_prompts: int, n_examples_per_cwe: int, last_indices: List[int]):
        """
        Generate few-shot prompts for FS-Codes method for a given CWE.
        """
        len_data_per_cwe = len(self.prompt_code_pairs[lang][cwe]['codes'])
        indices_list = self.get_random_indices(len_data_per_cwe, last_indices, n_examples_per_cwe, n_prompts)

        for i, indices in enumerate(indices_list):
            prompt_content = ""
            for j, idx in enumerate(indices):
                prompt_content += 'first:\n'
                prompt_content += self.prompt_code_pairs[lang][cwe]['codes'][idx]
                prompt_content += '\nsecond:\n'
                if j == n_examples_per_cwe - 1:
                    # For the last example, include only the libraries
                    match_import = re.findall('import.*|from.*import.*|#include .*', self.prompt_code_pairs[lang][cwe]['prompts'][idx])
                    prompt_content += '\n'.join(match_import)
                else:
                    prompt_content += self.prompt_code_pairs[lang][cwe]['prompts'][idx]
                    prompt_content += '\n###\n'

            # Save the prompt to a file
            prompt_file_name = os.path.join(self.config.few_shot_prompts_path, "fs_codes",lang,cwe,f"fs-codes_{cwe}_{i}.{lang}")
            if not os.path.exists(os.path.dirname(prompt_file_name)):
                os.makedirs(os.path.dirname(prompt_file_name))
            with open(prompt_file_name, 'w') as f:
                f.write(prompt_content)
            logging.info(f"Created FS-Codes prompt file: {prompt_file_name}")


    def generate_cross_context_prompts(self):
        """
        New function to generate "FileA" + "FileB" pairs that demonstrate
        cross-context prompt injection. The idea is:
         - FileA has some partial code that might do something suspicious or
           include an explicit instruction to disable security checks.
         - FileB references a function from FileA. We see if the model carries over the
           vulnerability or instruction from FileA into FileB.
        """
        n_prompts = self.config.number_of_fs_prompts
        n_examples_per_cwe = self.config.number_of_examples_per_cwe

        for lang in ['c','py']:
            if lang not in self.cwe_list:
                continue
            for cwe in self.cwe_list[lang]:
                logging.info(f"Generating cross-context prompts for CWE: {cwe}")
                self.cc_generate_pair(lang, cwe, n_prompts, n_examples_per_cwe)

    def cc_generate_pair(self, lang: str, cwe: str, n_prompts: int, n_examples_per_cwe: int):
        """
        Construct pairs of (FileA, FileB) for cross-context injection.
        We'll keep it simple: each pair is structured as a prompt that
        shows how fileA might define an insecure function, then fileB
        calls it.
        """
        # This is just an illustration; you might do something more advanced.
        # We'll do something like the standard fs-codes approach, but produce TWO files.
        len_data_per_cwe = len(self.prompt_code_pairs[lang][cwe]['codes'])
        indices_list = self.get_random_indices(
            len_data_per_cwe, list(range(n_examples_per_cwe)), n_examples_per_cwe, n_prompts
        )

        for i, indices in enumerate(indices_list):
            # Build file A content:
            file_a_content = []
            file_b_content = []
            for j, idx in enumerate(indices):
                # Suppose "codes[idx]" is the "vulnerable part"
                # and "prompts[idx]" is the partial prompt from your dataset
                # We'll do a minimal example: file A has a docstring telling it to skip auth.
                file_a_content.append(f"# Cross-context injection example\n"
                                      f"{self.prompt_code_pairs[lang][cwe]['codes'][idx]}")
                file_b_content.append(f"# This file uses the function from FileA.\n"
                                      f"{self.prompt_code_pairs[lang][cwe]['prompts'][idx]}")

            # Join them. In reality, you'd likely want to produce a single
            # multi-file structure or store them as separate .py / .c files.
            fileA_combined = "\n\n### CrossContext ###\n\n".join(file_a_content)
            fileB_combined = "\n\n### CrossContext ###\n\n".join(file_b_content)

            # Save the prompt pair
            base_dir = os.path.join(self.config.few_shot_prompts_path,
                                    "cross_context", lang, cwe, f"pair_{i}")
            os.makedirs(base_dir, exist_ok=True)

            fileA_name = os.path.join(base_dir, f"fileA_{cwe}_{i}.{lang}")
            fileB_name = os.path.join(base_dir, f"fileB_{cwe}_{i}.{lang}")

            with open(fileA_name, "w") as fa:
                fa.write(fileA_combined)
            with open(fileB_name, "w") as fb:
                fb.write(fileB_combined)

            logging.info(f"Created cross-context pair: {fileA_name}, {fileB_name}")

    def fs_prompts_generate(self) -> str:
        """
        Generate few-shot prompts for FS-Prompts.
        
        Returns:
            str: The generated prompts as a string.
        """
        prompts = ""
        return prompts

    def os_prompt_generate(self) -> str:
        """
        Generate few-shot prompts for OS-Prompt.
        
        Returns:
            str: The generated prompts as a string.
        """
        # TODO: Implement the logic for Method 3
        prompts = ""
        for pair in self.prompt_code_pairs:
            # Example placeholder logic with additional metadata
            prompts += f"// Prompt: {pair['prompt']}\n// Code:\n{pair['code']}\n\n"
        return prompts

def main():
    """
    Main function to execute the few-shot prompt generation process.
    """
    # Load configuration
    config = Config()
    
    # Ensure output directory exists
    os.makedirs(config.few_shot_prompts_path, exist_ok=True)

    # Initialize the generator
    generator = FewShotPromptGenerator(config)

    # Generate prompts
    generator.generate_prompts()

if __name__ == "__main__":
    main()