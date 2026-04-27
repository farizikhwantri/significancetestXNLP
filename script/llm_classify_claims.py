import pandas as pd
import json
import torch
from transformers import AutoTokenizer, pipeline
import time
from typing import Dict
import os
import bibtexparser

class ClaimClassifier:
    def __init__(self, model_id: str = "HuggingFaceH4/zephyr-7b-beta", device: str = "cuda"):
        """
        Initialize the claim classifier with a Hugging Face model.
        
        Args:
            model_id: Model identifier from Hugging Face Hub
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_id = model_id
        self.device = device
        self.bibtex_data = {}  # Store parsed bibtex data
        
        print(f"Loading tokenizer for {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        print(f"Loading model {model_id} with 4-bit quantization...")
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "load_in_4bit": True
            },
            device_map="auto",
            tokenizer=self.tokenizer
        )
        
        self.claim_types = {
            "1": "Illustrative Insight",
            "2": "Models Comparison",
            "3": "Robustness",
            "4": "Bias Detection",
            "5": "Human Decision Support",
            "6": "Deployment Readiness"
        }
    
    def load_bibtex(self, bibtex_path: str) -> None:
        """
        Load and parse BibTeX file to extract abstracts.
        
        Args:
            bibtex_path: Path to BibTeX file
        """
        print(f"Loading BibTeX file from {bibtex_path}...")
        try:
            with open(bibtex_path, 'r', encoding='utf-8') as f:
                bibtex_str = f.read()
            
            library = bibtexparser.parse_string(bibtex_str)
            
            for entry in library.entries:
                entry_key = entry.key
                abstract = entry.fields_dict.get('abstract', '')
                
                if abstract:
                    # Extract abstract value (it may have quotes or braces)
                    abstract_value = abstract.value if hasattr(abstract, 'value') else str(abstract)
                    self.bibtex_data[entry_key] = abstract_value.strip('{}\"')
            
            print(f"✓ Loaded {len(self.bibtex_data)} abstracts from BibTeX file\n")
        except Exception as e:
            print(f"Error loading BibTeX file: {e}")
            self.bibtex_data = {}
    
    def get_abstract(self, paper_id: str, row: pd.Series = None, abstract_column: str = "abstract") -> str:
        """
        Get abstract from multiple sources with priority order.
        
        Args:
            paper_id: Paper ID to lookup
            row: DataFrame row (optional, for CSV data)
            abstract_column: Column name in CSV containing abstracts
            
        Returns:
            Abstract text or empty string
        """
        # Priority 1: BibTeX data
        if paper_id in self.bibtex_data:
            return self.bibtex_data[paper_id]
        
        # Priority 2: CSV data
        if row is not None and abstract_column in row:
            abstract = row.get(abstract_column, "")
            if abstract and isinstance(abstract, str) and len(abstract.strip()) > 0:
                return abstract
        
        # Priority 3: Use title as fallback
        if row is not None and 'title' in row:
            title = row.get('title', "")
            if title and isinstance(title, str) and len(title.strip()) > 0:
                return title
        
        return ""
    
    def load_prompt_template(self, prompt_path: str) -> str:
        """Load the claim classification prompt from file."""
        with open(prompt_path, 'r') as f:
            return f.read()
    
    def format_prompt(self, abstract: str, prompt_template: str) -> str:
        """Format the prompt with the abstract."""
        return prompt_template.replace("{{ABSTRACT}}", abstract)
    
    def classify_paper(self, abstract: str, prompt_template: str, max_tokens: int = 150) -> Dict:
        """
        Classify a single paper abstract.
        
        Args:
            abstract: Paper abstract text
            prompt_template: Prompt template with taxonomy
            max_tokens: Max tokens to generate
            
        Returns:
            Dictionary with classification results
        """
        try:
            # Format prompt
            prompt = self.format_prompt(abstract, prompt_template)
            
            # Generate classification
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ] if "<|eot_id|>" in self.tokenizer.vocab else [self.tokenizer.eos_token_id]
            
            outputs = self.pipe(
                prompt,
                max_new_tokens=max_tokens,
                eos_token_id=terminators,
                do_sample=False,
                temperature=0.1,
            )
            
            raw_response = outputs[0]["generated_text"]
            
            # Extract the response after the prompt
            response_text = raw_response.split("assistant")[-1].strip() if "assistant" in raw_response else raw_response
            
            # Parse the response
            parsed = self.parse_response(response_text)
            
            return {
                "status": "success",
                "raw_response": response_text,
                "labels": parsed["labels"],
                "justification": parsed["justification"],
                "model": self.model_id.split("/")[-1]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "labels": [],
                "justification": "",
                "model": self.model_id.split("/")[-1]
            }
    
    def parse_response(self, response_text: str) -> Dict:
        """
        Parse the LLM response to extract labels and justification.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Dictionary with parsed labels and justification
        """
        labels = []
        justification = ""
        
        try:
            lines = response_text.split('\n')
            
            for i, line in enumerate(lines):
                line_lower = line.lower()
                
                # Extract labels
                if 'label' in line_lower and ':' in line:
                    label_part = line.split(':', 1)[1].strip()
                    label_part = label_part.strip('[]')
                    extracted_labels = [l.strip() for l in label_part.split(',') if l.strip()]
                    
                    for label in extracted_labels:
                        if label in self.claim_types:
                            labels.append(self.claim_types[label])
                        elif label in self.claim_types.values():
                            labels.append(label)
                        else:
                            for claim_type in self.claim_types.values():
                                if claim_type.lower() in label.lower():
                                    labels.append(claim_type)
                                    break
                
                # Extract justification
                if 'justif' in line_lower and ':' in line:
                    justification = line.split(':', 1)[1].strip()
                    if i + 1 < len(lines) and lines[i + 1].strip() and not any(
                        keyword in lines[i + 1].lower() for keyword in ['label', 'justif']
                    ):
                        justification += " " + lines[i + 1].strip()
            
            if not labels:
                for num, claim_type in self.claim_types.items():
                    if claim_type.lower() in response_text.lower():
                        labels.append(claim_type)
            
            if not justification:
                justification = response_text[:100] + "..." if len(response_text) > 100 else response_text
            
            return {
                "labels": list(set(labels)),
                "justification": justification
            }
            
        except Exception as e:
            return {
                "labels": [],
                "justification": f"Error parsing: {str(e)}"
            }
    
    def classify_batch(
        self,
        df: pd.DataFrame,
        prompt_template: str,
        abstract_column: str = "abstract",
        id_column: str = "id",
        batch_size: int = 5,
        delay: float = 0.5
    ) -> pd.DataFrame:
        """
        Classify multiple papers from a DataFrame.
        
        Args:
            df: DataFrame with paper data
            prompt_template: Prompt template string
            abstract_column: Name of column containing abstracts
            id_column: Name of column containing paper IDs
            batch_size: Number of papers before batch save
            delay: Delay between classifications (seconds)
            
        Returns:
            DataFrame with classification results
        """
        results = []
        total = len(df)
        
        for idx, row in df.iterrows():
            paper_id = row.get(id_column, f"paper_{idx}")
            
            # Get abstract from BibTeX or CSV
            abstract = self.get_abstract(paper_id, row, abstract_column)
            
            if not abstract or not isinstance(abstract, str):
                print(f"[{idx+1}/{total}] {paper_id}: No abstract, skipping")
                results.append({
                    "id": paper_id,
                    "title": row.get('title', ''),
                    "labels": [],
                    "justification": "No abstract provided",
                    "status": "skipped"
                })
                continue
            
            print(f"[{idx+1}/{total}] Classifying {paper_id}...", end=" ", flush=True)
            
            # Classify
            result = self.classify_paper(abstract, prompt_template)
            
            # Create result row
            result_row = {
                "id": paper_id,
                "title": row.get('title', ''),
                "status": result["status"],
                "labels": " | ".join(result["labels"]) if result["labels"] else "None",
                "primary_label": result["labels"][0] if result["labels"] else "",
                "justification": result["justification"][:200]
            }
            
            results.append(result_row)
            
            # Print status
            if result["status"] == "success":
                label_str = ", ".join(result["labels"][:2]) if result["labels"] else "No labels"
                print(f"✓ {label_str}")
            else:
                print(f"✗ Error: {result.get('error', 'Unknown')}")
            
            # Rate limiting
            time.sleep(delay)
        
        results_df = pd.DataFrame(results)
        return results_df


def main():
    """Main function to run claim classification."""
    
    # Configuration
    csv_path = "/Users/fariz/repositories/significancetestXNLP/output/filtered_main_conf_no_test.csv"
    bibtex_path = "/Users/fariz/repositories/significancetestXNLP/output/filtered_bibtex_since2020.bib"
    prompt_path = "/Users/fariz/repositories/significancetestXNLP/config/claim_prompt.txt"
    output_path = "/Users/fariz/repositories/significancetestXNLP/output/classified_claims_hf.csv"
    
    # model_id = "HuggingFaceH4/zephyr-7b-beta"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    # Load data
    print(f"Loading papers from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} papers\n")
    
    # Initialize classifier
    classifier = ClaimClassifier(model_id=model_id)
    
    # Load BibTeX file
    if os.path.exists(bibtex_path):
        classifier.load_bibtex(bibtex_path)
    else:
        print(f"⚠ BibTeX file not found at {bibtex_path}")
    
    # Load prompt template
    print(f"Loading prompt template from {prompt_path}...")
    prompt_template = classifier.load_prompt_template(prompt_path)
    print(f"Prompt loaded\n")
    
    # Classify papers
    print("Starting classification...\n")
    results_df = classifier.classify_batch(
        df,
        prompt_template,
        abstract_column="abstract",
        id_column="id",
        batch_size=10,
        delay=1.0
    )
    
    # Save results
    print(f"\nSaving results to {output_path}...")
    results_df.to_csv(output_path, index=False)
    print(f"✓ Results saved!")
    
    # Print summary
    print("\n" + "="*60)
    print("CLASSIFICATION SUMMARY")
    print("="*60)
    print(f"Total papers processed: {len(results_df)}")
    print(f"Successful: {len(results_df[results_df['status'] == 'success'])}")
    print(f"Failed: {len(results_df[results_df['status'] == 'error'])}")
    print(f"Skipped: {len(results_df[results_df['status'] == 'skipped'])}")
    
    print("\nTop Primary Labels:")
    label_counts = results_df['primary_label'].value_counts()
    for label, count in label_counts.head(10).items():
        if label:
            print(f"  {label}: {count}")
    
    print("\nSample Results:")
    print(results_df[['id', 'primary_label', 'justification']].head(10).to_string())
    
    return results_df


if __name__ == "__main__":
    results_df = main()