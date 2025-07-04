"""
GENERATIVE TEXT MODEL
Description:
This script uses a pre-trained GPT-2 language model from Hugging Face Transformers
to generate paragraphs of AI-generated text based on an input prompt.
It supports generating multiple paragraphs, and optionally saving the results
to a timestamped text file. The script is designed for simple usage via command line.
"""

import os
import torch
import textwrap
import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def format_text_block(text: str, width: int = 100) -> str:
    """
    Formats a long text string into readable blocks using word wrapping.
    
    Parameters:
        text (str): The input string to format.
        width (int): Maximum width of each line in characters.

    Returns:
        str: Formatted multi-line string.
    """
    return '\n'.join(textwrap.wrap(text, width))


def write_to_file(text_list, prompt, directory="outputs"):
    """
    Writes the generated text output to a file in a specified directory.

    Parameters:
        text_list (List[str]): List of paragraphs to write.
        prompt (str): The original input prompt for reference.
        directory (str): Folder to save the output file in.
    """
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(directory, f"generated_text_{timestamp}.txt")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Prompt: {prompt}\n\n")
        for i, paragraph in enumerate(text_list, 1):
            f.write(f"Paragraph {i}:\n{paragraph}\n\n")

    print(f"\nOutput saved to: {filename}")


def generate_text(prompt, model, tokenizer, device, num_paragraphs=1, max_length=200):
    """
    Generates AI-written paragraphs from a given prompt using GPT-2.

    Parameters:
        prompt (str): The user input to begin generation.
        model: Pre-trained GPT-2 model.
        tokenizer: GPT-2 tokenizer.
        device: Device to run the model on (CPU/GPU).
        num_paragraphs (int): Number of paragraphs to generate.
        max_length (int): Maximum length (in tokens) of generated text.

    Returns:
        List[str]: List of formatted generated paragraphs.
    """
    # Encode input and get attention mask
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,  # ✅ Added to avoid EOS/pad conflict
        max_length=max_length,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        num_return_sequences=num_paragraphs,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )

    results = []
    for output in outputs:
        decoded_text = tokenizer.decode(output, skip_special_tokens=True)
        formatted_text = format_text_block(decoded_text)
        results.append(formatted_text)

    return results


def main():
    """
    Main driver function for executing the generative text model workflow.
    Loads the GPT-2 model and tokenizer, handles user input, triggers generation,
    displays results, and optionally saves them to disk.
    """
    print("=" * 70)
    print("GENERATIVE TEXT MODEL".center(70))
    print("=" * 70)

    # Load GPT-2 model and tokenizer
    model_name = "gpt2"
    print("Loading language model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model is using device: {device}\n")

    # Get user input
    prompt = input("Enter a topic or starting sentence: ").strip()
    if not prompt:
        prompt = "Artificial intelligence is transforming the world"

    try:
        num_paragraphs = int(input("How many paragraphs to generate? [Default = 1]: ") or 1)
    except ValueError:
        print("Invalid input. Using default: 1 paragraph.")
        num_paragraphs = 1

    print("\nGenerating text. Please wait...\n")
    paragraphs = generate_text(prompt, model, tokenizer, device, num_paragraphs)

    # Display results
    for idx, paragraph in enumerate(paragraphs, 1):
        print(f"\nParagraph {idx}:\n{'-' * 80}")
        print(paragraph)
        print('-' * 80)

    # Save output option
    save = input("\nDo you want to save the output to a file? (y/n): ").strip().lower()
    if save == 'y':
        write_to_file(paragraphs, prompt)

    print("\nText generation process completed.")


if __name__ == "__main__":
    main()
