import tiktoken
import os

# Function to read the user prompt from a file
def read_text(prompt_file='prompt.txt'):
    """
    Reads text from the specified file and returns it.
    
    :param prompt_file: The name of the file containing the prompt text (default is 'prompt.txt')
    :return: The text content of the file, stripped of leading/trailing whitespace.
    """
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"The file '{prompt_file}' does not exist.")
    
    with open(prompt_file, "r", encoding="utf-8") as file:
        return file.read().strip()

def num_tokens_from_text(text: str, encoding_name: str = "cl100k_base", model_name: str = None) -> int:
    """
    Returns the number of tokens in a text string based on the encoding used by the model or a specific encoding name.
    
    :param text: The text to tokenize.
    :param encoding_name: The default encoding to use if no model is provided (default is "cl100k_base").
    :param model_name: The optional model name to fetch the appropriate encoding (e.g., "gpt-3.5-turbo").
    :return: The number of tokens in the text.
    """
    # Determine the encoding to use based on whether a model name is provided
    if model_name:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except ValueError:
            raise ValueError(f"Model '{model_name}' is not recognized or supported for tokenization.")
    else:
        encoding = tiktoken.get_encoding(encoding_name)

    # Tokenize the text and return the number of tokens
    num_tokens = len(encoding.encode(text))
    return num_tokens

def tokenize_sentence(sentence: str, encoding_name: str = "cl100k_base", model_name: str = None) -> list:
    """
    Tokenizes a sentence into a list of tokens using the specified encoding or model.
    
    :param sentence: The sentence to tokenize.
    :param encoding_name: The default encoding to use if no model is provided (default is "cl100k_base").
    :param model_name: The optional model name to fetch the appropriate encoding (e.g., "gpt-3.5-turbo").
    :return: A list of tokens representing the sentence.
    """
    # Determine the encoding to use based on whether a model name is provided
    if model_name:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except ValueError:
            raise ValueError(f"Model '{model_name}' is not recognized or supported for tokenization.")
    else:
        encoding = tiktoken.get_encoding(encoding_name)

    # Tokenize the sentence
    token_ids = encoding.encode(sentence)
    
    # Create a dictionary of token_id to token (subword)
    token_dict = {token_id: encoding.decode([token_id]) for token_id in token_ids}
    return token_dict


if __name__ == "__main__":
    try:
        # Read the text from the file
        # text = read_text(prompt_file="prompt_jd.txt")
        
        # Get the token count for the text
        # token_count = num_tokens_from_text(text)
        
        # Print the total number of tokens
        # print(f"Total Number of Tokens: {token_count}")

        # Tokenize a sentence and print the tokens
        sentence = "Unhappiness in post-modern art often stems from excessive self-criticism."
        tokens_dict = tokenize_sentence(sentence)
        print(f"Tokens for the sentence '{sentence}\n': {tokens_dict}")
    
    except Exception as e:
        print(f"Error: {e}")
