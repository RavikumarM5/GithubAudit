#!/usr/bin/env python
# coding: utf-8

# In[17]:


import tkinter as tk
import requests
import re 
import torch
import warnings
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from urllib.parse import urlparse


# In[ ]:


def fetch_github_repositories():
    warnings.filterwarnings("ignore")

    global most_complex_repo, justification_text
    
    user_url = url_entry.get()

    # here we are extracting the github username from the user url
    parsed_url = urlparse(user_url)
    username = parsed_url.path.strip("/")
    # we need to check if the given username is not empty
    if not username:
        print("Invalid Github user url.")
        return

    # api endpoint to fetch user repositories
    api_url = f"https://api.github.com/users/{username}/repos"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        repositories = response.json()

        # extracting the repository names
        repository_names = [repo['name'] for repo in repositories]

        # initializing tokenizer and its model
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")

        # preprocess repository names
        def preprocess_repository_names(repository_names):
            for repo in repository_names:
                repo = repo.strip().lower()
                if repo:
                    yield repo

        # process a repository
        def process_repository(repository_name):
            # generating a prompt on the repository name
            prompt = f"Evaluate the technical complexity of the repository: {repository_name}. Analyze the code and provide insights."

            # tokenize the prompt
            input_ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=100, return_tensors="pt")
            # generate attention mask
            attention_mask = torch.ones_like(input_ids)

            # generate output using the model
            with torch.no_grad():
                output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=200, pad_token_id=0)

            # decode the output
            processed_repo = tokenizer.decode(output[0], skip_special_tokens=True)
            
            return processed_repo

        preprocessed_names_generator = preprocess_repository_names(repository_names)

        # process and score the preprocessed names
        repository_scores = {}
        for preprocessed_name in preprocessed_names_generator:
            processed_repo = process_repository(preprocessed_name)

            # complexity score is based on the length of the processed repository
            complexity_score = len(processed_repo)
            repository_scores[preprocessed_name] = complexity_score

        # identify the repository with the highest complexity score
        most_complex_repo = max(repository_scores, key=repository_scores.get)

        # justify the selection using GPT
        justification_prompt = f"Justification for selecting the most technically complex repository: {most_complex_repo}."
        justification_input_ids = tokenizer.encode(justification_prompt, add_special_tokens=False, truncation=True, max_length=160)

        attention_mask = torch.ones_like(justification_input_ids)

        with torch.no_grad():
            justification_output = model.generate(input_ids=justification_input_ids, attention_mask=attention_mask, max_length=200, pad_token_id=token_id)

        justification_text = tokenizer.decode(justification_output[0], skip_special_tokens=True)

        # update the GUI labels with the values
        most_complex_repo_var.set(most_complex_repo)
        justification_text_var.set(justification_text)
    except requests.exceptions.HTTPError as err:
        print(f"An HTTP error occurred: {err}")
    except requests.exceptions.RequestException as err:
        print(f"An error occurred: {err}")

# GUI
window = tk.Tk()
window.title("GitHub Repository Analyzer")

# url entry
url_label = tk.Label(window, text="Github User URL:")
url_label.pack()
url_entry = tk.Entry(window, width=50)
url_entry.pack()

#Fetch button
fetch_button = tk.Button(window, text="Fetch Repositories", command=fetch_github_repositories)
fetch_button.pack()

#displaying repository name
repo_label = tk.Label(window, text="Most complex repository:")
repo_label.pack()

most_complex_repo_var = tk.StringVar()

# Create the Label widget using the variable
repo_name_label = tk.Label(window, textvariable=most_complex_repo_var)
repo_name_label.pack()

#Display justification
justification_label = tk.Label(window, text="Justification:")

# Use pack() to display the justification label
justification_label.pack()

justification_text_var = tk.StringVar()
justification_text_label = tk.Label(window, textvariable=justification_text_var, wraplength=400)

#it will handle the user interactions
window.mainloop()
 


# In[ ]:





# In[ ]:




