import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import pickle

#Load stored variables.
with open('myTokenizer.pkl', 'rb') as file:
    words_unique = pickle.load(file)
    word_to_int = pickle.load(file)
    int_to_word = pickle.load(file)
    n_vocab = pickle.load(file)

def encode_promt(prompt):
  prompt = prompt.split(' ')
  prompt_processed = []
  for word in prompt:
    if word in words_unique:
      prompt_processed.append(word)
    else:
      for c in word:
        if c in words_unique:
          prompt_processed.append(c)
    prompt_processed.append(' ')
  prompt_processed = [word for word in prompt_processed if word != '']
  return prompt_processed

def predict(prompt):
    prompt = encode_promt(prompt)
    sequence = [word_to_int[word] for word in prompt]
    answer = ''
    with torch.no_grad():
        for i in range(len(sequence)*3):
            x = np.reshape(sequence, (1, len(sequence), 1)) / float(n_vocab) # Reshape and normalize
            x = torch.tensor(x, dtype=torch.float32).to(device)
            prediction = model(x)
            index = int(prediction.argmax()) # Predict an array of n_vocab integers
            answer += ' ' + int_to_word[index]
            sequence.append(index) # Append the predicted integer into the current sequence
            sequence = sequence[1:] # Remove the first integer from the sequence
    return ' '.join([int_to_word[i] for i in sequence]), answer # Convert all the integers into characters


# Load model trained.
model = torch.jit.load('myModel.pt', map_location=torch.device('cpu'))
model.eval()

predict('cảo thơm lần dở trước đèn')[1]