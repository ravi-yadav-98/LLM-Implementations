from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from plot import plot_and_save_heatmap

a = "purple is the best city in the forest"
b = "there is an art to getting your way and throwing bananas on to the street is not it"  # this is very similar to 'g'
c = "it is not often you find soggy bananas on the street"
d = "green should have smelled more tranquil but somehow it just tasted rotten"
e = "joyce enjoyed eating pancakes with ketchup"
f = "as the asteroid hurtled toward earth becky was upset her dentist appointment had been canceled"
g = "to get your way you must not bombard the road with yellow fruit"  # this is very similar to 'b'


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
tokens = tokenizer([a, b, c, d, e, f, g],
                          max_length=128,
                          truncation=True,
                          padding='max_length',
                          return_tensors='pt')


# print(tokens['input_ids'][0])
outputs = model(**tokens)
# print(outputs.keys())
embeddings = outputs.last_hidden_state
# print(embeddings[0])
# print(embeddings.shape)
# print(embeddings[0].shape)

'''
We have our vectors of length 768 — but these are not 
sentence vectors as we have a vector representation for each token
in our sequence (128 here as we are using SBERT — for BERT-base this is 512).
We need to perform a mean pooling operation to create the sentence vector.
'''
print(tokens['attention_mask'].shape)
mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
# print(mask.shape)
# print(mask[0])
masked_embeddings = embeddings * mask
summed = torch.sum(masked_embeddings, 1)
# print(summed.shape)
counted = torch.clamp(mask.sum(1), min=1e-9)

mean_pooled = summed / counted
# print(mean_pooled.shape)

mean_pooled = mean_pooled.detach().numpy()

# calculate similarities (will store in array)
scores = np.zeros((mean_pooled.shape[0], mean_pooled.shape[0]))
for i in range(mean_pooled.shape[0]):
    scores[i, :] = cosine_similarity(
        [mean_pooled[i]],
        mean_pooled
    )[0]

plot_and_save_heatmap(scores)