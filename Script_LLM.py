import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# hyp parameters
max_iters = 500
eval_iters = 100
learning_rate = 1e-2
batch_size = 16  ### number of batches for every iter 
block_size = 32  ### length of the context 

embed_n = 64
layer_n = 4
head_n = 4  
dropout = 0.0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
###$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

with open('AlisInWonderland.txt', 'r', encoding = 'UTF-8') as file:
    raw_data = file.read()
    
words = raw_data.split()
chars = sorted(set(raw_data))
vocab_size = len(chars)  

chtoin = {c:i for i,c in enumerate(chars)} ## encoding one letter 
intoch = {i:c for i,c in enumerate(chars)}
encode_seq = lambda i : [chtoin[ch] for ch in i] ### take sequ and return indexes
decode_seq = lambda ch :[''.join(intoch[ind]) for ind in ch] ## take indexs and return "string"

enc_data = torch.tensor(encode_seq(raw_data), dtype = torch.long)
train_data = enc_data[:int(0.8 * len(enc_data))]
val_data = enc_data[int(0.8 * len(enc_data)):]

def get_batch(data_type):
    if data_type == 'train':
        data = train_data
    else:
        data = val_data

    temp = torch.randint(len(data) - block_size, (batch_size,))### will return list of 4 numbers like [5112,156,198,1555]
    x_b = torch.stack([data[i:i+block_size] for i in temp]).to(device)
    y_b = torch.stack([data[i+1:i+block_size+1] for i in temp]).to(device)
    return x_b, y_b

@torch.no_grad() ## tell the pytorch that i don't want to call the backword inside of the next function and that for memory optimziation 
def estimate_loss():
    out = {}
    bm.eval() ## actually it won't affect on any item next it will affect on other parameters but here we don't any use any one
    for split in ['train_data', 'val_data']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = bm(x,y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    bm.train() ## actually it won't affect on any item next it will affect on other parameters but here we don't any use any one
    return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_n, head_size, bias=False)
        self.query = nn.Linear(embed_n, head_size, bias=False)
        self.value = nn.Linear(embed_n, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out



class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_n, embed_n)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x




class bigram(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, embed_n)
        self.position_embedding_table = nn.Embedding(block_size, embed_n)
        self.blocks = nn.Sequential(*[Block(embed_n, n_head=head_n) for _ in range(layer_n)])
        self.ln_f = nn.LayerNorm(embed_n) # final layer norm
        self.lm_head = nn.Linear(embed_n, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
        
bm = bigram().to(device)                        

optimizer = torch.optim.AdamW(bm.parameters(), lr = 0.0001)
for steps in range(5000):
    if steps % eval_iters == 0:
        losses = estimate_loss()
        print(f"step :{steps} train loss = {losses['train_data']:.4f}, val loss = {losses['val_data']:.4f}")
        
        
    xb, yb = get_batch('train')
    logtis, loss = bm(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()


start_x = torch.zeros((1,1), dtype = torch.long, device=device)
rr = ''.join(decode_seq(bm.generate(start_x, 2000)[0].tolist()))
print(rr)

