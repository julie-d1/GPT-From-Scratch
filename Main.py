# 1. DATA PREPARATION
# Download Tiny Shakespeare if not present
if not os.path.exists("input.txt"):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    with open("input.txt", "w", encoding="utf-8") as file:
        file.write(response.text)

# Read the data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(text[:1000])  # Optional: Show a sample of the text

# Build vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
s2i = {ch: i for i, ch in enumerate(chars)}
i2s = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [s2i[c] for c in s]
decode = lambda l: ''.join([i2s[i] for i in l])

print("Vocab size:", vocab_size)
print("Example encoding:", encode("Hello World!"))
print("Example decoding:", decode(encode("Hello World!")))

encoded_text = encode(text)

# Prepare dataset: sequence-to-sequence for next-char prediction
context_len = 100
source_sequences = [encoded_text[i:i + context_len] for i in range(len(encoded_text)-context_len)]
target_sequences = [encoded_text[i+1:i+1+context_len] for i in range(len(encoded_text)-context_len)]
source_tensor = torch.tensor(source_sequences, dtype=torch.long)
target_tensor = torch.tensor(target_sequences, dtype=torch.long)

# PyTorch dataset/dataloader
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, src_seqs, tgt_seqs):
        self.src_seqs = src_seqs
        self.tgt_seqs = tgt_seqs

    def __len__(self):
        return len(self.src_seqs)

    def __getitem__(self, idx):
        return self.src_seqs[idx], self.tgt_seqs[idx]

dataset = TextDataset(source_tensor, target_tensor)
n = int(0.9*len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n, len(dataset)-n])

batch_size = 64
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 2. MODEL COMPONENTS

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, num_range=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(num_range, d_model)
        position = torch.arange(0, num_range).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Single Head Self-Attention
class Head(nn.Module):
    def __init__(self, d_model, d_k, d_v, context_len=100):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_v)
        self.register_buffer("mask", torch.tril(torch.ones(context_len, context_len)))

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        scores = scores.masked_fill(self.mask[:x.size(1), :x.size(1)] == 0, float('-inf'))
        probs = F.softmax(scores, dim=-1)
        output = probs @ V
        return output

# Multi-Head Self-Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, context_len=100):
        super().__init__()
        self.head_size = d_model // num_heads
        self.heads = nn.ModuleList([Head(d_model, self.head_size, self.head_size, context_len) for _ in range(num_heads)])
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        head_outputs = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.proj(head_outputs)
        return output

# Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.ffn(x)

# Transformer Decoder Layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads=8, d_ff=256, context_len=100):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, context_len)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# Stacked Decoder Layers
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads=8, d_ff=256, context_len=100, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, d_ff, context_len) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Complete GPT Model
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads=8, d_ff=256, context_len=100, num_layers=2):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, context_len)
        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, context_len, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.word_embedding(x)
        x = self.positional_encoding(x)
        x = self.decoder(x)
        logits = self.fc_out(x)
        return logits

    @torch.no_grad()
    def generate(self, start_tokens, max_new_tokens=50, temperature=1.0):
        self.eval()
        generated = start_tokens
        for _ in range(max_new_tokens):
            logits = self.forward(generated[:, -context_len:])
            logits = logits[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated

# 3. TRAINING/EVALUATION FUNCTIONS

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src)
        loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

# 4. MAIN TRAINING LOOP

if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    # Hyperparameters
    d_model = 128
    num_heads = 8
    d_ff = 256
    num_layers = 2
    max_iter = 10

    print(f"data hyper-paras: vocab_size: {vocab_size}, context_len: {context_len}, batch_size: {batch_size}")
    print(f"model hyper-paras: d_model: {d_model}, num_heads: {num_heads}, d_ff: {d_ff}, num_layers: {num_layers}")

    gpt = GPTModel(vocab_size, d_model, num_heads, d_ff, context_len, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gpt.parameters(), lr=0.01)

    train_losses = []
    val_losses = []
    for epoch in range(max_iter):
        train_loss = train(gpt, train_dataloader, criterion, optimizer, device)
        val_loss = evaluate(gpt, val_dataloader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{max_iter}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Optionally plot loss curves
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.show()

    # 5. GENERATION
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = gpt.generate(context, max_new_tokens=200, temperature=1.0)
    print("\nGenerated Sample:")
    print(decode(generated[0].tolist()))
