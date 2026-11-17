import time
import random
import string
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(42)
random.seed(42)

print("Завдання 1")
pairs = [
    ("How can I reset my password?", "You can reset it through the account settings."),
    ("Where can I find my invoice?", "You can download it from the billing page."),
    ("How do I cancel my subscription?", "Go to the subscription tab and click cancel."),
    ("My order is delayed", "We are sorry, please check the tracking page for updates."),
    ("How can I contact support?", "You can write to us in the chat or send an email."),
    ("I want to change my email", "Open the profile settings and update your email."),
    ("How do I update my payment method?", "You can add a new card in the billing settings."),
    ("Can I get a refund?", "Yes, refunds are available within thirty days of purchase."),
    ("The app is not working", "Please restart the app and update it to the latest version."),
    ("How do I delete my account?", "You can delete it in the privacy section of the settings."),
    ("Do you offer a free trial?", "Yes, we offer a free trial for seven days."),
    ("How can I track my order?", "Use the tracking link from the confirmation email.")
]


def clean(text):
    return text.lower().translate(str.maketrans("", "", string.punctuation))


questions = [clean(q).split() for q, a in pairs]
answers = [clean(a).split() for q, a in pairs]

specials = ["<pad>", "<sep>", "<eos>"]
words = sorted(set(w for s in questions + answers for w in s))
itos = specials + words
stoi = {w: i for i, w in enumerate(itos)}
V = len(itos)
PAD, SEP, EOS = 0, 1, 2

max_q = max(len(s) for s in questions)
max_a = max(len(s) for s in answers)
total_len = max_q + 1 + max_a + 1

print("Розмір словника:", V)
print("Максимальна довжина запиту:", max_q)
print("Максимальна довжина відповіді:", max_a)
print("Перші 10 елементів словника token-to-index:", dict(list(stoi.items())[:10]))

sequences = []
for q, a in zip(questions, answers):
    ids = [stoi[w] for w in q] + [SEP] + [stoi[w] for w in a] + [EOS]
    ids = ids + [PAD] * (total_len - len(ids))
    sequences.append(ids)

data = torch.tensor(sequences)
X = data[:, :-1]
Y = data[:, 1:]


class Seq(nn.Module):
    def __init__(self, vocab_size, kind):
        super().__init__()
        self.vocab_size = vocab_size
        if kind == "rnn":
            self.core = nn.RNN(vocab_size, 128, num_layers=1, batch_first=True)
        else:
            self.core = nn.LSTM(vocab_size, 128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, vocab_size)

    def forward(self, x):
        x = F.one_hot(x, self.vocab_size).float()
        out, _ = self.core(x)
        return self.fc(out)


def train(kind, epochs=10, repeat=30):
    model = Seq(V, kind)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset = TensorDataset(X.repeat(repeat, 1), Y.repeat(repeat, 1))
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    start = time.time()
    last = 0
    for epoch in range(epochs):
        model.train()
        total = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out.reshape(-1, V), yb.reshape(-1))
            loss.backward()
            optimizer.step()
            total += loss.item()
        last = total / len(loader)
        print(f"{kind.upper()} | Епоха {epoch + 1}/{epochs} | Loss: {last:.4f}")
    return model, last, time.time() - start


def generate(model, question):
    model.eval()
    ids = [stoi[w] for w in clean(question).split() if w in stoi] + [SEP]
    for _ in range(max_a + 1):
        with torch.no_grad():
            out = model(torch.tensor([ids]))
        nxt = out[0, -1].argmax().item()
        if nxt == EOS:
            break
        ids.append(nxt)
    pos = ids.index(SEP)
    return " ".join(itos[i] for i in ids[pos + 1:])


def evaluate(model):
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(-1)
    correct = 0
    total = 0
    exact = 0
    for i, (q, a) in enumerate(pairs):
        pos = (X[i] == SEP).nonzero()[0].item()
        mask = Y[i] != PAD
        mask[:pos] = False
        correct += (pred[i][mask] == Y[i][mask]).sum().item()
        total += mask.sum().item()
        if generate(model, q) == clean(a):
            exact += 1
    return correct / total, exact


print("\nЗавдання 2")
rnn_model, rnn_loss, rnn_time = train("rnn")

print("\nЗавдання 3")
lstm_model, lstm_loss, lstm_time = train("lstm")

print("\nПередбачені відповіді для 5 запитів:")
for q, a in pairs[:5]:
    print("Запит:", q)
    print("Очікувано:", clean(a))
    print("RNN:", generate(rnn_model, q))
    print("LSTM:", generate(lstm_model, q))
    print()

rnn_acc, rnn_exact = evaluate(rnn_model)
lstm_acc, lstm_exact = evaluate(lstm_model)

comparison = pd.DataFrame({
    "Модель": ["RNN", "LSTM"],
    "Параметри": [sum(p.numel() for p in rnn_model.parameters()), sum(p.numel() for p in lstm_model.parameters())],
    "Фінальний loss": [round(rnn_loss, 4), round(lstm_loss, 4)],
    "Точність токенів": [round(rnn_acc, 4), round(lstm_acc, 4)],
    "Точні відповіді": [f"{rnn_exact}/{len(pairs)}", f"{lstm_exact}/{len(pairs)}"],
    "Час навчання, с": [round(rnn_time, 1), round(lstm_time, 1)]
})
print("Порівняння RNN і LSTM:")
print(comparison.to_string(index=False))

print("\nЗавдання 4")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
gpt = AutoModelForCausalLM.from_pretrained("distilgpt2")
gpt.eval()

prompts = ["I forgot my password", "My order is delayed", "How can I cancel my subscription"]
for p in prompts:
    text = f"Customer: {p}\nSupport:"
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = gpt.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    reply = reply.split("\n")[0]
    print("Запит:", p)
    print("DistilGPT2:", reply)
    print()

gpt_params = sum(p.numel() for p in gpt.parameters())

architectures = pd.DataFrame({
    "Характеристика": [
        "Обробка послідовності",
        "Паралельне навчання",
        "Довгі залежності",
        "Механізм уваги",
        "Кількість параметрів",
        "Потреба в даних",
        "Якість відповідей"
    ],
    "RNN / LSTM": [
        "Послідовно, токен за токеном",
        "Ні, кроки залежать один від одного",
        "RNN погано, LSTM краще",
        "Немає",
        f"{sum(p.numel() for p in lstm_model.parameters()):,} (LSTM)",
        "Можна навчити на малому наборі",
        "Прості шаблонні відповіді"
    ],
    "Transformer (DistilGPT2)": [
        "Увесь контекст одночасно",
        "Так, добре розпаралелюється",
        "Добре завдяки self-attention",
        "Self-attention",
        f"{gpt_params:,}",
        "Потрібні великі корпуси текстів",
        "Зв'язний, різноманітний текст"
    ]
})
print("Порівняння архітектур:")
print(architectures.to_string(index=False))
