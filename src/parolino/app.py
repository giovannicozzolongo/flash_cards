"""
Parolino - Italian-German Flashcard App with Deep Learning
==========================================================

PURE PYTHON IMPLEMENTATION (No PyTorch required)
Suitable for Android deployment via BeeWare/Kivy

NEURAL NETWORK ARCHITECTURE:
----------------------------
Type: Multi-Layer Perceptron (MLP) / Feedforward Neural Network
This is SUPERVISED learning (user feedback = labels)

Architecture:
    Input (41 features) 
         |
    Dense(64) + ReLU + Dropout(0.2)
         |
    Dense(32) + ReLU + Dropout(0.2)
         |
    Dense(16) + ReLU
         |
    Dense(1) + Sigmoid -> Priority (0 to 1)

Hyperparameters:
    - Learning rate: 0.001
    - Optimizer: Adam
    - Loss: MSE (Mean Squared Error)
    - Batch size: 5 (mini-batch training)
    - Dropout: 0.2

==========================================================
"""

import toga
from toga.style.pack import COLUMN, ROW, Pack
import random
import os
import re
import json
import asyncio
import time
import math

# Try to use numpy for faster math, fall back to pure Python
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    print("[OK] NumPy available - using accelerated math")
except ImportError:
    NUMPY_AVAILABLE = False
    print("[OK] Pure Python math (no NumPy)")

# Deep learning is always available now (pure Python)
TORCH_AVAILABLE = True
print("[OK] Pure Python Deep Learning enabled!")


# =============================================================================
# PURE PYTHON NEURAL NETWORK IMPLEMENTATION
# =============================================================================

class PurePythonTensor:
    """Simple tensor class for pure Python neural network."""
    
    def __init__(self, data, shape=None):
        if NUMPY_AVAILABLE:
            if isinstance(data, np.ndarray):
                self.data = data.astype(np.float32)
            else:
                self.data = np.array(data, dtype=np.float32)
            self.shape = self.data.shape
        else:
            if isinstance(data, list):
                if shape:
                    self.shape = tuple(shape)
                    self.data = self._flatten(data)
                elif data and isinstance(data[0], list):
                    self.shape = (len(data), len(data[0]))
                    self.data = self._flatten(data)
                else:
                    self.shape = (len(data),)
                    self.data = [float(x) for x in data]
            else:
                self.shape = (1,)
                self.data = [float(data)]
    
    def _flatten(self, lst):
        result = []
        for item in lst:
            if isinstance(item, list):
                result.extend(self._flatten(item))
            else:
                result.append(float(item))
        return result
    
    def __getitem__(self, idx):
        if NUMPY_AVAILABLE:
            return self.data[idx]
        if len(self.shape) == 1:
            return self.data[idx]
        elif len(self.shape) == 2:
            rows, cols = self.shape
            if isinstance(idx, tuple):
                i, j = idx
                return self.data[i * cols + j]
            else:
                start = idx * cols
                return self.data[start:start + cols]
        return self.data[idx]
    
    def __setitem__(self, idx, value):
        if NUMPY_AVAILABLE:
            self.data[idx] = value
        elif len(self.shape) == 2:
            rows, cols = self.shape
            if isinstance(idx, tuple):
                i, j = idx
                self.data[i * cols + j] = float(value)
        else:
            self.data[idx] = float(value)
    
    def tolist(self):
        if NUMPY_AVAILABLE:
            return self.data.tolist()
        if len(self.shape) == 1:
            return list(self.data)
        elif len(self.shape) == 2:
            rows, cols = self.shape
            return [[self.data[i * cols + j] for j in range(cols)] for i in range(rows)]
        return list(self.data)
    
    def copy(self):
        if NUMPY_AVAILABLE:
            return PurePythonTensor(self.data.copy())
        return PurePythonTensor(list(self.data), self.shape)


def tensor_zeros(shape):
    if NUMPY_AVAILABLE:
        return PurePythonTensor(np.zeros(shape, dtype=np.float32))
    if isinstance(shape, int):
        shape = (shape,)
    total = 1
    for s in shape:
        total *= s
    return PurePythonTensor([0.0] * total, shape)


def tensor_randn(shape, scale=1.0):
    if NUMPY_AVAILABLE:
        return PurePythonTensor(np.random.randn(*shape).astype(np.float32) * scale)
    
    if isinstance(shape, int):
        shape = (shape,)
    total = 1
    for s in shape:
        total *= s
    
    data = []
    for _ in range((total + 1) // 2):
        u1 = random.random()
        u2 = random.random()
        while u1 == 0:
            u1 = random.random()
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2) * scale
        z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2) * scale
        data.extend([z0, z1])
    
    return PurePythonTensor(data[:total], shape)


def matmul(a, b):
    if NUMPY_AVAILABLE:
        result = np.dot(a.data, b.data)
        return PurePythonTensor(result)
    
    if len(a.shape) == 1 and len(b.shape) == 2:
        n, m = b.shape
        result = [0.0] * m
        for j in range(m):
            for i in range(n):
                result[j] += a.data[i] * b.data[i * m + j]
        return PurePythonTensor(result, (m,))
    
    elif len(a.shape) == 2 and len(b.shape) == 2:
        p, n = a.shape
        n2, m = b.shape
        assert n == n2
        result = [0.0] * (p * m)
        for i in range(p):
            for j in range(m):
                s = 0.0
                for k in range(n):
                    s += a.data[i * n + k] * b.data[k * m + j]
                result[i * m + j] = s
        return PurePythonTensor(result, (p, m))
    
    elif len(a.shape) == 2 and len(b.shape) == 1:
        p, n = a.shape
        assert n == b.shape[0]
        result = [0.0] * p
        for i in range(p):
            s = 0.0
            for j in range(n):
                s += a.data[i * n + j] * b.data[j]
            result[i] = s
        return PurePythonTensor(result, (p,))
    
    raise ValueError(f"Unsupported shapes for matmul: {a.shape} @ {b.shape}")


def tensor_add(a, b):
    if NUMPY_AVAILABLE:
        return PurePythonTensor(a.data + b.data)
    
    if a.shape != b.shape:
        if len(b.shape) == 1 and len(a.shape) == 2:
            rows, cols = a.shape
            result = list(a.data)
            for i in range(rows):
                for j in range(cols):
                    result[i * cols + j] += b.data[j]
            return PurePythonTensor(result, a.shape)
        elif len(a.shape) == 1 and len(b.shape) == 1:
            assert a.shape[0] == b.shape[0]
    
    result = [a.data[i] + b.data[i] for i in range(len(a.data))]
    return PurePythonTensor(result, a.shape)


class LinearLayer:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.weight = tensor_randn((in_features, out_features), scale)
        self.bias = tensor_zeros(out_features)
        self.weight_grad = tensor_zeros((in_features, out_features))
        self.bias_grad = tensor_zeros(out_features)
        self.input_cache = None
    
    def forward(self, x):
        self.input_cache = x
        out = matmul(x, self.weight)
        out = tensor_add(out, self.bias)
        return out
    
    def backward(self, grad_output):
        x = self.input_cache
        if NUMPY_AVAILABLE:
            if len(x.shape) == 1:
                self.weight_grad.data += np.outer(x.data, grad_output.data)
                self.bias_grad.data += grad_output.data
                grad_input = PurePythonTensor(grad_output.data @ self.weight.data.T)
            else:
                self.weight_grad.data += x.data.T @ grad_output.data
                self.bias_grad.data += np.sum(grad_output.data, axis=0)
                grad_input = PurePythonTensor(grad_output.data @ self.weight.data.T)
        else:
            if len(x.shape) == 1:
                for i in range(self.in_features):
                    for j in range(self.out_features):
                        idx = i * self.out_features + j
                        self.weight_grad.data[idx] += x.data[i] * grad_output.data[j]
                for j in range(self.out_features):
                    self.bias_grad.data[j] += grad_output.data[j]
                grad_input_data = [0.0] * self.in_features
                for i in range(self.in_features):
                    for j in range(self.out_features):
                        grad_input_data[i] += grad_output.data[j] * self.weight.data[i * self.out_features + j]
                grad_input = PurePythonTensor(grad_input_data, (self.in_features,))
            else:
                batch_size = x.shape[0]
                for b in range(batch_size):
                    for i in range(self.in_features):
                        for j in range(self.out_features):
                            idx = i * self.out_features + j
                            x_idx = b * self.in_features + i
                            g_idx = b * self.out_features + j
                            self.weight_grad.data[idx] += x.data[x_idx] * grad_output.data[g_idx]
                for b in range(batch_size):
                    for j in range(self.out_features):
                        g_idx = b * self.out_features + j
                        self.bias_grad.data[j] += grad_output.data[g_idx]
                grad_input_data = [0.0] * (batch_size * self.in_features)
                for b in range(batch_size):
                    for i in range(self.in_features):
                        s = 0.0
                        for j in range(self.out_features):
                            g_idx = b * self.out_features + j
                            s += grad_output.data[g_idx] * self.weight.data[i * self.out_features + j]
                        grad_input_data[b * self.in_features + i] = s
                grad_input = PurePythonTensor(grad_input_data, (batch_size, self.in_features))
        return grad_input
    
    def zero_grad(self):
        self.weight_grad = tensor_zeros((self.in_features, self.out_features))
        self.bias_grad = tensor_zeros(self.out_features)


class ReLU:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        if NUMPY_AVAILABLE:
            self.mask = (x.data > 0).astype(np.float32)
            return PurePythonTensor(np.maximum(0, x.data))
        else:
            self.mask = [1.0 if v > 0 else 0.0 for v in x.data]
            result = [max(0, v) for v in x.data]
            return PurePythonTensor(result, x.shape)
    
    def backward(self, grad_output):
        if NUMPY_AVAILABLE:
            return PurePythonTensor(grad_output.data * self.mask)
        else:
            result = [grad_output.data[i] * self.mask[i] for i in range(len(grad_output.data))]
            return PurePythonTensor(result, grad_output.shape)


class Sigmoid:
    def __init__(self):
        self.output = None
    
    def forward(self, x):
        if NUMPY_AVAILABLE:
            clipped = np.clip(x.data, -500, 500)
            self.output = 1.0 / (1.0 + np.exp(-clipped))
            return PurePythonTensor(self.output)
        else:
            result = []
            for v in x.data:
                v = max(-500, min(500, v))
                result.append(1.0 / (1.0 + math.exp(-v)))
            self.output = result
            return PurePythonTensor(result, x.shape)
    
    def backward(self, grad_output):
        if NUMPY_AVAILABLE:
            grad = self.output * (1 - self.output)
            return PurePythonTensor(grad_output.data * grad)
        else:
            grad = [self.output[i] * (1 - self.output[i]) for i in range(len(self.output))]
            result = [grad_output.data[i] * grad[i] for i in range(len(grad))]
            return PurePythonTensor(result, grad_output.shape)


class Dropout:
    def __init__(self, p=0.2):
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        if NUMPY_AVAILABLE:
            self.mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
            scale = 1.0 / (1.0 - self.p)
            return PurePythonTensor(x.data * self.mask * scale)
        else:
            self.mask = [1.0 if random.random() > self.p else 0.0 for _ in x.data]
            scale = 1.0 / (1.0 - self.p)
            result = [x.data[i] * self.mask[i] * scale for i in range(len(x.data))]
            return PurePythonTensor(result, x.shape)
    
    def backward(self, grad_output):
        if not self.training or self.p == 0:
            return grad_output
        scale = 1.0 / (1.0 - self.p)
        if NUMPY_AVAILABLE:
            return PurePythonTensor(grad_output.data * self.mask * scale)
        else:
            result = [grad_output.data[i] * self.mask[i] * scale for i in range(len(grad_output.data))]
            return PurePythonTensor(result, grad_output.shape)


class AdamOptimizer:
    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.layers = [l for l in layers if isinstance(l, LinearLayer)]
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m_w = []
        self.v_w = []
        self.m_b = []
        self.v_b = []
        for layer in self.layers:
            self.m_w.append(tensor_zeros(layer.weight.shape))
            self.v_w.append(tensor_zeros(layer.weight.shape))
            self.m_b.append(tensor_zeros(layer.bias.shape))
            self.v_b.append(tensor_zeros(layer.bias.shape))
    
    def step(self):
        self.t += 1
        for i, layer in enumerate(self.layers):
            self._update_param(layer.weight, layer.weight_grad, self.m_w[i], self.v_w[i])
            self._update_param(layer.bias, layer.bias_grad, self.m_b[i], self.v_b[i])
    
    def _update_param(self, param, grad, m, v):
        if NUMPY_AVAILABLE:
            m.data = self.beta1 * m.data + (1 - self.beta1) * grad.data
            v.data = self.beta2 * v.data + (1 - self.beta2) * (grad.data ** 2)
            m_hat = m.data / (1 - self.beta1 ** self.t)
            v_hat = v.data / (1 - self.beta2 ** self.t)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        else:
            for j in range(len(param.data)):
                m.data[j] = self.beta1 * m.data[j] + (1 - self.beta1) * grad.data[j]
                v.data[j] = self.beta2 * v.data[j] + (1 - self.beta2) * (grad.data[j] ** 2)
                m_hat = m.data[j] / (1 - self.beta1 ** self.t)
                v_hat = v.data[j] / (1 - self.beta2 ** self.t)
                param.data[j] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()


class WordPriorityNetwork:
    def __init__(self, input_dim=41):
        self.input_dim = input_dim
        self.fc1 = LinearLayer(input_dim, 64)
        self.relu1 = ReLU()
        self.drop1 = Dropout(0.2)
        self.fc2 = LinearLayer(64, 32)
        self.relu2 = ReLU()
        self.drop2 = Dropout(0.2)
        self.fc3 = LinearLayer(32, 16)
        self.relu3 = ReLU()
        self.fc4 = LinearLayer(16, 1)
        self.sigmoid = Sigmoid()
        self.layers = [
            self.fc1, self.relu1, self.drop1,
            self.fc2, self.relu2, self.drop2,
            self.fc3, self.relu3,
            self.fc4, self.sigmoid
        ]
        self.training_count = 0
        self.total_loss = 0.0
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                grad = layer.backward(grad)
        return grad
    
    def train_mode(self):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = True
    
    def eval_mode(self):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = False
    
    def get_state(self):
        state = {'training_count': self.training_count, 'total_loss': self.total_loss, 'layers': []}
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                state['layers'].append({'weight': layer.weight.tolist(), 'bias': layer.bias.tolist()})
        return state
    
    def load_state(self, state):
        self.training_count = state.get('training_count', 0)
        self.total_loss = state.get('total_loss', 0.0)
        layer_states = state.get('layers', [])
        linear_layers = [l for l in self.layers if isinstance(l, LinearLayer)]
        for layer, layer_state in zip(linear_layers, layer_states):
            layer.weight = PurePythonTensor(layer_state['weight'])
            layer.bias = PurePythonTensor(layer_state['bias'])


# =============================================================================
# APP INFO
# =============================================================================
APP_VERSION = "1.2"
APP_AUTHOR = "Giovanni Cozzolongo"

INFO_TEXT = {
    "italian": f"""Parolino ti aiuta a memorizzare vocaboli italiano e tedesco con le flashcard. Nella scheda ABC cerchi le parole e con un doppio click le aggiungi al tuo mazzo, la stella indica che la parola è già nel mazzo. Nella scheda Flashcards studi le carte e premi più se hai indovinato o meno se hai sbagliato, così l'app impara dai tuoi errori e ti ripropone più spesso le parole difficili.

Parolino v{APP_VERSION}
© 2025 {APP_AUTHOR}""",

    "german": f"""Parolino hilft dir, italienische und deutsche Vokabeln mit Karteikarten zu lernen. Im ABC Tab suchst du Wörter und fügst sie mit Doppelklick zu deinem Stapel hinzu. Im Flashcards Tab lernst du die Karten und drückst Plus wenn du es wusstest oder Minus wenn nicht.

Parolino v{APP_VERSION}
© 2025 {APP_AUTHOR}"""
}

TRANSLATIONS = {
    "italian": {
        "search_placeholder": "Cerca parole...",
        "click_info": "Doppio click per aggiungere/rimuovere",
        "clear_confirm": "Svuotare il deck?",
        "info_title": "Info"
    },
    "german": {
        "search_placeholder": "Wörter suchen...",
        "click_info": "Doppelklick zum Hinzufügen/Entfernen",
        "clear_confirm": "Deck leeren?",
        "info_title": "Info"
    }
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def extract_gender(word_raw):
    match = re.search(r'\{([mfn]|pl)\}', word_raw)
    return match.group(1) if match else None

def clean_word_for_display(word_raw):
    word = re.sub(r'\s*\{[^}]*\}', '', word_raw)
    word = re.sub(r'\s*\[[^\]]*\]', '', word)
    return word.strip()

def extract_base_word(word_raw):
    bracket_pos = word_raw.find('[')
    return word_raw[:bracket_pos].strip() if bracket_pos > 0 else word_raw

def normalize_for_search(word):
    word = re.sub(r'[!?.,:;]+$', '', word)
    return word.strip().lower()

def is_proper_noun(word, translation):
    w1, w2 = word.lower(), translation.lower()
    if len(w1) >= 3 and len(w2) >= 3:
        common_start = 0
        for i in range(min(len(w1), len(w2))):
            if w1[i] == w2[i]:
                common_start += 1
            else:
                break
        if common_start >= 3 and common_start >= min(len(w1), len(w2)) * 0.5:
            return True
    common_names = {'giovanni', 'giuseppe', 'maria', 'anna', 'paolo', 'pietro', 'francesco',
                   'johannes', 'josef', 'paul', 'peter', 'franz', 'marco', 'luca', 'matteo',
                   'andrea', 'carlo', 'luigi', 'antonio', 'markus', 'lukas', 'matthias'}
    return w1 in common_names or w2 in common_names

def add_german_article(word, gender, word_type, original_word=None, translation=None):
    if word_type != 'noun' or not gender:
        return word
    if original_word and translation and is_proper_noun(original_word, translation):
        return word
    articles = {'m': 'der', 'f': 'die', 'n': 'das', 'pl': 'die'}
    article = articles.get(gender)
    return f"{article} {word}" if article else word

def load_dictionary(filepath):
    dictionary = []
    seen = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                if line.startswith('ITALIANO') or line.startswith('TEDESCO'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                word1_raw, word2_raw = parts[0].strip(), parts[1].strip()
                word_type = parts[2].strip() if len(parts) > 2 else ""
                categories = parts[3].strip() if len(parts) > 3 else ""
                if not word1_raw or not word2_raw:
                    continue
                word1_base = extract_base_word(word1_raw)
                gender = extract_gender(word1_base)
                word1_display = clean_word_for_display(word1_base)
                word1_search = normalize_for_search(word1_display)
                if not word1_search or ' ' in word1_search or len(word1_search) > 30:
                    continue
                if word1_display.lower() in seen:
                    continue
                seen.add(word1_display.lower())
                word2_main = word2_raw.split(' / ')[0].strip() if ' / ' in word2_raw else word2_raw
                gender2 = extract_gender(word2_main)
                word2_display = clean_word_for_display(word2_main)
                entry_type = "general"
                wt_lower = word_type.lower()
                if "noun" in wt_lower: entry_type = "noun"
                elif "verb" in wt_lower: entry_type = "verb"
                elif "adj" in wt_lower: entry_type = "adjective"
                elif "adv" in wt_lower: entry_type = "adverb"
                topic = "general"
                cat_lower = categories.lower()
                if "[gastr.]" in cat_lower: topic = "food"
                elif "[bot.]" in cat_lower or "[zool.]" in cat_lower: topic = "nature"
                elif "[med.]" in cat_lower: topic = "health"
                elif "[sport" in cat_lower: topic = "sports"
                dictionary.append({
                    "word_display": word1_display, "word_search": word1_search,
                    "translation": word2_display, "gender": gender, "gender2": gender2,
                    "type": entry_type, "topic": topic
                })
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return dictionary


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
class WordFeatureExtractor:
    TOPICS = ["general", "food", "nature", "health", "sports", "work", "science", "music", "religion"]
    TYPES = ["general", "noun", "verb", "adjective", "adverb"]
    GENDERS = ["m", "f", "n", "pl", None]
    
    @staticmethod
    def extract(word, translation, word_type, topic, gender, gender2, score, lang,
                times_seen=0, days_since_last=0, recent_history=None):
        f = []
        f.append(min(len(word) / 20.0, 1.0))
        f.append(min(len(translation) / 20.0, 1.0))
        vowels = len(re.findall(r'[aeiouäöüAEIOUÄÖÜ]', word))
        f.append(min(vowels / 8.0, 1.0))
        f.append(min(times_seen / 20.0, 1.0))
        f.append(min(days_since_last / 30.0, 1.0))
        if recent_history and len(recent_history) > 0:
            recent_acc = sum(recent_history) / len(recent_history)
        else:
            recent_acc = 0.5
        f.append(recent_acc)
        streak = 0
        if recent_history:
            last_val = recent_history[-1] if recent_history else None
            for val in reversed(recent_history):
                if val == last_val:
                    streak += 1
                else:
                    break
            if not last_val:
                streak = -streak
        f.append((streak + 10) / 20.0)
        f.append(1.0 if any(c in word for c in 'äöüÄÖÜ') else 0.0)
        f.append(1.0 if any(c in translation for c in 'äöüÄÖÜ') else 0.0)
        f.append(1.0 if len(word) > 10 else 0.0)
        f.append(1.0 if 'ß' in word or 'ß' in translation else 0.0)
        f.append(1.0 if word and word[0].isupper() else 0.0)
        f.append(1.0 if lang == "italian" else 0.0)
        f.append(1.0 if lang == "german" else 0.0)
        for t in WordFeatureExtractor.TOPICS:
            f.append(1.0 if topic == t else 0.0)
        for wt in WordFeatureExtractor.TYPES:
            f.append(1.0 if word_type == wt else 0.0)
        for g in WordFeatureExtractor.GENDERS:
            f.append(1.0 if gender == g else 0.0)
        for g in WordFeatureExtractor.GENDERS:
            f.append(1.0 if gender2 == g else 0.0)
        f.append((score + 5) / 10.0)
        f.append(1.0 if score < -2 else 0.0)
        f.append(1.0 if score > 2 else 0.0)
        return f
    
    @staticmethod
    def dim():
        return 41


# =============================================================================
# SMART DECK SELECTOR
# =============================================================================
class SmartDeckSelector:
    BATCH_SIZE = 5
    
    def __init__(self, model_path=None):
        self.model = None
        self.optimizer = None
        self.loss_history = []
        self.prediction_log = []
        self.batch_x = []
        self.batch_y = []
        self.model = WordPriorityNetwork(input_dim=WordFeatureExtractor.dim())
        self.optimizer = AdamOptimizer(self.model.layers, lr=0.001)
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                self.model.load_state(checkpoint)
                self.loss_history = checkpoint.get('loss_history', [])
                print(f"[OK] Loaded model: {self.model.training_count} training samples")
            except Exception as e:
                print(f"[!!] Could not load model: {e}")
    
    def save_model(self, model_path):
        if self.model:
            try:
                state = self.model.get_state()
                state['loss_history'] = self.loss_history[-500:]
                if model_path.endswith('.pth'):
                    model_path = model_path[:-4] + '.json'
                with open(model_path, 'w', encoding='utf-8') as f:
                    json.dump(state, f)
            except Exception as e:
                print(f"[!!] Could not save model: {e}")
    
    def _get_temporal_features(self, word_data):
        times_seen = word_data.get('times_seen', 0)
        last_seen = word_data.get('last_seen', 0)
        history = word_data.get('history', [])
        if last_seen > 0:
            days_since = (time.time() - last_seen) / 86400.0
        else:
            days_since = 30
        return times_seen, days_since, history[-5:] if history else []
    
    def compute_priority(self, entry, word_data, lang):
        if not self.model:
            score = word_data.get('score', 0)
            return (5 - score) / 10.0
        score = word_data.get('score', 0)
        times_seen, days_since, recent_history = self._get_temporal_features(word_data)
        features = WordFeatureExtractor.extract(
            entry["word_display"], entry["translation"], entry["type"], entry["topic"],
            entry.get("gender"), entry.get("gender2"), score, lang, times_seen, days_since, recent_history
        )
        self.model.eval_mode()
        x = PurePythonTensor(features)
        output = self.model.forward(x)
        if NUMPY_AVAILABLE:
            priority = float(output.data[0]) if len(output.shape) > 0 else float(output.data)
        else:
            priority = output.data[0] if isinstance(output.data, list) else output.data
        score_priority = (5 - score) / 10.0
        spacing_boost = min(days_since / 7.0, 0.3)
        return 0.6 * priority + 0.2 * score_priority + 0.2 * spacing_boost
    
    def train_step(self, entry, word_data, was_correct, lang):
        if not self.model:
            return None
        score = word_data.get('score', 0)
        times_seen, days_since, recent_history = self._get_temporal_features(word_data)
        features = WordFeatureExtractor.extract(
            entry["word_display"], entry["translation"], entry["type"], entry["topic"],
            entry.get("gender"), entry.get("gender2"), score, lang, times_seen, days_since, recent_history
        )
        target = 0.1 if was_correct else 0.9
        self.batch_x.append(features)
        self.batch_y.append(target)
        loss_val = None
        pred_val = None
        if len(self.batch_x) >= self.BATCH_SIZE:
            self.model.train_mode()
            self.optimizer.zero_grad()
            total_loss = 0.0
            for i in range(len(self.batch_x)):
                x = PurePythonTensor(self.batch_x[i])
                y_target = self.batch_y[i]
                output = self.model.forward(x)
                if NUMPY_AVAILABLE:
                    pred = float(output.data[0]) if len(output.shape) > 0 else float(output.data)
                else:
                    pred = output.data[0] if isinstance(output.data, list) else output.data
                error = pred - y_target
                loss = error ** 2
                total_loss += loss
                grad = PurePythonTensor([2 * error])
                self.model.backward(grad)
            loss_val = total_loss / len(self.batch_x)
            pred_val = pred
            self.optimizer.step()
            self.model.training_count += len(self.batch_x)
            self.model.total_loss += loss_val * len(self.batch_x)
            self.loss_history.append(loss_val)
            self.batch_x = []
            self.batch_y = []
        self.prediction_log.append({
            'word': entry["word_display"], 'lang': lang, 'score': score,
            'was_correct': was_correct, 'prediction': pred_val, 'target': target, 'loss': loss_val
        })
        return loss_val, pred_val
    
    def select_words(self, deck, entries_it, entries_de, word_data_dict, max_count):
        if not deck:
            return []
        priorities = []
        for word_key in deck:
            lang, word = word_key.split(":", 1)
            lang = lang.lower()
            entries = entries_it if lang == "italian" else entries_de
            entry = entries.get(word.lower())
            word_data = word_data_dict.get(word_key, {'score': 0})
            if entry:
                priority = self.compute_priority(entry, word_data, lang)
            else:
                score = word_data.get('score', 0)
                priority = (5 - score) / 10.0
            priority += random.uniform(0, 0.05)
            priorities.append((word_key, priority))
        priorities.sort(key=lambda x: x[1], reverse=True)
        selected = [w for w, _ in priorities[:max_count]]
        random.shuffle(selected)
        return selected
    
    def print_training_log(self):
        if not self.model:
            print("No model available")
            return
        print("\n" + "="*60)
        print("NEURAL NETWORK LOG (Pure Python)")
        print("="*60)
        n = self.model.training_count
        avg_loss = self.model.total_loss / max(1, n)
        print(f"Samples: {n}, Avg loss: {avg_loss:.4f}")
        if self.loss_history and len(self.loss_history) >= 5:
            recent = self.loss_history[-10:]
            print(f"Recent loss: {sum(recent)/len(recent):.4f}")
        print("="*60 + "\n")


# =============================================================================
# MAIN APPLICATION
# =============================================================================
class Parolino(toga.App):
    def startup(self):
        print("\n" + "="*50)
        print("PAROLINO STARTUP (Pure Python DL)")
        print("="*50)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.deck_save_path = os.path.join(script_dir, 'deck_save.json')
        self.model_save_path = os.path.join(script_dir, 'word_model.json')
        
        self.dict_it = []
        self.dict_de = []
        
        for path in [os.path.join(script_dir, 'resources', 'it_de.txt'),
                     os.path.join(script_dir, 'it_de.txt')]:
            if os.path.exists(path):
                self.dict_it = load_dictionary(path)
                print(f"[OK] Italian dictionary: {len(self.dict_it)} words")
                break
        
        for path in [os.path.join(script_dir, 'resources', 'de_it.txt'),
                     os.path.join(script_dir, 'de_it.txt')]:
            if os.path.exists(path):
                self.dict_de = load_dictionary(path)
                print(f"[OK] German dictionary: {len(self.dict_de)} words")
                break
        
        self.entries_it = {e["word_display"].lower(): e for e in self.dict_it}
        self.entries_de = {e["word_display"].lower(): e for e in self.dict_de}
        
        self.deck = set()
        self.shuffled_deck = []
        self.word_data = {}
        self.max_deck_size = None
        self.current_dict = "italian"
        self.card_direction = "it_de"
        self.current_index = 0
        self.show_front = True
        self.ui_language = "italian"
        
        self.selector = SmartDeckSelector(self.model_save_path)
        self._load_deck()
        
        # UI - main container
        main_box = toga.Box(style=Pack(direction=COLUMN))
        
        self.dict_view = self._create_dict_view()
        self.flash_view = self._create_flash_view()
        self.stats_view = self._create_stats_view()
        
        self.tabs = toga.OptionContainer(
            style=Pack(flex=1),
            content=[
                ("ABC", self.dict_view),
                ("Flashcards", self.flash_view),
                ("Nerd", self.stats_view)
            ]
        )
        
        main_box.add(self.tabs)
        
        self.main_window = toga.MainWindow(title="Parolino", size=(420, 580))
        self.main_window.content = main_box
        self.main_window.show()
        
        self._update_flash_view()
        self._update_stats_view()
        
        print("="*50)
        print("STARTUP COMPLETE")
        print("="*50 + "\n")
    
    def _load_deck(self):
        try:
            if os.path.exists(self.deck_save_path):
                with open(self.deck_save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.deck = set(data.get('deck', []))
                    self.shuffled_deck = data.get('shuffled_deck', [])
                    self.max_deck_size = data.get('max_deck_size')
                    self.current_index = data.get('current_index', 0)
                    if 'word_data' in data:
                        self.word_data = data['word_data']
                    elif 'word_scores' in data:
                        for key, score in data['word_scores'].items():
                            self.word_data[key] = {'score': score, 'times_seen': abs(score), 
                                                   'last_seen': time.time(), 'history': []}
                    print(f"[OK] Loaded deck: {len(self.deck)} words")
        except Exception as e:
            print(f"[!!] Error loading deck: {e}")
        if not self.shuffled_deck:
            self._rebuild_deck()
    
    def _save_deck(self):
        try:
            with open(self.deck_save_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'deck': list(self.deck), 'shuffled_deck': self.shuffled_deck,
                    'word_data': self.word_data, 'max_deck_size': self.max_deck_size,
                    'current_index': self.current_index
                }, f, ensure_ascii=False, indent=2)
            self.selector.save_model(self.model_save_path)
        except Exception as e:
            print(f"[!!] Error saving: {e}")
    
    def _rebuild_deck(self):
        if not self.deck:
            self.shuffled_deck = []
            self.current_index = 0
            return
        max_count = self.max_deck_size or len(self.deck)
        self.shuffled_deck = self.selector.select_words(
            self.deck, self.entries_it, self.entries_de, self.word_data, max_count
        )
        self.current_index = 0
    
    def _t(self, key):
        return TRANSLATIONS[self.ui_language].get(key, key)
    
    def _show_info(self, widget):
        self.main_window.info_dialog(self._t("info_title"), INFO_TEXT[self.ui_language])
    
    def _get_word_data(self, word_key):
        if word_key not in self.word_data:
            self.word_data[word_key] = {'score': 0, 'times_seen': 0, 'last_seen': 0, 'history': []}
        return self.word_data[word_key]
    
    # =========================================================================
    # DICTIONARY VIEW - ANDROID FIXED
    # =========================================================================
    def _create_dict_view(self):
        # Use ScrollContainer for Android compatibility
        scroll = toga.ScrollContainer(horizontal=False, style=Pack(flex=1))
        container = toga.Box(style=Pack(direction=COLUMN, padding=5))
        
        # Top buttons
        lang_bar = toga.Box(style=Pack(direction=ROW))
        self.btn_it = toga.Button("IT", on_press=self._select_italian, style=Pack(flex=1, height=50, padding=2))
        self.btn_de = toga.Button("DE", on_press=self._select_german, style=Pack(flex=1, height=50, padding=2))
        self.btn_clear = toga.Button("X", on_press=self._clear_deck, style=Pack(width=50, height=50, padding=2))
        self.btn_info = toga.Button("?", on_press=self._show_info, style=Pack(width=50, height=50, padding=2))
        lang_bar.add(self.btn_it)
        lang_bar.add(self.btn_de)
        lang_bar.add(self.btn_clear)
        lang_bar.add(self.btn_info)
        container.add(lang_bar)
        
        # Search
        self.search_input = toga.TextInput(
            placeholder=self._t("search_placeholder"),
            on_change=self._on_search,
            style=Pack(height=50, padding=5)
        )
        container.add(self.search_input)
        
        # Word list - fixed height for Android (will scroll inside ScrollContainer)
        self.word_list = toga.Table(
            headings=["Parola", "★"],
            accessors=["word", "star"],
            data=[],
            on_activate=self._on_word_click,
            style=Pack(height=400, padding=5)
        )
        container.add(self.word_list)
        
        # Info label
        self.info_label = toga.Label(
            self._t("click_info"),
            style=Pack(padding=5, font_size=11)
        )
        container.add(self.info_label)
        
        scroll.content = container
        self._update_word_list()
        return scroll
    
    def _select_italian(self, widget):
        self.current_dict = "italian"
        self.ui_language = "italian"
        self.search_input.value = ""
        self.search_input.placeholder = self._t("search_placeholder")
        self.info_label.text = self._t("click_info")
        self._update_word_list()
    
    def _select_german(self, widget):
        self.current_dict = "german"
        self.ui_language = "german"
        self.search_input.value = ""
        self.search_input.placeholder = self._t("search_placeholder")
        self.info_label.text = self._t("click_info")
        self._update_word_list()
    
    def _on_search(self, widget):
        self._update_word_list()
    
    def _update_word_list(self):
        search = self.search_input.value.strip().lower() if hasattr(self, 'search_input') else ""
        dictionary = self.dict_it if self.current_dict == "italian" else self.dict_de
        lang_key = "ITALIAN" if self.current_dict == "italian" else "GERMAN"
        data = []
        if not search:
            for word_key in sorted(self.deck):
                lang, word = word_key.split(":", 1)
                if lang != lang_key:
                    continue
                data.append({"word": word, "star": "★"})
        else:
            for entry in dictionary:
                if not entry["word_search"].startswith(search):
                    continue
                word_key = f"{lang_key}:{entry['word_display']}"
                star = "★" if word_key in self.deck else ""
                data.append({"word": entry["word_display"], "star": star})
        self.word_list.data = data
    
    def _on_word_click(self, widget, row):
        if not row:
            return
        lang_key = "ITALIAN" if self.current_dict == "italian" else "GERMAN"
        word_key = f"{lang_key}:{row.word}"
        if word_key in self.deck:
            self.deck.remove(word_key)
            if word_key in self.shuffled_deck:
                self.shuffled_deck.remove(word_key)
        else:
            self.deck.add(word_key)
            self._get_word_data(word_key)
            self.shuffled_deck.append(word_key)
        self._save_deck()
        self._update_word_list()
        self._update_flash_view()
    
    def _clear_deck(self, widget):
        asyncio.create_task(self._confirm_clear())
    
    async def _confirm_clear(self):
        if not self.deck:
            return
        confirmed = await self.main_window.confirm_dialog("Clear", self._t("clear_confirm") + f" ({len(self.deck)})")
        if confirmed:
            self.deck.clear()
            self.shuffled_deck.clear()
            self.word_data.clear()
            self._save_deck()
            self._update_word_list()
            self._update_flash_view()
    
    # =========================================================================
    # FLASHCARD VIEW - ANDROID FIXED
    # =========================================================================
    def _create_flash_view(self):
        # Use ScrollContainer for Android compatibility
        scroll = toga.ScrollContainer(horizontal=False, style=Pack(flex=1))
        container = toga.Box(style=Pack(direction=COLUMN, padding=5))
        
        # Top bar
        top_bar = toga.Box(style=Pack(direction=ROW))
        self.btn_itde = toga.Button("IT→DE", on_press=self._set_it_de, style=Pack(flex=1, height=50, padding=2))
        self.btn_deit = toga.Button("DE→IT", on_press=self._set_de_it, style=Pack(flex=1, height=50, padding=2))
        self.max_input = toga.TextInput(value=str(self.max_deck_size or ""), on_change=self._on_max_change, style=Pack(width=55, height=50, padding=2))
        self.btn_new = toga.Button("⇄", on_press=self._generate_deck, style=Pack(width=55, height=50, padding=2))
        top_bar.add(self.btn_itde)
        top_bar.add(self.btn_deit)
        top_bar.add(self.max_input)
        top_bar.add(self.btn_new)
        container.add(top_bar)
        
        # Counter
        self.counter_label = toga.Label("", style=Pack(height=40, padding=10, text_align="center"))
        container.add(self.counter_label)
        
        # Card - fixed height for Android
        card_box = toga.Box(style=Pack(direction=COLUMN, height=200, padding=20))
        self.card_label = toga.Label("", style=Pack(font_size=32, text_align="center"))
        card_box.add(self.card_label)
        container.add(card_box)
        
        # Nav buttons
        nav_box = toga.Box(style=Pack(direction=ROW))
        self.btn_prev = toga.Button("<", on_press=self._prev_card, style=Pack(flex=1, height=60, padding=2))
        self.btn_flip = toga.Button("↻", on_press=self._flip_card, style=Pack(flex=1, height=60, padding=2))
        self.btn_next = toga.Button(">", on_press=self._next_card, style=Pack(flex=1, height=60, padding=2))
        nav_box.add(self.btn_prev)
        nav_box.add(self.btn_flip)
        nav_box.add(self.btn_next)
        container.add(nav_box)
        
        # Feedback buttons
        feedback_box = toga.Box(style=Pack(direction=ROW))
        self.btn_wrong = toga.Button("−", on_press=self._mark_wrong, style=Pack(flex=1, height=60, padding=2))
        self.btn_correct = toga.Button("+", on_press=self._mark_correct, style=Pack(flex=1, height=60, padding=2))
        feedback_box.add(self.btn_wrong)
        feedback_box.add(self.btn_correct)
        container.add(feedback_box)
        
        scroll.content = container
        return scroll
    
    def _set_it_de(self, widget):
        self.card_direction = "it_de"
        self.show_front = True
        self._update_flash_view()
    
    def _set_de_it(self, widget):
        self.card_direction = "de_it"
        self.show_front = True
        self._update_flash_view()
    
    def _on_max_change(self, widget):
        try:
            val = int(widget.value) if widget.value else None
            if val and val > 0:
                self.max_deck_size = val
                self._save_deck()
        except ValueError:
            pass
    
    def _update_flash_view(self):
        if not self.shuffled_deck:
            self.card_label.text = ""
            self.counter_label.text = "No cards"
            return
        if self.current_index >= len(self.shuffled_deck):
            self.current_index = 0
        word_key = self.shuffled_deck[self.current_index]
        lang, word = word_key.split(":", 1)
        entries = self.entries_it if lang == "ITALIAN" else self.entries_de
        entry = entries.get(word.lower())
        if not entry:
            self.card_label.text = word
            return
        self.counter_label.text = f"{self.current_index + 1}/{len(self.shuffled_deck)}"
        is_italian = lang == "ITALIAN"
        if self.card_direction == "it_de":
            if self.show_front:
                self.card_label.text = entry["word_display"] if is_italian else entry["translation"]
            else:
                if is_italian:
                    text = entry["translation"]
                    if entry["type"] == "noun" and entry.get("gender2"):
                        text = add_german_article(text, entry["gender2"], "noun", entry["word_display"], entry["translation"])
                else:
                    text = entry["word_display"]
                    if entry["type"] == "noun" and entry.get("gender"):
                        text = add_german_article(text, entry["gender"], "noun", entry["translation"], entry["word_display"])
                self.card_label.text = text
        else:
            if self.show_front:
                if is_italian:
                    text = entry["translation"]
                    if entry["type"] == "noun" and entry.get("gender2"):
                        text = add_german_article(text, entry["gender2"], "noun", entry["word_display"], entry["translation"])
                else:
                    text = entry["word_display"]
                    if entry["type"] == "noun" and entry.get("gender"):
                        text = add_german_article(text, entry["gender"], "noun", entry["translation"], entry["word_display"])
                self.card_label.text = text
            else:
                self.card_label.text = entry["word_display"] if is_italian else entry["translation"]
    
    def _flip_card(self, widget):
        if self.shuffled_deck:
            self.show_front = not self.show_front
            self._update_flash_view()
    
    def _next_card(self, widget):
        if self.shuffled_deck:
            self.current_index = (self.current_index + 1) % len(self.shuffled_deck)
            self.show_front = True
            self._update_flash_view()
            self._save_deck()
    
    def _prev_card(self, widget):
        if self.shuffled_deck:
            self.current_index = (self.current_index - 1) % len(self.shuffled_deck)
            self.show_front = True
            self._update_flash_view()
            self._save_deck()
    
    def _get_current_entry(self):
        if not self.shuffled_deck:
            return None, None, None, None
        word_key = self.shuffled_deck[self.current_index]
        lang, word = word_key.split(":", 1)
        entries = self.entries_it if lang == "ITALIAN" else self.entries_de
        entry = entries.get(word.lower())
        return word_key, entry, lang.lower(), self._get_word_data(word_key)
    
    def _mark_correct(self, widget):
        word_key, entry, lang, word_data = self._get_current_entry()
        if not entry:
            return
        word_data['score'] = min(5, word_data['score'] + 1)
        word_data['times_seen'] = word_data.get('times_seen', 0) + 1
        word_data['last_seen'] = time.time()
        word_data.setdefault('history', []).append(True)
        word_data['history'] = word_data['history'][-10:]
        result = self.selector.train_step(entry, word_data, was_correct=True, lang=lang)
        if result and result[0] is not None:
            print(f"[+] {entry['word_display'][:12]:<12} loss:{result[0]:.4f}")
        self._save_deck()
        self._next_card(widget)
    
    def _mark_wrong(self, widget):
        word_key, entry, lang, word_data = self._get_current_entry()
        if not entry:
            return
        word_data['score'] = max(-5, word_data['score'] - 1)
        word_data['times_seen'] = word_data.get('times_seen', 0) + 1
        word_data['last_seen'] = time.time()
        word_data.setdefault('history', []).append(False)
        word_data['history'] = word_data['history'][-10:]
        result = self.selector.train_step(entry, word_data, was_correct=False, lang=lang)
        if result and result[0] is not None:
            print(f"[-] {entry['word_display'][:12]:<12} loss:{result[0]:.4f}")
        self._save_deck()
        self._next_card(widget)
    
    def _generate_deck(self, widget):
        if not self.deck:
            return
        try:
            max_size = int(self.max_input.value) if self.max_input.value else len(self.deck)
        except ValueError:
            max_size = len(self.deck)
        self.shuffled_deck = self.selector.select_words(
            self.deck, self.entries_it, self.entries_de, self.word_data, max_size
        )
        self.current_index = 0
        self.show_front = True
        self._update_flash_view()
        self._save_deck()
        self.selector.print_training_log()
    
    # =========================================================================
    # NERD VIEW
    # =========================================================================
    def _create_stats_view(self):
        scroll = toga.ScrollContainer(horizontal=False, style=Pack(flex=1))
        container = toga.Box(style=Pack(direction=COLUMN, padding=10))
        
        self.stats_text = toga.MultilineTextInput(readonly=True, style=Pack(height=300, padding=5))
        container.add(self.stats_text)
        
        btn_refresh = toga.Button("Refresh", on_press=self._refresh_stats, style=Pack(height=50, padding=5))
        container.add(btn_refresh)
        
        btn_priorities = toga.Button("Priorities", on_press=self._show_priorities, style=Pack(height=50, padding=5))
        container.add(btn_priorities)
        
        btn_log = toga.Button("Full Log", on_press=self._print_log, style=Pack(height=50, padding=5))
        container.add(btn_log)
        
        scroll.content = container
        return scroll
    
    def _update_stats_view(self):
        if not hasattr(self, 'stats_text'):
            return
        lines = ["Pure Python Neural Net", ""]
        if self.selector.model:
            n = self.selector.model.training_count
            avg = self.selector.model.total_loss / max(1, n)
            lines.append(f"Samples: {n}")
            lines.append(f"Loss: {avg:.3f}")
            if self.selector.loss_history and len(self.selector.loss_history) >= 3:
                recent = self.selector.loss_history[-5:]
                lines.append(f"Recent: {sum(recent)/len(recent):.3f}")
            lines.append("")
            if n < 20:
                lines.append("Status: warming up")
            elif n < 50:
                lines.append("Status: learning")
            else:
                lines.append("Status: trained")
        else:
            lines.append("No model")
        self.stats_text.value = "\n".join(lines)
    
    def _refresh_stats(self, widget):
        self._update_stats_view()
    
    def _show_priorities(self, widget):
        if not self.deck:
            self.stats_text.value = "No words"
            return
        lines = ["Priority ranking:", ""]
        priorities = []
        for word_key in self.deck:
            lang, word = word_key.split(":", 1)
            entries = self.entries_it if lang == "ITALIAN" else self.entries_de
            entry = entries.get(word.lower())
            word_data = self.word_data.get(word_key, {'score': 0})
            if entry:
                priority = self.selector.compute_priority(entry, word_data, lang.lower())
            else:
                priority = 0.5
            priorities.append((word, priority))
        priorities.sort(key=lambda x: x[1], reverse=True)
        for word, pri in priorities[:20]:
            lines.append(f"{word[:12]:<12} {pri:.2f}")
        self.stats_text.value = "\n".join(lines)
    
    def _print_log(self, widget):
        self.selector.print_training_log()
        self.stats_text.value = "Log printed to terminal"


def main():
    return Parolino()