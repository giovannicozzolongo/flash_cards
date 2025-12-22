"""
Parolino - Italian-German Flashcard App with Deep Learning
==========================================================

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

FEATURES (41 total):
-------------------
Continuous (7):
    - Word length (normalized 0-1)
    - Translation length (normalized 0-1)
    - Vowel count (normalized 0-1)
    - Times seen (normalized 0-1)
    - Days since last seen (normalized 0-1)
    - Recent accuracy (last 5 attempts)
    - Current streak (normalized)

Binary (7):
    - Has umlauts (ä, ö, ü) in word
    - Has umlauts in translation
    - Is long word (>10 chars)
    - Has ß (eszett)
    - First letter uppercase
    - Is Italian
    - Is German

One-hot encoded:
    - Topic (9): general, food, nature, health, sports, work, science, music, religion
    - Type (5): general, noun, verb, adjective, adverb
    - Gender word (5): m, f, n, pl, None
    - Gender translation (5): m, f, n, pl, None

Score features (3):
    - Normalized score (0-1)
    - Very negative (<-2)
    - Very positive (>2)

TRAINING:
---------
- User presses + → target = 0.1 (low priority, knows it)
- User presses - → target = 0.9 (high priority, needs practice)
- Mini-batch: accumulates 5 samples, then trains together
- Selection: picks words with HIGHEST predicted priority

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

# PyTorch
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    print("[OK] PyTorch loaded - Deep Learning enabled!")
except ImportError:
    print("[!!] PyTorch not available - install with: pip install torch")


# =============================================================================
# APP INFO
# =============================================================================
APP_VERSION = "1.1"
APP_AUTHOR = "Giovanni Cozzolongo"

INFO_TEXT = {
    "italian": f"""Parolino ti aiuta a memorizzare vocaboli italiano e tedesco con le flashcard. Nella scheda ABC cerchi le parole e con un doppio click le aggiungi al tuo mazzo, la stella indica che la parola è già nel mazzo. Nella scheda Flashcards studi le carte e premi più se hai indovinato o meno se hai sbagliato, così l'app impara dai tuoi errori e ti ripropone più spesso le parole difficili. Il numero nel campo di testo limita quante carte pescare e il bottone con le frecce genera un nuovo mazzo.

Parolino v{APP_VERSION}
© 2025 {APP_AUTHOR}""",

    "german": f"""Parolino hilft dir, italienische und deutsche Vokabeln mit Karteikarten zu lernen. Im ABC Tab suchst du Wörter und fügst sie mit Doppelklick zu deinem Stapel hinzu, der Stern zeigt an, dass das Wort schon im Stapel ist. Im Flashcards Tab lernst du die Karten und drückst Plus, wenn du es wusstest, oder Minus, wenn nicht, so lernt die App aus deinen Fehlern und zeigt dir schwierige Wörter öfter. Die Zahl im Textfeld begrenzt, wie viele Karten gezogen werden, und der Knopf mit den Pfeilen erzeugt einen neuen Stapel.

Parolino v{APP_VERSION}
© 2025 {APP_AUTHOR}"""
}


# =============================================================================
# TRANSLATIONS
# =============================================================================
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
    w1 = word.lower()
    w2 = translation.lower()
    
    if len(w1) >= 3 and len(w2) >= 3:
        common_start = 0
        for i in range(min(len(w1), len(w2))):
            if w1[i] == w2[i]:
                common_start += 1
            else:
                break
        if common_start >= 3 and common_start >= min(len(w1), len(w2)) * 0.5:
            return True
    
    common_names = {
        'giovanni', 'giuseppe', 'maria', 'anna', 'paolo', 'pietro', 'francesco',
        'johannes', 'josef', 'paul', 'peter', 'franz', 'marco', 'luca', 'matteo',
        'andrea', 'carlo', 'luigi', 'antonio', 'markus', 'lukas', 'matthias',
        'andreas', 'karl', 'ludwig', 'anton'
    }
    
    return w1 in common_names or w2 in common_names


def add_german_article(word, gender, word_type, original_word=None, translation=None):
    if word_type != 'noun' or not gender:
        return word
    
    if original_word and translation:
        if is_proper_noun(original_word, translation):
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
                    "word_display": word1_display,
                    "word_search": word1_search,
                    "translation": word2_display,
                    "gender": gender,
                    "gender2": gender2,
                    "type": entry_type,
                    "topic": topic
                })
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    
    return dictionary


# =============================================================================
# FEATURE EXTRACTION (41 features)
# =============================================================================
class WordFeatureExtractor:
    TOPICS = ["general", "food", "nature", "health", "sports", "work", "science", "music", "religion"]
    TYPES = ["general", "noun", "verb", "adjective", "adverb"]
    GENDERS = ["m", "f", "n", "pl", None]
    
    @staticmethod
    def extract(word, translation, word_type, topic, gender, gender2, score, lang,
                times_seen=0, days_since_last=0, recent_history=None):
        f = []
        
        # === Continuous features (7) ===
        f.append(min(len(word) / 20.0, 1.0))
        f.append(min(len(translation) / 20.0, 1.0))
        vowels = len(re.findall(r'[aeiouäöüAEIOUÄÖÜ]', word))
        f.append(min(vowels / 8.0, 1.0))
        
        # Temporal features (crucial for spaced repetition)
        f.append(min(times_seen / 20.0, 1.0))
        f.append(min(days_since_last / 30.0, 1.0))  # normalize to ~1 month
        
        # Recent accuracy
        if recent_history and len(recent_history) > 0:
            recent_acc = sum(recent_history) / len(recent_history)
        else:
            recent_acc = 0.5  # neutral
        f.append(recent_acc)
        
        # Current streak (positive = correct streak, negative = wrong streak)
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
        f.append((streak + 10) / 20.0)  # normalize -10..+10 to 0..1
        
        # === Binary features (7) ===
        f.append(1.0 if any(c in word for c in 'äöüÄÖÜ') else 0.0)
        f.append(1.0 if any(c in translation for c in 'äöüÄÖÜ') else 0.0)
        f.append(1.0 if len(word) > 10 else 0.0)
        f.append(1.0 if 'ß' in word or 'ß' in translation else 0.0)
        f.append(1.0 if word and word[0].isupper() else 0.0)
        f.append(1.0 if lang == "italian" else 0.0)
        f.append(1.0 if lang == "german" else 0.0)
        
        # === One-hot: Topic (9) ===
        for t in WordFeatureExtractor.TOPICS:
            f.append(1.0 if topic == t else 0.0)
        
        # === One-hot: Type (5) ===
        for wt in WordFeatureExtractor.TYPES:
            f.append(1.0 if word_type == wt else 0.0)
        
        # === One-hot: Gender (5) ===
        for g in WordFeatureExtractor.GENDERS:
            f.append(1.0 if gender == g else 0.0)
        
        # === One-hot: Gender2 (5) ===
        for g in WordFeatureExtractor.GENDERS:
            f.append(1.0 if gender2 == g else 0.0)
        
        # === Score features (3) ===
        f.append((score + 5) / 10.0)
        f.append(1.0 if score < -2 else 0.0)
        f.append(1.0 if score > 2 else 0.0)
        
        return f
    
    @staticmethod
    def dim():
        return 41


# =============================================================================
# NEURAL NETWORK
# =============================================================================
if TORCH_AVAILABLE:
    class WordPriorityNetwork(nn.Module):
        def __init__(self, input_dim=41):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
            self.training_count = 0
            self.total_loss = 0.0
        
        def forward(self, x):
            return self.network(x)


# =============================================================================
# SMART DECK SELECTOR
# =============================================================================
class SmartDeckSelector:
    BATCH_SIZE = 5  # Mini-batch size
    
    def __init__(self, model_path=None):
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.loss_history = []
        self.prediction_log = []
        
        # Mini-batch accumulator
        self.batch_x = []
        self.batch_y = []
        
        if TORCH_AVAILABLE:
            self.model = WordPriorityNetwork(input_dim=WordFeatureExtractor.dim())
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.MSELoss()
            
            if model_path and os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, weights_only=False)
                    self.model.load_state_dict(checkpoint['model_state'])
                    self.model.training_count = checkpoint.get('training_count', 0)
                    self.model.total_loss = checkpoint.get('total_loss', 0.0)
                    self.loss_history = checkpoint.get('loss_history', [])
                    print(f"[OK] Loaded model: {self.model.training_count} training samples")
                except Exception as e:
                    print(f"[!!] Could not load model: {e}")
    
    def save_model(self, model_path):
        if TORCH_AVAILABLE and self.model:
            try:
                torch.save({
                    'model_state': self.model.state_dict(),
                    'training_count': self.model.training_count,
                    'total_loss': self.model.total_loss,
                    'loss_history': self.loss_history[-500:]
                }, model_path)
            except Exception as e:
                print(f"[!!] Could not save model: {e}")
    
    def _get_temporal_features(self, word_data):
        """Extract temporal features from word data."""
        times_seen = word_data.get('times_seen', 0)
        last_seen = word_data.get('last_seen', 0)
        history = word_data.get('history', [])
        
        if last_seen > 0:
            days_since = (time.time() - last_seen) / 86400.0
        else:
            days_since = 30  # never seen = treat as old
        
        return times_seen, days_since, history[-5:] if history else []
    
    def compute_priority(self, entry, word_data, lang):
        if not TORCH_AVAILABLE or not self.model:
            score = word_data.get('score', 0)
            return (5 - score) / 10.0
        
        score = word_data.get('score', 0)
        times_seen, days_since, recent_history = self._get_temporal_features(word_data)
        
        features = WordFeatureExtractor.extract(
            entry["word_display"], entry["translation"],
            entry["type"], entry["topic"],
            entry.get("gender"), entry.get("gender2"),
            score, lang, times_seen, days_since, recent_history
        )
        
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32)
            priority = self.model(x).item()
        
        # Blend with simple heuristic (helps when model is new)
        score_priority = (5 - score) / 10.0
        spacing_boost = min(days_since / 7.0, 0.3)  # boost if not seen for days
        
        return 0.6 * priority + 0.2 * score_priority + 0.2 * spacing_boost
    
    def train_step(self, entry, word_data, was_correct, lang):
        if not TORCH_AVAILABLE or not self.model:
            return None
        
        score = word_data.get('score', 0)
        times_seen, days_since, recent_history = self._get_temporal_features(word_data)
        
        features = WordFeatureExtractor.extract(
            entry["word_display"], entry["translation"],
            entry["type"], entry["topic"],
            entry.get("gender"), entry.get("gender2"),
            score, lang, times_seen, days_since, recent_history
        )
        
        target = 0.1 if was_correct else 0.9
        
        # Accumulate for mini-batch
        self.batch_x.append(features)
        self.batch_y.append([target])
        
        loss_val = None
        pred_val = None
        
        # Train when batch is full
        if len(self.batch_x) >= self.BATCH_SIZE:
            x = torch.tensor(self.batch_x, dtype=torch.float32)
            y = torch.tensor(self.batch_y, dtype=torch.float32)
            
            self.model.train()
            self.optimizer.zero_grad()
            predictions = self.model(x)
            loss = self.criterion(predictions, y)
            loss.backward()
            self.optimizer.step()
            
            loss_val = loss.item()
            pred_val = predictions[-1].item()  # last prediction
            
            self.model.training_count += len(self.batch_x)
            self.model.total_loss += loss_val * len(self.batch_x)
            self.loss_history.append(loss_val)
            
            # Clear batch
            self.batch_x = []
            self.batch_y = []
        
        self.prediction_log.append({
            'word': entry["word_display"],
            'lang': lang,
            'score': score,
            'was_correct': was_correct,
            'prediction': pred_val,
            'target': target,
            'loss': loss_val
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
            
            priority += random.uniform(0, 0.05)  # small randomness
            priorities.append((word_key, priority))
        
        priorities.sort(key=lambda x: x[1], reverse=True)
        selected = [w for w, _ in priorities[:max_count]]
        random.shuffle(selected)
        return selected
    
    def print_training_log(self):
        if not self.model:
            print("No model available")
            return
        
        print("\n" + "="*70)
        print("NEURAL NETWORK LOG")
        print("="*70)
        
        n = self.model.training_count
        avg_loss = self.model.total_loss / max(1, n)
        print(f"\nSamples: {n}")
        print(f"Avg loss: {avg_loss:.4f}")
        
        if self.loss_history and len(self.loss_history) >= 5:
            recent = self.loss_history[-10:]
            old = self.loss_history[:10]
            print(f"First 10 avg: {sum(old)/len(old):.4f}")
            print(f"Last 10 avg: {sum(recent)/len(recent):.4f}")
        
        if self.prediction_log:
            correct = 0
            recent_logs = self.prediction_log[-50:]
            for log in recent_logs:
                if log['prediction'] is not None:
                    if (not log['was_correct'] and log['prediction'] > 0.5) or \
                       (log['was_correct'] and log['prediction'] < 0.5):
                        correct += 1
            
            if recent_logs:
                acc = correct / len(recent_logs) * 100
                print(f"Accuracy: {acc:.0f}%")
        
        print("="*70 + "\n")


# =============================================================================
# MAIN APPLICATION
# =============================================================================
class Parolino(toga.App):
    def startup(self):
        print("\n" + "="*50)
        print("PAROLINO STARTUP")
        print("="*50)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.deck_save_path = os.path.join(script_dir, 'deck_save.json')
        self.model_save_path = os.path.join(script_dir, 'word_model.pth')
        
        # Load dictionaries
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
        
        # State
        self.deck = set()
        self.shuffled_deck = []
        self.word_data = {}  # Extended data: {word_key: {score, times_seen, last_seen, history}}
        self.max_deck_size = None
        self.current_dict = "italian"
        self.card_direction = "it_de"
        self.current_index = 0
        self.show_front = True
        self.ui_language = "italian"
        
        self.selector = SmartDeckSelector(self.model_save_path)
        
        self._load_deck()
        
        # UI
        main_box = toga.Box(style=Pack(direction=COLUMN))
        
        self.dict_view = self._create_dict_view()
        self.flash_view = self._create_flash_view()
        self.stats_view = self._create_stats_view()
        
        self.tabs = toga.OptionContainer(content=[
            ("ABC", self.dict_view),
            ("Flashcards", self.flash_view),
            ("Nerd", self.stats_view)
        ])
        
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
                    
                    # Load extended word data or migrate from old format
                    if 'word_data' in data:
                        self.word_data = data['word_data']
                    elif 'word_scores' in data:
                        # Migrate from old format
                        for key, score in data['word_scores'].items():
                            self.word_data[key] = {
                                'score': score,
                                'times_seen': abs(score),
                                'last_seen': time.time(),
                                'history': []
                            }
                    
                    print(f"[OK] Loaded deck: {len(self.deck)} words")
        except Exception as e:
            print(f"[!!] Error loading deck: {e}")
        
        if not self.shuffled_deck:
            self._rebuild_deck()
    
    def _save_deck(self):
        try:
            with open(self.deck_save_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'deck': list(self.deck),
                    'shuffled_deck': self.shuffled_deck,
                    'word_data': self.word_data,
                    'max_deck_size': self.max_deck_size,
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
            self.deck, self.entries_it, self.entries_de,
            self.word_data, max_count
        )
        self.current_index = 0
    
    def _t(self, key):
        return TRANSLATIONS[self.ui_language].get(key, key)
    
    def _show_info(self, widget):
        info_text = INFO_TEXT[self.ui_language]
        self.main_window.info_dialog(self._t("info_title"), info_text)
    
    def _get_word_data(self, word_key):
        if word_key not in self.word_data:
            self.word_data[word_key] = {
                'score': 0,
                'times_seen': 0,
                'last_seen': 0,
                'history': []
            }
        return self.word_data[word_key]
    
    # =========================================================================
    # DICTIONARY VIEW
    # =========================================================================
    def _create_dict_view(self):
        container = toga.Box(style=Pack(direction=COLUMN, flex=1))
        
        lang_bar = toga.Box(style=Pack(direction=ROW, margin=5))
        
        self.btn_it = toga.Button(
            "IT", on_press=self._select_italian,
            style=Pack(flex=1, height=40, margin=2)
        )
        self.btn_de = toga.Button(
            "DE", on_press=self._select_german,
            style=Pack(flex=1, height=40, margin=2)
        )
        self.btn_clear = toga.Button(
            "X", on_press=self._clear_deck,
            style=Pack(width=50, height=40, margin=2)
        )
        self.btn_info = toga.Button(
            "?", on_press=self._show_info,
            style=Pack(width=40, height=40, margin=2)
        )
        
        lang_bar.add(self.btn_it)
        lang_bar.add(self.btn_de)
        lang_bar.add(self.btn_clear)
        lang_bar.add(self.btn_info)
        container.add(lang_bar)
        
        self.search_input = toga.TextInput(
            placeholder=self._t("search_placeholder"),
            on_change=self._on_search,
            style=Pack(margin=5, height=35)
        )
        container.add(self.search_input)
        
        self.word_list = toga.Table(
            headings=None,
            accessors=["word", "star"],
            data=[],
            on_activate=self._on_word_click,
            style=Pack(flex=1)
        )
        container.add(self.word_list)
        
        self.info_label = toga.Label(
            self._t("click_info"),
            style=Pack(margin=5, font_size=9)
        )
        container.add(self.info_label)
        
        self._update_word_list()
        return container
    
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
            self._get_word_data(word_key)  # Initialize if needed
            self.shuffled_deck.append(word_key)
        
        self._save_deck()
        self._update_word_list()
        self._update_flash_view()
    
    def _clear_deck(self, widget):
        asyncio.create_task(self._confirm_clear())
    
    async def _confirm_clear(self):
        if not self.deck:
            return
        
        confirmed = await self.main_window.confirm_dialog(
            "Clear", self._t("clear_confirm") + f" ({len(self.deck)})"
        )
        
        if confirmed:
            self.deck.clear()
            self.shuffled_deck.clear()
            self.word_data.clear()
            self._save_deck()
            self._update_word_list()
            self._update_flash_view()
    
    # =========================================================================
    # FLASHCARD VIEW
    # =========================================================================
    def _create_flash_view(self):
        container = toga.Box(style=Pack(direction=COLUMN, flex=1))
        
        top_bar = toga.Box(style=Pack(direction=ROW, margin=5))
        
        self.btn_itde = toga.Button(
            "IT→DE", on_press=self._set_it_de,
            style=Pack(flex=1, height=40, margin=2)
        )
        self.btn_deit = toga.Button(
            "DE→IT", on_press=self._set_de_it,
            style=Pack(flex=1, height=40, margin=2)
        )
        top_bar.add(self.btn_itde)
        top_bar.add(self.btn_deit)
        
        self.max_input = toga.TextInput(
            value=str(self.max_deck_size or ""),
            on_change=self._on_max_change,
            style=Pack(width=50, height=35, margin_left=10)
        )
        top_bar.add(self.max_input)
        
        self.btn_new = toga.Button(
            "⇄", on_press=self._generate_deck,
            style=Pack(width=50, height=40, margin=2)
        )
        top_bar.add(self.btn_new)
        
        container.add(top_bar)
        
        self.counter_label = toga.Label("", style=Pack(margin=10, text_align="center"))
        container.add(self.counter_label)
        
        self.card_label = toga.Label("", style=Pack(margin=20, font_size=24, text_align="center"))
        container.add(self.card_label)
        
        container.add(toga.Box(style=Pack(flex=1)))
        
        nav_box = toga.Box(style=Pack(direction=ROW, margin=5))
        self.btn_prev = toga.Button("<", on_press=self._prev_card, style=Pack(flex=1, height=50, margin=2))
        self.btn_flip = toga.Button("↻", on_press=self._flip_card, style=Pack(flex=1, height=50, margin=2))
        self.btn_next = toga.Button(">", on_press=self._next_card, style=Pack(flex=1, height=50, margin=2))
        nav_box.add(self.btn_prev)
        nav_box.add(self.btn_flip)
        nav_box.add(self.btn_next)
        container.add(nav_box)
        
        feedback_box = toga.Box(style=Pack(direction=ROW, margin=5))
        self.btn_wrong = toga.Button("-", on_press=self._mark_wrong, style=Pack(flex=1, height=50, margin=2))
        self.btn_correct = toga.Button("+", on_press=self._mark_correct, style=Pack(flex=1, height=50, margin=2))
        feedback_box.add(self.btn_wrong)
        feedback_box.add(self.btn_correct)
        container.add(feedback_box)
        
        return container
    
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
                        text = add_german_article(text, entry["gender2"], "noun",
                                                  entry["word_display"], entry["translation"])
                else:
                    text = entry["word_display"]
                    if entry["type"] == "noun" and entry.get("gender"):
                        text = add_german_article(text, entry["gender"], "noun",
                                                  entry["translation"], entry["word_display"])
                self.card_label.text = text
        else:
            if self.show_front:
                if is_italian:
                    text = entry["translation"]
                    if entry["type"] == "noun" and entry.get("gender2"):
                        text = add_german_article(text, entry["gender2"], "noun",
                                                  entry["word_display"], entry["translation"])
                else:
                    text = entry["word_display"]
                    if entry["type"] == "noun" and entry.get("gender"):
                        text = add_german_article(text, entry["gender"], "noun",
                                                  entry["translation"], entry["word_display"])
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
        
        # Update word data
        old_score = word_data['score']
        word_data['score'] = min(5, old_score + 1)
        word_data['times_seen'] = word_data.get('times_seen', 0) + 1
        word_data['last_seen'] = time.time()
        word_data.setdefault('history', []).append(True)
        word_data['history'] = word_data['history'][-10:]  # Keep last 10
        
        result = self.selector.train_step(entry, word_data, was_correct=True, lang=lang)
        if result and result[0] is not None:
            print(f"[+] {entry['word_display'][:12]:<12} batch trained, loss:{result[0]:.4f}")
        
        self._save_deck()
        self._next_card(widget)
    
    def _mark_wrong(self, widget):
        word_key, entry, lang, word_data = self._get_current_entry()
        if not entry:
            return
        
        # Update word data
        old_score = word_data['score']
        word_data['score'] = max(-5, old_score - 1)
        word_data['times_seen'] = word_data.get('times_seen', 0) + 1
        word_data['last_seen'] = time.time()
        word_data.setdefault('history', []).append(False)
        word_data['history'] = word_data['history'][-10:]
        
        result = self.selector.train_step(entry, word_data, was_correct=False, lang=lang)
        if result and result[0] is not None:
            print(f"[-] {entry['word_display'][:12]:<12} batch trained, loss:{result[0]:.4f}")
        
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
            self.deck, self.entries_it, self.entries_de,
            self.word_data, max_size
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
        container = toga.Box(style=Pack(direction=COLUMN, flex=1, margin=10))
        
        self.stats_text = toga.MultilineTextInput(readonly=True, style=Pack(flex=1, margin=5))
        container.add(self.stats_text)
        
        btn_refresh = toga.Button("Refresh", on_press=self._refresh_stats, style=Pack(height=40, margin=5))
        container.add(btn_refresh)
        
        btn_priorities = toga.Button("Priorities", on_press=self._show_priorities, style=Pack(height=40, margin=5))
        container.add(btn_priorities)
        
        btn_log = toga.Button("Full Log", on_press=self._print_log, style=Pack(height=40, margin=5))
        container.add(btn_log)
        
        return container
    
    def _update_stats_view(self):
        if not hasattr(self, 'stats_text'):
            return
        
        lines = []
        
        if TORCH_AVAILABLE and self.selector.model:
            n = self.selector.model.training_count
            avg = self.selector.model.total_loss / max(1, n)
            
            lines.append(f"Samples: {n}")
            lines.append(f"Loss: {avg:.3f}")
            
            if self.selector.loss_history and len(self.selector.loss_history) >= 3:
                recent = self.selector.loss_history[-5:]
                lines.append(f"Recent: {sum(recent)/len(recent):.3f}")
            
            lines.append("")
            
            if n < 20:
                lines.append(f"Status: warming up")
            elif n < 50:
                lines.append(f"Status: learning")
            else:
                lines.append(f"Status: trained")
            
            if self.selector.prediction_log:
                correct = 0
                recent_logs = [l for l in self.selector.prediction_log[-30:] if l['prediction'] is not None]
                for log in recent_logs:
                    if (not log['was_correct'] and log['prediction'] > 0.5) or \
                       (log['was_correct'] and log['prediction'] < 0.5):
                        correct += 1
                if recent_logs:
                    acc = correct / len(recent_logs) * 100
                    lines.append(f"Accuracy: {acc:.0f}%")
            
            lines.append("")
            lines.append("Scores:")
            
            counts = {}
            for wd in self.word_data.values():
                score = wd.get('score', 0)
                counts[score] = counts.get(score, 0) + 1
            
            for score in range(-5, 6):
                count = counts.get(score, 0)
                if count > 0:
                    lines.append(f"  {score:+d}: {count}")
        else:
            lines.append("PyTorch not available")
        
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