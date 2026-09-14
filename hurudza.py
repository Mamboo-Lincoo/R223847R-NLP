# ============================================================================
# HURUDZA NZWISISO - COMPLETE NLP SYSTEM FOR AGRICULTURAL ADVISORY
# Production-ready version with save/load, API integration, and multiple NLP tasks
# Fixed: robust training with rare-class filtering and adaptive stratification
# ============================================================================

import os
import json
import re
import pickle
import joblib
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.filterwarnings('ignore')

# NLTK imports
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
for resource in ['tokenizers/punkt', 'corpora/stopwords']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1])


# ============================================================================
# PART 1: DATA PROCESSING
# ============================================================================

class HurudzaDataProcessor:
    """Process Hurudza/ICAR JSON data: load, clean, and extract text sections"""

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.cleaned_data = None
        self.sections = []

    def is_number(self, token):
        try:
            float(token)
            return True
        except ValueError:
            return False

    def clean_text(self, text):
        if not isinstance(text, str):
            return text
        text = text.lower()
        text = re.sub(r'\bpage\s+\d+\b', '', text)
        text = re.sub(r'[^a-z0-9\s\.]', '', text)
        text = re.sub(r'\.{2,}', '', text)
        tokens = text.split()
        meaningful = [t for t in tokens if t not in self.stop_words and not self.is_number(t)]
        return ' '.join(meaningful)

    def process_item(self, item):
        if isinstance(item, str):
            return self.clean_text(item)
        elif isinstance(item, list):
            return [self.process_item(x) for x in item]
        elif isinstance(item, dict):
            return {k: self.process_item(v) for k, v in item.items()}
        return item

    def load_and_clean(self, input_file):
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: {input_file} not found.")
            return None
        self.cleaned_data = self.process_item(data)
        return self.cleaned_data

    def extract_text_sections(self, data=None, min_length=50):
        if data is None:
            data = self.cleaned_data
        if data is None:
            return []

        sections = []

        def recursive_extract(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    recursive_extract(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    recursive_extract(item, f"{path}[{idx}]")
            elif isinstance(obj, str) and len(obj) >= min_length:
                sections.append({'path': path, 'text': obj, 'length': len(obj)})

        recursive_extract(data)
        self.sections = sections
        return sections


# ============================================================================
# PART 2: AGRICULTURAL TAXONOMY
# ============================================================================

class AgriculturalTaxonomy:
    """Hierarchical taxonomy for agricultural issues"""

    def __init__(self):
        self.taxonomy = {
            'crop_health': {
                'diseases': ['blast', 'blight', 'rust', 'mildew', 'rot', 'smut',
                             'anthracnose', 'bacterial blight', 'bacterial spot',
                             'soft rot', 'mosaic', 'yellow vein mosaic', 'leaf curl',
                             'tungro', 'root knot', 'cyst nematode', 'lesion nematode'],
                'pests': ['stem borer', 'shoot borer', 'pod borer', 'fruit borer',
                          'aphids', 'jassids', 'whitefly', 'thrips', 'mites',
                          'caterpillars', 'beetles', 'semiloopers', 'leaf rollers'],
                'physiological': ['nitrogen deficiency', 'zinc deficiency', 'iron deficiency',
                                  'drought', 'waterlogging', 'flooding', 'salinity',
                                  'heat stress', 'cold stress', 'frost damage']
            },
            'crop_production': {
                'cultivation': ['sowing', 'transplanting', 'spacing', 'planting method'],
                'nutrient_management': ['fertilizer application', 'organic manure',
                                        'biofertilizers', 'micronutrients'],
                'water_management': ['irrigation methods', 'drainage', 'water conservation'],
                'weed_management': ['chemical control', 'mechanical control', 'biological control']
            },
            'post_harvest': {
                'handling': ['harvesting', 'threshing', 'drying', 'cleaning'],
                'storage': ['storage structures', 'pest control', 'moisture control'],
                'processing': ['milling', 'value addition', 'packaging', 'quality assessment']
            },
            'soil_health': {
                'physical': ['soil structure', 'compaction', 'erosion', 'water holding'],
                'chemical': ['ph', 'salinity', 'organic carbon', 'nutrient availability'],
                'biological': ['microbial diversity', 'earthworms', 'organic matter decomposition']
            }
        }

        self.flattened = {}
        self._flatten_taxonomy()

    def _flatten_taxonomy(self):
        for level1, level2_dict in self.taxonomy.items():
            for level2, terms in level2_dict.items():
                for term in terms:
                    self.flattened[term] = {
                        'level1': level1,
                        'level2': level2,
                        'term': term
                    }

    def classify_issue(self, text):
        text_lower = text.lower()
        classifications = []
        for term, hierarchy in self.flattened.items():
            if term in text_lower:
                classifications.append(hierarchy.copy())
        return classifications


# ============================================================================
# PART 3: CHARACTER N-GRAM VECTORIZER
# ============================================================================

class CharacterNGramVectorizer(BaseEstimator, TransformerMixin):
    """Character n-gram vectorizer compatible with sklearn pipelines"""

    def __init__(self, ngram_range=(2, 5), max_features=10000):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vocabulary_ = {}

    def fit(self, X, y=None):
        all_ngrams = defaultdict(int)
        for text in X:
            text_lower = text.lower()
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                for i in range(len(text_lower) - n + 1):
                    all_ngrams[text_lower[i:i+n]] += 1
        sorted_ngrams = sorted(all_ngrams.items(), key=lambda x: x[1], reverse=True)
        self.vocabulary_ = {ng: idx for idx, (ng, _) in enumerate(sorted_ngrams[:self.max_features])}
        return self

    def transform(self, X):
        X_t = np.zeros((len(X), len(self.vocabulary_)))
        for i, text in enumerate(X):
            text_lower = text.lower()
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                for j in range(len(text_lower) - n + 1):
                    ng = text_lower[j:j+n]
                    if ng in self.vocabulary_:
                        X_t[i, self.vocabulary_[ng]] += 1
        return X_t

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


# ============================================================================
# PART 4: FEATURE EXTRACTOR
# ============================================================================

class AdvancedFeatureExtractor:
    """Extract domain-specific features from agricultural text"""

    def __init__(self, taxonomy):
        self.taxonomy = taxonomy
        self.crop_patterns = {
            'cereals': ['rice', 'wheat', 'maize', 'sorghum', 'barley', 'pearl millet'],
            'pulses': ['chickpea', 'pigeonpea', 'mungbean', 'urdbean', 'lentil', 'cowpea'],
            'oilseeds': ['groundnut', 'soybean', 'sunflower', 'sesame', 'castor', 'linseed', 'mustard'],
            'commercial': ['cotton', 'sugarcane', 'jute', 'tobacco'],
            'horticulture': ['mango', 'banana', 'citrus', 'grape', 'pomegranate', 'guava'],
            'vegetables': ['tomato', 'potato', 'onion', 'garlic', 'chilli', 'brinjal', 'okra'],
            'spices': ['turmeric', 'ginger', 'black pepper', 'cardamom', 'coriander']
        }
        self.chemical_patterns = {
            'fungicides': ['mancozeb', 'carbendazim', 'propiconazole', 'chlorothalonil'],
            'insecticides': ['cypermethrin', 'malathion', 'monocrotophos', 'imidacloprid', 'acephate'],
            'herbicides': ['glyphosate', 'atrazine', 'pendimethalin', 'paraquat'],
            'fertilizers': ['urea', 'dap', 'potash', 'superphosphate'],
            'bio_agents': ['trichoderma', 'pseudomonas', 'bacillus', 'beauveria', 'neem']
        }

    def extract_features(self, text):
        text_lower = text.lower()
        features = {}
        for cat, crops in self.crop_patterns.items():
            found = [c for c in crops if c in text_lower]
            if found:
                features[f'crop_{cat}'] = len(found)
        for cat, chems in self.chemical_patterns.items():
            found = [c for c in chems if c in text_lower]
            if found:
                features[f'chem_{cat}'] = len(found)
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        return features


# ============================================================================
# PART 5: THE MAIN MODEL - TRAINED, SAVED, AND LOADED
# ============================================================================

class HurudzaModel:
    """
    The core trained model that can be saved, loaded, and used for inference.
    """

    VERSION = "1.0.0"

    def __init__(self, taxonomy=None):
        self.taxonomy = taxonomy or AgriculturalTaxonomy()
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.categories = []
        self.training_metrics = {}
        self.trained_at = None
        self.is_trained = False

    def train(self, texts, labels):
        """Train the model (robust to small classes)"""
        print(f"Training HurudzaModel on {len(texts)} examples...")

        # Encode labels
        self.label_encoder = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.categories = list(self.label_encoder.keys())
        y_encoded = np.array([self.label_encoder[l] for l in labels])

        # ---- Decide whether stratification is safe ----
        label_counts = Counter(y_encoded)
        min_count = min(label_counts.values())
        use_stratify = min_count >= 2

        if not use_stratify:
            print(f"⚠️  Some classes have <2 samples; using non-stratified split.")
        else:
            print(f"✓  Using stratified split (min class count = {min_count}).")

        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded if use_stratify else None,
        )

        # Build pipeline: TF-IDF + Logistic Regression
        self.vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 3), stop_words='english', min_df=1
        )
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        self.classifier = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        self.classifier.fit(X_train_tfidf, y_train)

        # Evaluate
        y_pred = self.classifier.predict(X_test_tfidf)
        present_labels = sorted(set(y_test) | set(y_pred))
        target_names = [self.categories[i] for i in present_labels]

        self.training_metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
            'num_train': len(X_train),
            'num_test': len(X_test),
            'num_categories': len(self.categories),
            'categories': self.categories,
            'classification_report': classification_report(
                y_test, y_pred,
                labels=present_labels,
                target_names=target_names,
                output_dict=True, zero_division=0
            )
        }

        self.trained_at = datetime.utcnow().isoformat()
        self.is_trained = True

        print(f"Model trained. Accuracy: {self.training_metrics['accuracy']:.4f}")
        return self.training_metrics

    def predict(self, text):
        """Predict category for a single text"""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() or load() first.")

        cleaned = self._clean_for_inference(text)
        vec = self.vectorizer.transform([cleaned])
        pred_idx = int(self.classifier.predict(vec)[0])
        proba = self.classifier.predict_proba(vec)[0]

        category = self.categories[pred_idx]
        confidence = float(proba[pred_idx])

        top_idx = np.argsort(proba)[::-1][:3]
        top_predictions = [
            {'category': self.categories[i], 'confidence': float(proba[i])}
            for i in top_idx
        ]

        return {
            'category': category,
            'confidence': confidence,
            'top_predictions': top_predictions
        }

    def _clean_for_inference(self, text):
        text = str(text).lower()
        text = re.sub(r'\bpage\s+\d+\b', '', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def save(self, path='hurudza_model.pkl'):
        """Save trained model to disk"""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model.")
        state = {
            'version': self.VERSION,
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'categories': self.categories,
            'training_metrics': self.training_metrics,
            'trained_at': self.trained_at,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, path='hurudza_model.pkl'):
        """Load trained model from disk"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        model = cls()
        model.vectorizer = state['vectorizer']
        model.classifier = state['classifier']
        model.label_encoder = state['label_encoder']
        model.categories = state['categories']
        model.training_metrics = state['training_metrics']
        model.trained_at = state['trained_at']
        model.is_trained = True
        print(f"Model loaded from {path} (trained at {model.trained_at})")
        return model


# ============================================================================
# PART 6: MULTI-TASK NLP ENGINE
# ============================================================================

class HurudzaNLSEngine:
    """
    Higher-level NLP engine that combines classification with:
    - Information extraction (pests, diseases, chemicals, crops)
    - Keyword extraction
    - Sentiment analysis (heuristic for urgency)
    - Summarization (extractive)
    - Question answering (rule-based, template-driven)
    """

    def __init__(self, model: HurudzaModel):
        self.model = model
        self.taxonomy = model.taxonomy
        self.feature_extractor = AdvancedFeatureExtractor(self.taxonomy)
        self.language_keywords = {
            'hi': ['और', 'की', 'से', 'में', 'है', 'का', 'या'],
            'ta': ['ஒரு', 'இந்த', 'இது', 'அதில்', 'என'],
            'te': ['ఒక', 'ఈ', 'ఇది', 'ఆ', 'వారు'],
            'ml': ['ഒരു', 'ഈ', 'ഇത്', 'അത്'],
            'kn': ['ಒಂದು', 'ಈ', 'ಇದು', 'ಆ'],
            'en': ['the', 'and', 'for', 'with', 'this', 'that']
        }

    # ---------------- Language detection ----------------
    def detect_language(self, text):
        text_lower = str(text).lower()
        scores = {lang: sum(1 for kw in kws if kw in text_lower)
                  for lang, kws in self.language_keywords.items()}
        if max(scores.values()) == 0:
            return 'en'
        return max(scores, key=scores.get)

    # ---------------- Information extraction ----------------
    def extract_entities(self, text):
        text_lower = text.lower()
        entities = {
            'crops': [],
            'chemicals': [],
            'pests': [],
            'diseases': [],
            'symptoms': []
        }

        for cat, crops in self.feature_extractor.crop_patterns.items():
            for c in crops:
                if c in text_lower:
                    entities['crops'].append({'name': c, 'category': cat})

        for cat, chems in self.feature_extractor.chemical_patterns.items():
            for c in chems:
                if c in text_lower:
                    entities['chemicals'].append({'name': c, 'category': cat})

        for pest in self.taxonomy.taxonomy['crop_health']['pests']:
            if pest in text_lower:
                entities['pests'].append(pest)

        for disease in self.taxonomy.taxonomy['crop_health']['diseases']:
            if disease in text_lower:
                entities['diseases'].append(disease)

        symptom_terms = ['spot', 'blight', 'rot', 'wilt', 'yellow', 'brown',
                         'stunted', 'curl', 'mildew', 'rust', 'lesion']
        for s in symptom_terms:
            if s in text_lower:
                entities['symptoms'].append(s)

        return entities

    # ---------------- Sentiment / urgency ----------------
    def analyze_urgency(self, text):
        text_lower = text.lower()
        urgent_words = ['urgent', 'emergency', 'dying', 'severe', 'spreading',
                        'worst', 'lost', 'destroy', 'kill', 'all my', 'entire']
        negative_words = ['problem', 'disease', 'pest', 'dying', 'wilting',
                          'yellowing', 'rot', 'damage', 'loss', 'fail']
        positive_words = ['healthy', 'good', 'fine', 'okay', 'improving',
                          'recovered', 'thriving', 'strong']

        urgent_score = sum(1 for w in urgent_words if w in text_lower)
        neg_score = sum(1 for w in negative_words if w in text_lower)
        pos_score = sum(1 for w in positive_words if w in text_lower)

        if urgent_score > 0:
            level = 'critical'
            sentiment = 'negative'
        elif neg_score > pos_score:
            level = 'moderate'
            sentiment = 'negative'
        elif pos_score > neg_score:
            level = 'low'
            sentiment = 'positive'
        else:
            level = 'low'
            sentiment = 'neutral'

        return {
            'urgency': level,
            'sentiment': sentiment,
            'signals': {
                'urgent_words': urgent_score,
                'negative_words': neg_score,
                'positive_words': pos_score
            }
        }

    # ---------------- Extractive summarization ----------------
    def summarize(self, text, max_sentences=3):
        sentences = re.split(r'(?<=[.!?])\s+', str(text).strip())
        if len(sentences) <= max_sentences:
            return ' '.join(sentences)

        words = re.findall(r'\b\w+\b', text.lower())
        stop_words = set(stopwords.words('english'))
        word_freq = Counter(w for w in words if w not in stop_words and len(w) > 3)

        if not word_freq:
            return ' '.join(sentences[:max_sentences])

        max_freq = max(word_freq.values())
        scored = []
        for sent in sentences:
            sent_words = re.findall(r'\b\w+\b', sent.lower())
            score = sum(word_freq.get(w, 0) for w in sent_words) / max_freq
            scored.append((score, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [s for _, s in scored[:max_sentences]]
        ordered = [s for s in sentences if s in top]
        return ' '.join(ordered)

    # ---------------- Question answering ----------------
    def answer_question(self, question, context=None):
        if context:
            return self._extractive_qa(question, context)

        pred = self.model.predict(question)
        entities = self.extract_entities(question)
        urgency = self.analyze_urgency(question)

        return {
            'answer': self._build_advisory(pred, entities, urgency),
            'prediction': pred,
            'entities': entities,
            'urgency': urgency,
            'mode': 'generative_advisory'
        }

    def _extractive_qa(self, question, context):
        q_words = set(re.findall(r'\b\w+\b', question.lower()))
        sentences = re.split(r'(?<=[.!?])\s+', context)

        best_sentence = ''
        best_score = 0
        for sent in sentences:
            s_words = set(re.findall(r'\b\w+\b', sent.lower()))
            overlap = len(q_words & s_words)
            if overlap > best_score:
                best_score = overlap
                best_sentence = sent

        return {
            'answer': best_sentence or "No relevant information found in the provided context.",
            'score': best_score,
            'mode': 'extractive_qa'
        }

    def _build_advisory(self, prediction, entities, urgency):
        parts = []
        category = prediction['category']

        parts.append(f"Category: {category} (confidence: {prediction['confidence']:.2%})")

        if entities['crops']:
            crops = ', '.join(e['name'] for e in entities['crops'][:3])
            parts.append(f"Crop(s) mentioned: {crops}")

        if entities['diseases']:
            parts.append(f"Possible diseases: {', '.join(entities['diseases'][:3])}")
        if entities['pests']:
            parts.append(f"Possible pests: {', '.join(entities['pests'][:3])}")
        if entities['symptoms']:
            parts.append(f"Symptoms: {', '.join(entities['symptoms'][:5])}")

        parts.append(f"Urgency level: {urgency['urgency']}")

        recs = []
        if category == 'disease_management':
            recs = [
                "Isolate affected plants to prevent spread",
                "Apply recommended fungicide based on diagnosis",
                "Remove and destroy severely infected plant parts",
                "Ensure proper spacing and air circulation"
            ]
        elif category == 'pest_management':
            recs = [
                "Scout fields early morning and evening",
                "Use pheromone traps for monitoring",
                "Apply IPM: biological control first, then chemicals",
                "Rotate crops to break pest cycles"
            ]
        elif category == 'soil_management':
            recs = [
                "Conduct soil test to confirm deficiency",
                "Apply balanced fertilizers based on soil test",
                "Add organic matter to improve soil health"
            ]
        elif category == 'water_management':
            recs = [
                "Check irrigation scheduling",
                "Improve drainage if waterlogged",
                "Use mulching to conserve moisture"
            ]
        else:
            recs = [
                "Consult your local agricultural extension officer",
                "Contact nearest KVK (Krishi Vigyan Kendra) for advice"
            ]

        parts.append("Recommendations:")
        for i, r in enumerate(recs, 1):
            parts.append(f"  {i}. {r}")

        return '\n'.join(parts)

    # ---------------- Unified analyze ----------------
    def analyze(self, text):
        if not self.model.is_trained:
            return {'error': 'Model not trained'}

        prediction = self.model.predict(text)
        entities = self.extract_entities(text)
        urgency = self.analyze_urgency(text)
        summary = self.summarize(text) if len(text) > 200 else text

        return {
            'input': text,
            'language': self.detect_language(text),
            'classification': prediction,
            'entities': entities,
            'urgency': urgency,
            'summary': summary,
            'advisory': self._build_advisory(prediction, entities, urgency),
            'timestamp': datetime.utcnow().isoformat()
        }


# ============================================================================
# PART 7: TRAINING PIPELINE (run this once to create the model)
# ============================================================================

def train_model_from_json(json_path='ICAR_Text_Extracted.json',
                          model_output='hurudza_model.pkl',
                          min_samples_per_class=5):
    """Full pipeline: load JSON, extract text, label, train, save model"""

    print("=" * 70)
    print("HURUDZA NZWISISO - MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load and clean
    processor = HurudzaDataProcessor()
    processor.load_and_clean(json_path)
    sections = processor.extract_text_sections()
    print(f"\n[1] Extracted {len(sections)} text sections from {json_path}")

    if not sections:
        raise RuntimeError("No text sections found in the JSON file.")

    # 2. Label sections using keyword heuristics
    label_keywords = {
        'crop_production': ['cultivation', 'sowing', 'transplanting', 'fertilizer',
                            'irrigation', 'harvest', 'yield', 'sowing'],
        'crop_improvement': ['breeding', 'variety', 'hybrid', 'resistant',
                             'improvement', 'selection', 'genetic', 'cultivar',
                             'germplasm', 'yield potential', 'released', 'notified'],
        'pest_management': ['pest', 'insect', 'borer', 'hopper', 'worm',
                            'beetle', 'aphid', 'thrips', 'weevil'],
        'disease_management': ['disease', 'blast', 'blight', 'rust', 'mildew',
                               'rot', 'wilt', 'pathogen', 'fungus', 'bacterial'],
        'soil_management': ['soil', 'nutrient', 'organic', 'carbon',
                            'nitrogen', 'phosphorus', 'potassium', 'fertility'],
        'water_management': ['water', 'irrigation', 'drainage', 'rainfall',
                             'drought', 'saline', 'moisture']
    }

    texts, labels = [], []
    for section in sections:
        text = section['text'].lower()
        scores = {label: sum(1 for kw in kws if kw in text)
                  for label, kws in label_keywords.items()}
        best_label = max(scores, key=scores.get)
        if scores[best_label] > 0:
            texts.append(section['text'])
            labels.append(best_label)

    print(f"\n[2] Labeled {len(texts)} examples before filtering")
    print(f"    Raw distribution: {dict(Counter(labels))}")

    # ---- Filter out rare classes (needed for stratified split) ----
    label_counts = Counter(labels)
    rare_labels = {lbl for lbl, cnt in label_counts.items()
                   if cnt < min_samples_per_class}

    if rare_labels:
        print(f"\n[2b] Dropping rare categories (< {min_samples_per_class} samples):")
        for lbl in rare_labels:
            print(f"     - {lbl}: {label_counts[lbl]} sample(s)")

        filtered_texts, filtered_labels = [], []
        for t, l in zip(texts, labels):
            if l not in rare_labels:
                filtered_texts.append(t)
                filtered_labels.append(l)
        texts, labels = filtered_texts, filtered_labels

    print(f"\n[2c] Final dataset: {len(texts)} examples")
    print(f"     Distribution: {dict(Counter(labels))}")

    if len(set(labels)) < 2:
        raise RuntimeError(
            "Need at least 2 distinct categories with enough samples to train."
        )

    if len(texts) < 10:
        raise RuntimeError("Too few labeled examples to train a reliable model.")

    # 3. Train
    print("\n[3] Training model...")
    model = HurudzaModel()
    metrics = model.train(texts, labels)

    # 4. Save
    print(f"\n[4] Saving model to {model_output}")
    model.save(model_output)

    print("\n[5] Training Summary:")
    print(f"    Accuracy:      {metrics['accuracy']:.4f}")
    print(f"    F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"    F1 (macro):    {metrics['f1_macro']:.4f}")
    print(f"    Categories:    {metrics['categories']}")
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE — model saved and ready for API integration")
    print("=" * 70)

    return model, metrics


if __name__ == "__main__":
    train_model_from_json()