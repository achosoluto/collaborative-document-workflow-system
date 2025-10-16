"""
Advanced Configurable Summarization Engine
Supports multiple summarization approaches with configurable length and detail levels
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json

# AI and NLP libraries
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer

# Custom imports
from .content_extractor import ContentProcessingPipeline

logger = logging.getLogger(__name__)


class SummarizationMethod(Enum):
    """Available summarization methods"""
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    HYBRID = "hybrid"
    BULLET_POINTS = "bullet_points"
    KEY_SENTENCES = "key_sentences"
    PARAGRAPH = "paragraph"


class DetailLevel(Enum):
    """Summarization detail levels"""
    BRIEF = "brief"  # 10-15% of original
    CONCISE = "concise"  # 15-25% of original
    DETAILED = "detailed"  # 25-40% of original
    COMPREHENSIVE = "comprehensive"  # 40-60% of original


@dataclass
class SummarizationConfig:
    """Configuration for summarization"""
    method: SummarizationMethod = SummarizationMethod.HYBRID
    detail_level: DetailLevel = DetailLevel.CONCISE
    target_length: Optional[int] = None  # Characters
    target_sentences: Optional[int] = None
    language: str = "english"
    preserve_formatting: bool = True
    include_key_phrases: bool = True
    focus_areas: List[str] = None  # Areas to focus on (e.g., ["processes", "requirements"])

    def __post_init__(self):
        if self.focus_areas is None:
            self.focus_areas = []


class SummarizationResult:
    """Result of summarization operation"""

    def __init__(self, config: SummarizationConfig):
        self.config = config
        self.summary = ""
        self.method_used = ""
        self.original_length = 0
        self.summary_length = 0
        self.compression_ratio = 0.0
        self.key_phrases = []
        self.important_sentences = []
        self.metadata = {}
        self.processing_time = 0.0
        self.success = False
        self.error_message = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            'summary': self.summary,
            'method_used': self.method_used,
            'config': {
                'method': self.config.method.value,
                'detail_level': self.config.detail_level.value,
                'target_length': self.config.target_length,
                'target_sentences': self.config.target_sentences,
                'language': self.config.language
            },
            'statistics': {
                'original_length': self.original_length,
                'summary_length': self.summary_length,
                'compression_ratio': self.compression_ratio,
                'processing_time': self.processing_time
            },
            'key_phrases': self.key_phrases,
            'important_sentences': self.important_sentences,
            'metadata': self.metadata,
            'success': self.success,
            'error_message': self.error_message
        }


class BaseSummarizer:
    """Base class for all summarizers"""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.tokenizer = None

    def summarize(self, text: str, config: SummarizationConfig) -> SummarizationResult:
        """Summarize text according to config"""
        raise NotImplementedError

    def _calculate_compression_ratio(self, original: str, summary: str) -> float:
        """Calculate compression ratio"""
        if not original:
            return 0.0
        return len(summary) / len(original)

    def _extract_key_phrases(self, text: str, top_n: int = 10) -> List[str]:
        """Extract key phrases from text"""
        # Simple key phrase extraction based on word frequency and position
        sentences = text.split('.')
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter meaningful words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        meaningful_words = [w for w in words if len(w) > 3 and w not in stop_words]

        # Count frequency
        from collections import Counter
        word_freq = Counter(meaningful_words)

        # Return top phrases
        return [word for word, _ in word_freq.most_common(top_n)]


class ExtractiveSummarizer(BaseSummarizer):
    """Extractive summarization using statistical methods"""

    def __init__(self):
        super().__init__("extractive")
        self.summarizer = None

    def summarize(self, text: str, config: SummarizationConfig) -> SummarizationResult:
        """Perform extractive summarization"""
        result = SummarizationResult(config)
        result.method_used = "extractive"

        try:
            # Initialize sumy summarizer
            parser = PlaintextParser.from_string(text, Tokenizer(config.language))

            # Choose summarizer based on text characteristics
            if len(text.split()) > 1000:
                self.summarizer = LexRankSummarizer()
            else:
                self.summarizer = TextRankSummarizer()

            # Calculate target sentence count
            sentences = text.split('.')
            total_sentences = len([s for s in sentences if s.strip()])

            if config.target_sentences:
                sentence_count = min(config.target_sentences, total_sentences)
            else:
                # Calculate based on detail level
                ratios = {
                    DetailLevel.BRIEF: 0.15,
                    DetailLevel.CONCISE: 0.25,
                    DetailLevel.DETAILED: 0.35,
                    DetailLevel.COMPREHENSIVE: 0.50
                }
                target_ratio = ratios.get(config.detail_level, 0.25)
                sentence_count = max(1, int(total_sentences * target_ratio))

            # Generate summary
            summary_sentences = self.summarizer(parser.document, sentence_count)
            summary = '. '.join([str(sentence) for sentence in summary_sentences])

            # Update result
            result.summary = summary
            result.original_length = len(text)
            result.summary_length = len(summary)
            result.compression_ratio = self._calculate_compression_ratio(text, summary)
            result.important_sentences = [str(s) for s in summary_sentences]

            if config.include_key_phrases:
                result.key_phrases = self._extract_key_phrases(text)

            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"Extractive summarization error: {e}")

        return result


class AbstractiveSummarizer(BaseSummarizer):
    """Abstractive summarization using transformer models"""

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        super().__init__("abstractive")
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Load the transformer model"""
        try:
            # Use smaller model for CPU environments
            if torch.cuda.is_available():
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            else:
                # Use smaller, CPU-friendly model
                small_model = "facebook/bart-large-cnn"
                self.model = AutoModelForSeq2SeqLM.from_pretrained(small_model)
                self.tokenizer = AutoTokenizer.from_pretrained(small_model)

            self.device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Loaded abstractive summarization model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load abstractive model: {e}")
            # Fallback to extractive
            self.model = None
            self.tokenizer = None

    def summarize(self, text: str, config: SummarizationConfig) -> SummarizationResult:
        """Perform abstractive summarization"""
        result = SummarizationResult(config)
        result.method_used = "abstractive"

        if not self.model or not self.tokenizer:
            # Fallback to extractive
            fallback = ExtractiveSummarizer()
            return fallback.summarize(text, config)

        try:
            # Calculate target length
            original_length = len(text)

            if config.target_length:
                max_length = min(config.target_length, original_length)
                min_length = max_length // 2
            else:
                # Calculate based on detail level
                ratios = {
                    DetailLevel.BRIEF: 0.15,
                    DetailLevel.CONCISE: 0.25,
                    DetailLevel.DETAILED: 0.35,
                    DetailLevel.COMPREHENSIVE: 0.50
                }
                target_ratio = ratios.get(config.detail_level, 0.25)
                max_length = int(original_length * target_ratio)
                min_length = max_length // 2

            # Truncate input if too long (BART limitation)
            max_input_length = 1024
            if len(text) > max_input_length:
                text = text[:max_input_length]

            # Generate summary
            inputs = self.tokenizer(text, return_tensors="pt", max_length=max_input_length, truncation=True)

            # Generate summary
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Update result
            result.summary = summary
            result.original_length = original_length
            result.summary_length = len(summary)
            result.compression_ratio = self._calculate_compression_ratio(text, summary)

            if config.include_key_phrases:
                result.key_phrases = self._extract_key_phrases(text)

            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"Abstractive summarization error: {e}")

        return result


class HybridSummarizer(BaseSummarizer):
    """Hybrid summarization combining extractive and abstractive methods"""

    def __init__(self):
        super().__init__("hybrid")
        self.extractive = ExtractiveSummarizer()
        self.abstractive = AbstractiveSummarizer()

    def summarize(self, text: str, config: SummarizationConfig) -> SummarizationResult:
        """Perform hybrid summarization"""
        result = SummarizationResult(config)
        result.method_used = "hybrid"

        try:
            # Step 1: Extractive summarization for key sentences
            extractive_result = self.extractive.summarize(text, config)

            if not extractive_result.success:
                return extractive_result

            # Step 2: Use extractive summary as input for abstractive refinement
            extractive_summary = extractive_result.summary

            # Create refined config for abstractive step
            refined_config = SummarizationConfig(
                method=SummarizationMethod.ABSTRACTIVE,
                detail_level=config.detail_level,
                target_length=config.target_length,
                language=config.language
            )

            abstractive_result = self.abstractive.summarize(extractive_summary, refined_config)

            # Combine results
            result.summary = abstractive_result.summary
            result.original_length = extractive_result.original_length
            result.summary_length = len(result.summary)
            result.compression_ratio = self._calculate_compression_ratio(text, result.summary)
            result.key_phrases = list(set(extractive_result.key_phrases + abstractive_result.key_phrases))
            result.important_sentences = extractive_result.important_sentences
            result.metadata = {
                'extractive_ratio': extractive_result.compression_ratio,
                'abstractive_ratio': abstractive_result.compression_ratio
            }

            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"Hybrid summarization error: {e}")

        return result


class BulletPointSummarizer(BaseSummarizer):
    """Summarization focused on creating bullet points"""

    def __init__(self):
        super().__init__("bullet_points")

    def summarize(self, text: str, config: SummarizationConfig) -> SummarizationResult:
        """Create bullet point summary"""
        result = SummarizationResult(config)
        result.method_used = "bullet_points"

        try:
            # First get extractive summary
            extractive = ExtractiveSummarizer()
            extractive_config = SummarizationConfig(
                method=SummarizationMethod.EXTRACTIVE,
                detail_level=config.detail_level,
                target_sentences=10,
                language=config.language
            )

            extractive_result = extractive.summarize(text, extractive_config)

            if not extractive_result.success:
                return extractive_result

            # Convert to bullet points
            sentences = extractive_result.important_sentences
            bullet_points = []

            for sentence in sentences[:8]:  # Limit to 8 bullet points
                # Clean and format as bullet point
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 10:  # Meaningful length
                    # Capitalize first letter
                    if clean_sentence:
                        clean_sentence = clean_sentence[0].upper() + clean_sentence[1:]
                    bullet_points.append(clean_sentence)

            # Join as bullet points
            summary = '\n'.join([f"• {point}" for point in bullet_points])

            # Update result
            result.summary = summary
            result.original_length = extractive_result.original_length
            result.summary_length = len(summary)
            result.compression_ratio = self._calculate_compression_ratio(text, summary)
            result.key_phrases = extractive_result.key_phrases
            result.important_sentences = bullet_points

            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"Bullet point summarization error: {e}")

        return result


class ConfigurableSummarizationEngine:
    """Main summarization engine with configurable options"""

    def __init__(self):
        self.summarizers = {
            SummarizationMethod.EXTRACTIVE: ExtractiveSummarizer(),
            SummarizationMethod.ABSTRACTIVE: AbstractiveSummarizer(),
            SummarizationMethod.HYBRID: HybridSummarizer(),
            SummarizationMethod.BULLET_POINTS: BulletPointSummarizer(),
        }

        self.content_processor = ContentProcessingPipeline()

    def summarize_document(self, file_path: str, config: SummarizationConfig = None) -> SummarizationResult:
        """
        Summarize a document with the given configuration

        Args:
            file_path: Path to the document
            config: Summarization configuration

        Returns:
            SummarizationResult with summary and metadata
        """
        if config is None:
            config = SummarizationConfig()

        start_time = datetime.now()

        try:
            # Step 1: Process document
            processed = self.content_processor.process_document(file_path)

            if not processed['success']:
                result = SummarizationResult(config)
                result.success = False
                result.error_message = f"Document processing failed: {processed.get('error', 'Unknown error')}"
                return result

            # Step 2: Extract text for summarization
            text = processed.get('preprocessed_content', processed.get('content', ''))

            if not text or len(text.strip()) < 50:
                result = SummarizationResult(config)
                result.success = False
                result.error_message = "Insufficient content for summarization"
                return result

            # Step 3: Get appropriate summarizer
            summarizer = self.summarizers.get(config.method, self.summarizers[SummarizationMethod.HYBRID])

            # Step 4: Perform summarization
            result = summarizer.summarize(text, config)

            # Step 5: Add processing metadata
            result.processing_time = (datetime.now() - start_time).total_seconds()
            result.metadata.update({
                'document_info': {
                    'file_path': file_path,
                    'file_type': processed.get('file_type', 'unknown'),
                    'original_length': len(processed.get('content', '')),
                    'preprocessed_length': len(text)
                },
                'processing_timestamp': start_time.isoformat(),
                'engine_version': '1.0'
            })

            return result

        except Exception as e:
            logger.error(f"Summarization engine error: {e}")
            result = SummarizationResult(config)
            result.success = False
            result.error_message = str(e)
            result.processing_time = (datetime.now() - start_time).total_seconds()
            return result

    def summarize_text(self, text: str, config: SummarizationConfig = None) -> SummarizationResult:
        """
        Summarize raw text with the given configuration

        Args:
            text: Text content to summarize
            config: Summarization configuration

        Returns:
            SummarizationResult with summary and metadata
        """
        if config is None:
            config = SummarizationConfig()

        start_time = datetime.now()

        try:
            # Validate input
            if not text or len(text.strip()) < 50:
                result = SummarizationResult(config)
                result.success = False
                result.error_message = "Insufficient text for summarization"
                return result

            # Get summarizer
            summarizer = self.summarizers.get(config.method, self.summarizers[SummarizationMethod.HYBRID])

            # Perform summarization
            result = summarizer.summarize(text, config)

            # Add metadata
            result.processing_time = (datetime.now() - start_time).total_seconds()
            result.metadata.update({
                'input_type': 'raw_text',
                'original_length': len(text),
                'processing_timestamp': start_time.isoformat(),
                'engine_version': '1.0'
            })

            return result

        except Exception as e:
            logger.error(f"Text summarization error: {e}")
            result = SummarizationResult(config)
            result.success = False
            result.error_message = str(e)
            result.processing_time = (datetime.now() - start_time).total_seconds()
            return result

    def get_available_methods(self) -> List[Dict[str, Any]]:
        """Get list of available summarization methods"""
        return [
            {
                'method': method.value,
                'name': method.value.replace('_', ' ').title(),
                'description': self._get_method_description(method)
            }
            for method in SummarizationMethod
        ]

    def get_available_detail_levels(self) -> List[Dict[str, Any]]:
        """Get list of available detail levels"""
        return [
            {
                'level': level.value,
                'name': level.value.title(),
                'ratio': self._get_detail_ratio(level),
                'description': self._get_detail_description(level)
            }
            for level in DetailLevel
        ]

    def _get_method_description(self, method: SummarizationMethod) -> str:
        """Get description for summarization method"""
        descriptions = {
            SummarizationMethod.EXTRACTIVE: "Selects and combines existing sentences from the document",
            SummarizationMethod.ABSTRACTIVE: "Generates new sentences that capture the document's meaning",
            SummarizationMethod.HYBRID: "Combines extractive and abstractive methods for best results",
            SummarizationMethod.BULLET_POINTS: "Creates concise bullet point summaries",
            SummarizationMethod.KEY_SENTENCES: "Extracts only the most important sentences",
            SummarizationMethod.PARAGRAPH: "Generates coherent paragraph summaries"
        }
        return descriptions.get(method, "Unknown method")

    def _get_detail_ratio(self, level: DetailLevel) -> float:
        """Get compression ratio for detail level"""
        ratios = {
            DetailLevel.BRIEF: 0.15,
            DetailLevel.CONCISE: 0.25,
            DetailLevel.DETAILED: 0.35,
            DetailLevel.COMPREHENSIVE: 0.50
        }
        return ratios.get(level, 0.25)

    def _get_detail_description(self, level: DetailLevel) -> str:
        """Get description for detail level"""
        descriptions = {
            DetailLevel.BRIEF: "Very short summary focusing on key points only",
            DetailLevel.CONCISE: "Balanced summary with essential information",
            DetailLevel.DETAILED: "Comprehensive summary with important details",
            DetailLevel.COMPREHENSIVE: "Thorough summary preserving most information"
        }
        return descriptions.get(level, "Unknown level")


# Global summarization engine instance
summarization_engine = ConfigurableSummarizationEngine()