"""Natasha-based entity extraction implementation."""

import logging
from typing import Sequence

from ner_controller.domain.interfaces.entity_extractor_interface import EntityExtractorInterface
from ner_controller.domain.services.levenshtein_utils import deduplicate_entities
from ner_controller.infrastructure.ner.configs.natasha_entity_extractor_config import (
    NatashaEntityExtractorConfig,
)

logger = logging.getLogger(__name__)


class NatashaEntityExtractor(EntityExtractorInterface):
    """Extract entities using Natasha (Slovnet NER)."""

    _PERSON_ALIASES = {"PERSON", "PER", "PERSONA"}
    _ORG_ALIASES = {"ORGANIZATION", "ORG", "COMPANY"}
    _LOC_ALIASES = {"LOCATION", "LOC", "GPE", "CITY", "COUNTRY"}

    def __init__(self, config: NatashaEntityExtractorConfig) -> None:
        """Initialize extractor with Natasha model configuration."""
        self._config = config
        self._ner_model = None
        self._disabled = False

    def extract(self, text: str, entity_types: Sequence[str]) -> Sequence[str]:
        """Extract entities from text using Natasha."""
        if not text or not entity_types:
            return []

        target_labels = self._resolve_target_labels(entity_types)
        if not target_labels:
            return []

        if self._disabled:
            return []

        try:
            ner_model = self._load_ner_model()
        except FileNotFoundError as exc:
            logger.warning("Natasha extractor disabled: %s", exc)
            self._disabled = True
            return []
        markup = ner_model(text)

        entities: list[str] = []
        for span in getattr(markup, "spans", []):
            span_label = str(getattr(span, "type", "")).upper()
            if span_label not in target_labels:
                continue

            start = int(getattr(span, "start", -1))
            stop = int(getattr(span, "stop", -1))
            if start < 0 or stop <= start:
                continue

            entity_text = text[start:stop].strip()
            if entity_text:
                entities.append(entity_text)

        return deduplicate_entities(entities, threshold=2)

    def _resolve_target_labels(self, entity_types: Sequence[str]) -> set[str]:
        """Map API entity types to Natasha labels."""
        mapped_labels: set[str] = set()

        for entity_type in entity_types:
            normalized = str(entity_type).strip().upper()
            if normalized in self._PERSON_ALIASES:
                mapped_labels.add("PER")
            if normalized in self._ORG_ALIASES:
                mapped_labels.add("ORG")
            if normalized in self._LOC_ALIASES:
                mapped_labels.add("LOC")

        return mapped_labels

    def _load_ner_model(self):
        """Load Natasha models from local files only."""
        if self._ner_model is not None:
            return self._ner_model

        from navec import Navec
        from slovnet import NER

        navec_path = self._config.navec_model_path()
        ner_path = self._config.ner_model_path()

        if not navec_path.exists():
            raise FileNotFoundError(
                f"Missing Natasha Navec model at {navec_path}. Run download_model.py first."
            )
        if not ner_path.exists():
            raise FileNotFoundError(
                f"Missing Natasha NER model at {ner_path}. Run download_model.py first."
            )

        logger.info("Loading Natasha models from local cache: %s, %s", navec_path, ner_path)
        navec = Navec.load(str(navec_path))
        ner_model = NER.load(str(ner_path))
        ner_model.navec(navec)
        self._ner_model = ner_model
        return self._ner_model
