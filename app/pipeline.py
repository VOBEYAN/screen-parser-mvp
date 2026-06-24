from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from PIL import Image

from .component_library import ComponentLibrary
from .detectors import build_detector
from .graph_hierarchy import GraphHierarchyParser
from .hierarchy import HierarchyParser
from .matcher import ComponentMatcher
from .multimodal_classifier import MultimodalComponentClassifier
from .overlap import OverlapDetector
from .reporter import ReportWriter
from .schema_hydrator import build_ai_schema_components
from .visual_matcher import VisualReferenceLibrary


class ScreenParser:
    def __init__(
        self,
        catalog_path: str,
        artifact_root: str = "artifacts",
        yolo_model: Optional[str] = None,
        yolo_conf: Optional[float] = None,
        graph_model: Optional[str] = None,
        reference_library: Optional[str] = None,
        multimodal_classifier: bool = False,
        multimodal_model: Optional[str] = None,
        multimodal_base_url: Optional[str] = None,
        multimodal_api_key: Optional[str] = None,
    ):
        self.catalog_path = catalog_path
        self.artifact_root = artifact_root
        self.detector = build_detector(yolo_model, conf_threshold=yolo_conf)
        self.library = ComponentLibrary.from_catalog(catalog_path)
        self.reference_library_path = self._resolve_reference_library(reference_library)
        self.visual_library = VisualReferenceLibrary.from_path(self.reference_library_path)
        graph_model_path = Path(graph_model) if graph_model else None
        if graph_model_path and graph_model_path.exists():
            self.hierarchy = GraphHierarchyParser(str(graph_model_path))
            self.hierarchyMode = "graph_transformer"
        else:
            self.hierarchy = HierarchyParser()
            self.hierarchyMode = "rules_missing_graph_model" if graph_model else "rules"
        self.overlap = OverlapDetector()
        self.matcher = ComponentMatcher(self.library, self.visual_library)
        self.multimodal_classifier = MultimodalComponentClassifier.from_env(
            self.library,
            enabled=multimodal_classifier,
            model=multimodal_model,
            base_url=multimodal_base_url,
            api_key=multimodal_api_key,
            visual_library=self.visual_library,
        )
        self.reporter = ReportWriter(artifact_root)

    def parse(
        self,
        image_path: str,
        input_type: str = "design",
        top_k: int = 1,
        multimodal_model: Optional[str] = None,
        multimodal_base_url: Optional[str] = None,
        multimodal_api_key: Optional[str] = None,
        force_llm: Optional[bool] = None,
    ) -> Dict[str, str]:
        image = Image.open(image_path)
        image_size = image.size

        detections = self.detector.detect(image_path)
        nodes, relations = self.hierarchy.parse(detections, image_size[0], image_size[1])
        self.matcher.match_nodes(nodes, top_k=top_k, image_path=image_path)
        request_classifier = self.multimodal_classifier.for_request(
            model=multimodal_model,
            base_url=multimodal_base_url,
            api_key=multimodal_api_key,
            force_llm=force_llm,
        )
        classifier_summary = request_classifier.refine_nodes(nodes, image_path=image_path, top_k=top_k)
        ai_schema_components = build_ai_schema_components(nodes, self.library, classifier_summary=classifier_summary)
        overlaps = self.overlap.detect(nodes)

        artifacts = self.reporter.write(
            image_path=image_path,
            image_size=image_size,
            detections=detections,
            nodes=nodes,
            relations=relations,
            overlaps=overlaps,
            extras={
                "contentClassifier": classifier_summary,
                "aiSchemaComponents": ai_schema_components,
            },
        )
        artifacts["inputType"] = input_type
        artifacts["componentCount"] = str(max(0, len(nodes) - 1))
        artifacts["overlapCount"] = str(len(overlaps))
        artifacts["catalogPath"] = str(Path(self.catalog_path))
        artifacts["hierarchyMode"] = self.hierarchyMode
        artifacts["matcherMode"] = "visual_reference" if self.visual_library.enabled else "catalog_rules"
        artifacts["contentClassifierMode"] = str(classifier_summary.get("mode", "disabled"))
        artifacts["contentClassifierProcessed"] = str(classifier_summary.get("processedNodeCount", 0))
        artifacts["contentClassifierLlmCalls"] = str(classifier_summary.get("llmCallCount", 0))
        if self.reference_library_path:
            artifacts["referenceLibraryPath"] = str(Path(self.reference_library_path))
        return artifacts

    def _resolve_reference_library(self, reference_library: Optional[str]) -> Optional[str]:
        if reference_library:
            return reference_library
        default_path = Path(__file__).resolve().parents[1] / "data" / "component-reference"
        return str(default_path) if default_path.exists() else None
