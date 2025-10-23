from src.features import FeatureExtractor, CleanFeatureExtractor
from src.utils.config import RecallConfig

class FeaturesPipeline:
    def __init__(self, config: RecallConfig) -> None:
        self.config = config
        self.feature_extractor = CleanFeatureExtractor(config)

    def extract(self):
        self.feature_extractor.extract_features()

if __name__ == "__main__":
    config = RecallConfig()
    pipeline = FeaturesPipeline(config)
    pipeline.extract()
    
    # python -m src.pipeline.features_pipeline 