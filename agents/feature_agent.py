import logging

logger = logging.getLogger(__name__)

class FeatureAgent:
    """
    Base class for feature engineering agents.
    """
    def __init__(self):
        logger.info("FeatureAgent initialized")

    def engineer_features(self, symbol, data=None):
        """
        Abstract method to engineer features.
        """
        raise NotImplementedError("Subclasses must implement engineer_features")
