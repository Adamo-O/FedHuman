import os
from comet_ml import Experiment,ExistingExperiment

class CometLogger:
    _experiment = None  # Static/class variable

    @staticmethod
    def initialize_comet():
        if CometLogger._experiment is None:
            CometLogger._experiment = Experiment(
                api_key=os.environ['COMET_API_KEY'],  
                project_name="Auxilary-Debugging",
                workspace="pinon"
            )
        return CometLogger._experiment