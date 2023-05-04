from typing import Any

import numpy as np
import xgboost as xgb


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):
    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        pass

    def preprocess(self, body: dict, state: dict, collect_custom_statistics_fn=None) -> Any:
        print(f"BODY HERE: {body}")
        pixels = body.get('pixels')
        return xgb.DMatrix([pixels])

    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> dict:
        # post process the data returned from the model inference engine
        # data is the return value from model.predict we will put is inside a return value as Y
        return dict(y=[round(y) for y in data.tolist()])