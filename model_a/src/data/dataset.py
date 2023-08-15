from sklearn.datasets import load_iris
from typing import List, Text



def get_target_names() -> List[Text]:
    """Get target class names.
    Returns:
        List[Text]: list of target class names
    """
    return load_iris(as_frame=True).target_names.tolist()
