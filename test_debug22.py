from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager

semantics = SemanticsManager()
engine = ASTEngine(semantics, RuntimeConfig(source_framework="numpy", target_framework="torch", strict_mode=False))
print(engine.run("import numpy as np\ndef f(x):\n    return np.zeros_like(x)").code)
