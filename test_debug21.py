from ml_switcheroo.semantics.manager import SemanticsManager

sm = SemanticsManager()
print(sm.get_definition("numpy.zeros"))
print(sm.get_definition("numpy.ones"))
print(sm.get_definition("numpy.ones_like"))
