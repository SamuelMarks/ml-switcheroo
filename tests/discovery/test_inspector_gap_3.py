from ml_switcheroo.discovery.inspector import ApiInspector


def test_recurse_runtime_visited():
  inspector = ApiInspector()
  obj = object()
  visited = {id(obj)}
  catalog = {}
  inspector._recurse_runtime(obj, "dummy", catalog, visited, set())
  assert catalog == {}


def test_recurse_runtime_getmembers_exception():
  inspector = ApiInspector()

  class BadDir:
    def __dir__(self):
      raise Exception("Fail dir")

  obj = BadDir()
  visited = set()
  catalog = {}
  inspector._recurse_runtime(obj, "dummy", catalog, visited, set())
  assert catalog == {}
