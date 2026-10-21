"""Optional native-dependency tests; no training, real data or network access."""
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NATIVE_AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "lightgbm", "pyarrow"))


@unittest.skipUnless(NATIVE_AVAILABLE, "requires requirements-reproduction.txt")
class NativeProtocolTests(unittest.TestCase):
    def run_native(self, code):
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_all_recorded_backbone_plans_parse_without_execution(self):
        self.run_native('''
import contextlib, importlib, io, json, sys
from unittest.mock import patch
sys.path[:0] = ["scripts", "reproduction/native"]
from train_backbone import main
for dataset in ("music4all_onion", "yambda500m"):
 for backbone in ("sasrec", "ssd4rec"):
  for seed in (2024, 2025, 2026):
   argv = ["plan", "--dataset", dataset, "--backbone", backbone, "--seed", str(seed),
    "--data-dir", "absent/prepared", "--prepared-manifest", "absent/prep.json",
    "--labels-manifest", "absent/labels.json", "--output", "absent/output"]
   if backbone == "ssd4rec":
    argv += ["--event-times", "absent/times.npy", "--timestamp-manifest", "absent/times.json"]
   stream = io.StringIO()
   with patch.object(sys, "argv", argv), contextlib.redirect_stdout(stream): main()
   plan = json.loads(stream.getvalue())
   assert not plan["execute"]
   module = importlib.import_module(plan["argv"][2])
   with patch.object(sys, "argv", [module.__file__, *plan["argv"][3:]]): parsed = module.parse_args()
   assert parsed.seed == seed
   assert parsed.objective == ("continuation_only" if backbone == "sasrec" else "oneshot_only")
''')

    def test_temporal_exclusion_is_not_silently_replaced_by_user_folds(self):
        self.run_native('''
import sys, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, "reproduction/native")
import arrive_framework_datasets as reg
reg.ONION_TEMPORAL_CV_ROWS = 2
sessions = np.asarray([[u, i*4, i*4+1, i*4+4, 60*86400, 0]
 for i,u in enumerate([1,1,2,2])])
data = SimpleNamespace(sessions=sessions, events=np.tile([3,4,5,6],4), row_ids=lambda split: np.arange(4))
im = {"configuration": {"cv_rule": reg.ONION_TEMPORAL_CV_RULE, "cv_limit": 2, "horizon": 64, "seed": 2024}}
with tempfile.TemporaryDirectory() as directory:
 root = Path(directory)
 index = Path(reg.bundle(root, "music4all_onion")["index"])
 assert reg.validate_train_cv_exclusion(root,"music4all_onion",index,im,data,np.asarray([1,3])) == "temporal_row_exclusion"
 for dataset,path,rows in (("music4all_onion",root/"custom",[1,3]), ("yambda500m",index,[1,3]),
                          ("music4all_onion",index,[0,2]), ("music4all_onion",index,[1,1])):
  try: reg.validate_train_cv_exclusion(root,dataset,path,im,data,np.asarray(rows))
  except ValueError: pass
  else: raise AssertionError((dataset,path,rows))
 assert reg.validate_train_cv_exclusion(root,"music4all_onion",root/"fold",im,data,np.asarray([0,1])) == "user_disjoint_fold"
''')

    def test_five_fold_assignment_covers_every_train_request_once(self):
        self.run_native('''
import sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,"reproduction/native")
from arrive_train_fold_coverage import fold_of_rows
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory)
 sessions=np.asarray([[u,i*4,i*4+1,i*4+4,35*86400,0] for i,u in enumerate(np.repeat(np.arange(10),2))])
 np.save(root/"sessions.npy",sessions)
 np.save(root/"events.npy",np.tile([3,4,5,6],20))
 np.save(root/"train_frequencies.npy",np.asarray([0,0,0,1,1,1,1]))
 rows,users,assignment=fold_of_rows(root,5,20260927)
 assert np.array_equal(rows,np.arange(20)) and set(assignment)==set(range(5))
 for user in np.unique(users): assert len(set(assignment[users==user]))==1
 assert np.array_equal(assignment,fold_of_rows(root,5,20260927)[2])
''')


if __name__ == "__main__":
    unittest.main()
