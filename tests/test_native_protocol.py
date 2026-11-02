"""Optional native-dependency tests; no training, real data or network access."""
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NATIVE_AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "lightgbm", "pyarrow"))


@unittest.skipUnless(NATIVE_AVAILABLE, "requires requirements/native.txt")
class NativeProtocolTests(unittest.TestCase):
    def run_native(self, code):
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_joint_history_never_filters_by_current_cue(self):
        self.run_native('''
import sys
import numpy as np
sys.path.insert(0, "reproduction/native")
from run_arrive_full_rcce_control import history_only_anchors, JOINT_REMOVED_COLUMNS
from arrive_framework_datasets import source_policy
events=np.asarray([99,1,3,4,3,5,2,6,5,7,5,8])
session=np.asarray([1,1,10,12,35*86400,1])
assert history_only_anchors(events,session)==[7,5,6,3,4]
events[10]=7
assert history_only_anchors(events,session)==[7,5,6,3,4]
assert JOINT_REMOVED_COLUMNS==(3,5,12,13,14,16,19,21,24,27)
assert source_policy({})=="full" and source_policy({"source_policy":"joint-removal"})=="joint-removal"
try: source_policy({"source_policy":"other"})
except ValueError: pass
else: raise AssertionError("unknown policy admitted")
''')

    def test_ssd_query_adapter_uses_encode_batch_and_sealed_event_times(self):
        self.run_native('''
import json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,"reproduction/native")
from run_music4all_candidate_support_a1 import load_base_data, encode_base_states
from run_arrive_main_table_baseline import MainTableData
class BatchOnly:
 def encode_batch(self,batch):
  assert batch["event_times"][0,0].item()==12345
  return torch.ones((*batch["sequence"].shape,2))
 def gather_states(self,hidden,positions): return hidden[torch.arange(len(positions)),positions]
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory)
 np.save(root/"events.npy",np.asarray([3,4,5],np.int32))
 np.save(root/"sessions.npy",np.asarray([[1,0,1,3,35*86400,2]],np.int64))
 np.save(root/"train_frequencies.npy",np.asarray([0,0,0,1,1,1]))
 np.save(root/"event_times.npy",np.asarray([12345,12346,12347],np.int64))
 manifest=root/"base.json"
 manifest.write_text(json.dumps({"binding":{"method":"ssd4rec"}}))
 args=SimpleNamespace(data_dir=root,event_times=root/"event_times.npy",max_context=129,horizon=64,device="cpu")
 data=load_base_data(args,manifest)
 assert isinstance(data,MainTableData)
 states,_=encode_base_states(args,data,BatchOnly(),[0])
 assert states.shape==(1,2)
''')

    def test_reviewer_freeze_rejects_source_policy_drift(self):
        self.run_native('''
import json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,"reproduction/native")
import reviewer_freeze as freeze
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory)
 (root/"yeti_ndcg.cbm").write_bytes(b"synthetic model hash only")
 record=dict(schema="arrive.reviewer_ranker.v1",status="pass",dataset="music4all_onion",seed=2024,
  feature_count=31,feature_schema=freeze.feature_schema(),test_accessed=False,smoke=False,synthetic=False,
  cue_match_mode="true",selected_trees=220,model_sha256=freeze.digest(root/"yeti_ndcg.cbm"),source_policy="joint-removal")
 (root/"training_manifest.json").write_text(json.dumps(record))
 args=SimpleNamespace(model_dir=root,dataset="music4all_onion",seed=2024,source_policy="joint-removal")
 freeze.model_identity(args)
 args.source_policy="full"
 try: freeze.model_identity(args)
 except ValueError as exc: assert "source_policy" in str(exc)
 else: raise AssertionError("ranker policy drift admitted")
 args.source_policy="joint-removal"
 args.freeze_record=root/"freeze.json";args.test_manifest=root/"sealed.json";args.test_data=root/"sealed"
 args.legacy20k_retrieval=False
 args.test_manifest.write_text(json.dumps({"schema":freeze.TEST_SCHEMAS[args.dataset]}))
 frozen=dict(schema="arrive.reviewer_freeze.v1",status="frozen",dataset=args.dataset,seed=2024,
  model_sha256=record["model_sha256"],training_manifest_sha256=freeze.digest(root/"training_manifest.json"),
  freeze_helper_sha256=freeze.digest(Path(freeze.__file__)),code_sha256={"toy":"bound"},serving_inputs={"toy":"bound"},
  test_manifest_sha256=freeze.digest(args.test_manifest),test_data=str(args.test_data.resolve()),retrieval_contract="direct",source_policy="joint-removal")
 args.freeze_record.write_text(json.dumps(frozen))
 with patch.object(freeze,"serving_inputs",return_value={"toy":"bound"}),patch.object(freeze,"code_bindings",return_value={"toy":"bound"}):
  freeze.validate_before_export(args)
  frozen["source_policy"]="full";args.freeze_record.write_text(json.dumps(frozen))
  try: freeze.validate_before_export(args)
  except ValueError as exc: assert "source_policy" in str(exc)
  else: raise AssertionError("freeze policy drift admitted")
''')

    def test_framework_export_preserves_ssd_adapter_and_sealed_timestamps(self):
        self.run_native('''
import json,sys,tempfile,shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
sys.path[:0]=["scripts","reproduction/native"]
from smoke_reproduction import fixture, save_json
import run_arrive_full_rcce_control as native
import reviewer_freeze
import evaluate_arrive_frozen_test
from arrive_framework_datasets import bundle, feature_input_paths, hashes
class BatchOnly:
 def __init__(self,expected):
  self.expected=expected;self.item_embedding=torch.nn.Embedding(259,8)
 def encode_batch(self,batch):
  assert batch["event_times"][0,0].item()==self.expected
  return self.item_embedding(batch["sequence"])
 def gather_states(self,hidden,positions): return hidden[torch.arange(len(positions)),positions]
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);data,_,_,index,checkpoints=fixture(root)
 ds=bundle(root,"music4all_onion")
 times=Path(ds["event_times"]);times.parent.mkdir(parents=True)
 np.save(times,np.arange(len(np.load(data/"events.npy")))+100)
 bm=checkpoints["validation"].parent/"training_manifest.json"
 record=json.loads(bm.read_text());record.pop("baseline");record["binding"]["method"]="ssd4rec";save_json(bm,record)
 sealed=root/"sealed";shutil.copytree(data,sealed)
 sessions=np.load(sealed/"sessions.npy");sessions[sessions[:,5]==1,5]=2;np.save(sealed/"sessions.npy",sessions)
 np.save(sealed/"event_times.npy",np.arange(len(np.load(sealed/"events.npy")))+1000)
 test_manifest=root/"sealed.json";save_json(test_manifest,{"synthetic":True})
 frozen=root/"freeze.json";save_json(frozen,{"synthetic":True})
 inputs=hashes(feature_input_paths(ds,index,"full"));inputs.update(base_checkpoint=native.digest(checkpoints["validation"]),base_manifest=native.digest(bm))
 for surface,expected in (("validation",132),("test",1032)):
  args=SimpleNamespace(project=root,dataset="music4all_onion",seed=2024,surface=surface,
   output=root/("export_"+surface),base_manifest=bm,base_checkpoint=checkpoints["validation"],
   index_dir=index,source_policy="joint-removal",legacy20k_retrieval=False,device="cpu",batch_size=128,
   reviewer_export_interface=True,test_data=sealed,test_manifest=test_manifest,freeze_record=frozen)
  model=BatchOnly(expected)
  with patch.object(native,"load_standard_base",return_value=(model,None)),\
       patch.object(native,"base_topk",side_effect=lambda m,h,s,k:torch.arange(3,203).expand(len(h),200)),\
       patch.object(reviewer_freeze,"validate_before_export",return_value={"source_policy":"joint-removal","retrieval_contract":"direct","serving_inputs":inputs}),\
       patch.object(evaluate_arrive_frozen_test,"validate_test",return_value={}):
   native.framework_export(args)
  assert json.loads((args.output/"EXPORT_MANIFEST.json").read_text())["source_policy"]=="joint-removal"
''')

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
